//! The instrumented object-storage client.
//!
//! Every read and write in the critical path goes through here so that (a) roundtrips are
//! counted against the budget and (b) coordination uses conditional writes rather than
//! locks (INV-3).

use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use futures::TryStreamExt;
use object_store::path::Path;
use object_store::{
    Attributes, ObjectStore, PutMode, PutOptions, PutPayload, TagSet, UpdateVersion,
};

use crate::metrics::QueryMetrics;
use crate::{Error, Result};

/// An object's version token, used to make a later write conditional on nothing having
/// changed in between.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Etag(pub String);

/// An object read together with the version it was read at.
#[derive(Clone, Debug)]
pub struct Versioned {
    pub bytes: Bytes,
    pub etag: Option<Etag>,
}

/// Handle to the object store backing one deployment.
#[derive(Clone)]
pub struct Store {
    inner: Arc<dyn ObjectStore>,
}

impl Store {
    pub fn new(inner: Arc<dyn ObjectStore>) -> Self {
        Self { inner }
    }

    /// A store backed by the local filesystem. Used by unit tests and by the latency-shim
    /// rig; `LocalFileSystem` implements the same conditional-write semantics MinIO does.
    pub fn local(root: impl AsRef<std::path::Path>) -> Result<Self> {
        let fs = object_store::local::LocalFileSystem::new_with_prefix(root)?;
        Ok(Self::new(Arc::new(fs)))
    }

    pub fn in_memory() -> Self {
        Self::new(Arc::new(object_store::memory::InMemory::new()))
    }

    /// A store backed by S3 or an S3-compatible endpoint (MinIO in dev).
    pub fn s3(
        bucket: &str,
        endpoint: Option<&str>,
        access_key: &str,
        secret_key: &str,
        region: &str,
    ) -> Result<Self> {
        let mut builder = object_store::aws::AmazonS3Builder::new()
            .with_bucket_name(bucket)
            .with_access_key_id(access_key)
            .with_secret_access_key(secret_key)
            .with_region(region)
            // MinIO is served over plain HTTP locally and addresses buckets by path.
            .with_conditional_put(object_store::aws::S3ConditionalPut::ETagMatch);
        if let Some(ep) = endpoint {
            builder = builder
                .with_endpoint(ep)
                .with_allow_http(true)
                .with_virtual_hosted_style_request(false);
        }
        Ok(Self::new(Arc::new(builder.build()?)))
    }

    /// Read an object whole, recording the request against a query's budget.
    pub async fn get(
        &self,
        path: &str,
        ctx: Option<(&QueryMetrics, usize)>,
    ) -> Result<Versioned> {
        let start = Instant::now();
        let result = self.inner.get(&Path::from(path)).await;
        let result = match result {
            Ok(r) => r,
            Err(object_store::Error::NotFound { .. }) => {
                return Err(Error::NotFound(path.to_string()))
            }
            Err(e) => return Err(e.into()),
        };
        let etag = result.meta.e_tag.clone().map(Etag);
        let bytes = result.bytes().await?;
        if let Some((metrics, rt)) = ctx {
            metrics.record_request(rt, bytes.len() as u64, start.elapsed());
        }
        Ok(Versioned { bytes, etag })
    }

    /// Read a byte range. This is the workhorse of the warm path: the hotcache and
    /// sparse indexes exist so that a query can turn "which bytes do I need" into a
    /// handful of coalesced ranged GETs.
    pub async fn get_range(
        &self,
        path: &str,
        range: std::ops::Range<usize>,
        ctx: Option<(&QueryMetrics, usize)>,
    ) -> Result<Bytes> {
        let start = Instant::now();
        let bytes = self
            .inner
            .get_range(&Path::from(path), range)
            .await
            .map_err(|e| match e {
                object_store::Error::NotFound { .. } => Error::NotFound(path.to_string()),
                other => other.into(),
            })?;
        if let Some((metrics, rt)) = ctx {
            metrics.record_request(rt, bytes.len() as u64, start.elapsed());
        }
        Ok(bytes)
    }

    /// Read several ranges of one object. The store coalesces adjacent ranges, so this
    /// stays a single roundtrip even when the ranges are not contiguous.
    pub async fn get_ranges(
        &self,
        path: &str,
        ranges: &[std::ops::Range<usize>],
        ctx: Option<(&QueryMetrics, usize)>,
    ) -> Result<Vec<Bytes>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }
        let start = Instant::now();
        let parts = self
            .inner
            .get_ranges(&Path::from(path), ranges)
            .await
            .map_err(|e| match e {
                object_store::Error::NotFound { .. } => Error::NotFound(path.to_string()),
                other => other.into(),
            })?;
        if let Some((metrics, rt)) = ctx {
            let total: u64 = parts.iter().map(|b| b.len() as u64).sum();
            metrics.record_request(rt, total, start.elapsed());
        }
        Ok(parts)
    }

    /// Unconditional write. Only legal for immutable objects, whose paths are unique by
    /// construction (INV-2) — never for the manifest or a WAL slot.
    pub async fn put(&self, path: &str, bytes: Vec<u8>) -> Result<Option<Etag>> {
        let result = self
            .inner
            .put(&Path::from(path), PutPayload::from_bytes(Bytes::from(bytes)))
            .await?;
        Ok(result.e_tag.map(Etag))
    }

    /// Create an object only if it does not already exist (`If-None-Match: *`).
    ///
    /// This is how a WAL sequence number is claimed: concurrent writers racing for the
    /// same slot all issue this, exactly one wins, and the losers get
    /// [`Error::AlreadyExists`] and retry at the next sequence.
    pub async fn put_if_absent(&self, path: &str, bytes: Vec<u8>) -> Result<Option<Etag>> {
        let opts = PutOptions {
            mode: PutMode::Create,
            tags: TagSet::default(),
            attributes: Attributes::default(),
        };
        match self
            .inner
            .put_opts(
                &Path::from(path),
                PutPayload::from_bytes(Bytes::from(bytes)),
                opts,
            )
            .await
        {
            Ok(r) => Ok(r.e_tag.map(Etag)),
            Err(object_store::Error::AlreadyExists { .. }) => {
                Err(Error::AlreadyExists(path.to_string()))
            }
            Err(e) => Err(e.into()),
        }
    }

    /// Replace an object only if it is still at the expected version (`If-Match`).
    ///
    /// The manifest swap. A caller that loses the race must re-read and merge rather than
    /// blindly retrying, or it would silently drop the winner's generation (SPEC §3.1).
    pub async fn cas_swap(
        &self,
        path: &str,
        expected: &Etag,
        bytes: Vec<u8>,
    ) -> Result<Option<Etag>> {
        let opts = PutOptions {
            mode: PutMode::Update(UpdateVersion {
                e_tag: Some(expected.0.clone()),
                version: None,
            }),
            tags: TagSet::default(),
            attributes: Attributes::default(),
        };
        match self
            .inner
            .put_opts(
                &Path::from(path),
                PutPayload::from_bytes(Bytes::from(bytes)),
                opts,
            )
            .await
        {
            Ok(r) => Ok(r.e_tag.map(Etag)),
            Err(object_store::Error::Precondition { .. }) => {
                Err(Error::CasConflict(path.to_string()))
            }
            Err(object_store::Error::NotModified { .. }) => {
                Err(Error::CasConflict(path.to_string()))
            }
            Err(e) => Err(e.into()),
        }
    }

    pub async fn delete(&self, path: &str) -> Result<()> {
        match self.inner.delete(&Path::from(path)).await {
            Ok(()) => Ok(()),
            // Deletion is idempotent: GC may race another node doing the same work.
            Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    /// List object paths under a prefix, sorted ascending.
    pub async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let mut paths: Vec<String> = self
            .inner
            .list(Some(&Path::from(prefix)))
            .map_ok(|meta| meta.location.to_string())
            .try_collect()
            .await?;
        paths.sort();
        Ok(paths)
    }

    /// Size of an object without fetching it.
    pub async fn head(&self, path: &str) -> Result<u64> {
        match self.inner.head(&Path::from(path)).await {
            Ok(meta) => Ok(meta.size as u64),
            Err(object_store::Error::NotFound { .. }) => Err(Error::NotFound(path.to_string())),
            Err(e) => Err(e.into()),
        }
    }

    pub fn exists(&self, path: &str) -> impl std::future::Future<Output = Result<bool>> + '_ {
        let path = path.to_string();
        async move {
            match self.head(&path).await {
                Ok(_) => Ok(true),
                Err(Error::NotFound(_)) => Ok(false),
                Err(e) => Err(e),
            }
        }
    }
}
