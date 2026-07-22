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

/// Object-store connection parameters. A service assembles this from its own (prefixed)
/// environment and hands it to [`Store::from_s3_config`]; the store never reads the environment,
/// so each service owns its config.
#[derive(Clone, Debug)]
pub struct S3Config {
    pub bucket: String,
    /// `None` selects real AWS S3; `Some` points at MinIO / any S3-compatible endpoint (dev).
    pub endpoint: Option<String>,
    pub access_key: String,
    pub secret_key: String,
    pub region: String,
}

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
    /// Optional NVMe read cache for immutable objects. This is the warm path: a query node
    /// materializes generation files here once and serves subsequent reads from local disk
    /// (INV-4 — the cache only ever changes latency, never results).
    cache: Option<Arc<crate::cache::DiskCache>>,
    /// Optional lifetime op accounting, for the cost model in the performance suite.
    store_metrics: Option<Arc<crate::metrics::StoreMetrics>>,
}

impl Store {
    pub fn new(inner: Arc<dyn ObjectStore>) -> Self {
        Self {
            inner,
            cache: None,
            store_metrics: None,
        }
    }

    /// Attach lifetime op accounting; every GET/PUT/LIST/DELETE is then counted.
    pub fn with_store_metrics(mut self, m: Arc<crate::metrics::StoreMetrics>) -> Self {
        self.store_metrics = Some(m);
        self
    }

    pub fn store_metrics(&self) -> Option<&Arc<crate::metrics::StoreMetrics>> {
        self.store_metrics.as_ref()
    }

    /// Attach an NVMe read cache. Immutable reads ([`Store::get_immutable`]) are served
    /// from it on a hit and admitted on a miss.
    pub fn with_cache(mut self, cache: Arc<crate::cache::DiskCache>) -> Self {
        self.cache = Some(cache);
        self
    }

    pub fn cache(&self) -> Option<&Arc<crate::cache::DiskCache>> {
        self.cache.as_ref()
    }

    /// An in-memory store, for fast unit tests. `object_store`'s `InMemory` implements the
    /// same conditional-put semantics (`If-None-Match` / `If-Match`) as S3, so it is a
    /// faithful stand-in for the S3 *interface* without a network — unlike a local
    /// filesystem, which cannot do conditional updates at all. It backs no deployment.
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

    /// Build a store from an [`S3Config`] the caller assembled. The store never reads the
    /// environment itself — each service parses its own (prefixed) config and passes it here.
    pub fn from_s3_config(cfg: &S3Config) -> Result<Self> {
        Self::s3(
            &cfg.bucket,
            cfg.endpoint.as_deref(),
            &cfg.access_key,
            &cfg.secret_key,
            &cfg.region,
        )
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
        if let Some(m) = &self.store_metrics {
            m.record_get(bytes.len() as u64);
        }
        Ok(Versioned { bytes, etag })
    }

    /// Read an *immutable* object, through the NVMe cache if one is attached.
    ///
    /// Generation files live under a per-attempt nonce prefix, so their path uniquely
    /// identifies their bytes for all time — the cache is keyed by path alone, no etag
    /// revalidation needed. A hit is a local-disk read counted as a cache hit (zero
    /// roundtrips); a miss fetches once, admits to the cache, and counts a roundtrip.
    pub async fn get_immutable(
        &self,
        path: &str,
        ctx: Option<(&QueryMetrics, usize)>,
    ) -> Result<bytes::Bytes> {
        if let Some(cache) = &self.cache {
            let key = crate::cache::CacheKey::new("", path, "immutable");
            if let Some(bytes) = cache.get(&key) {
                if let Some((metrics, _)) = ctx {
                    metrics.record_cache_hit();
                }
                return Ok(bytes);
            }
            let versioned = self.get(path, ctx).await?;
            if let Some((metrics, _)) = ctx {
                metrics.record_cache_miss();
            }
            cache.put(key, versioned.bytes.clone());
            return Ok(versioned.bytes);
        }
        Ok(self.get(path, ctx).await?.bytes)
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
        if let Some(m) = &self.store_metrics {
            m.record_get(bytes.len() as u64);
        }
        Ok(bytes)
    }

    /// Read several ranges of one *immutable* object, through the NVMe cache when attached.
    ///
    /// Each `(path, range)` names immutable bytes (SSTable blocks under a nonce prefix), so
    /// the cache is keyed by path+range. Ranges already resident are served locally; only
    /// the misses go to S3, coalesced into a single request. This is what warms the graph
    /// arm's `radj`/`pk` block reads.
    pub async fn get_ranges(
        &self,
        path: &str,
        ranges: &[std::ops::Range<usize>],
        ctx: Option<(&QueryMetrics, usize)>,
    ) -> Result<Vec<Bytes>> {
        if ranges.is_empty() {
            return Ok(Vec::new());
        }

        let range_key = |r: &std::ops::Range<usize>| format!("{path}#{}-{}", r.start, r.end);

        // Cache lookup per range; collect the misses to fetch.
        let mut out: Vec<Option<Bytes>> = vec![None; ranges.len()];
        let mut miss_idx: Vec<usize> = Vec::new();
        if let Some(cache) = &self.cache {
            for (i, r) in ranges.iter().enumerate() {
                let key = crate::cache::CacheKey::new("", &range_key(r), "immutable");
                if let Some(b) = cache.get(&key) {
                    out[i] = Some(b);
                    if let Some((metrics, _)) = ctx {
                        metrics.record_cache_hit();
                    }
                } else {
                    miss_idx.push(i);
                }
            }
        } else {
            miss_idx = (0..ranges.len()).collect();
        }

        if !miss_idx.is_empty() {
            let miss_ranges: Vec<std::ops::Range<usize>> =
                miss_idx.iter().map(|&i| ranges[i].clone()).collect();
            let start = Instant::now();
            let fetched = self
                .inner
                .get_ranges(&Path::from(path), &miss_ranges)
                .await
                .map_err(|e| match e {
                    object_store::Error::NotFound { .. } => Error::NotFound(path.to_string()),
                    other => other.into(),
                })?;
            let total: u64 = fetched.iter().map(|b| b.len() as u64).sum();
            if let Some((metrics, rt)) = ctx {
                metrics.record_request(rt, total, start.elapsed());
                metrics.record_cache_miss();
            }
            if let Some(m) = &self.store_metrics {
                m.record_get(total);
            }
            for (&i, bytes) in miss_idx.iter().zip(fetched) {
                if let Some(cache) = &self.cache {
                    let key = crate::cache::CacheKey::new("", &range_key(&ranges[i]), "immutable");
                    cache.put(key, bytes.clone());
                }
                out[i] = Some(bytes);
            }
        }

        Ok(out.into_iter().map(|b| b.unwrap_or_default()).collect())
    }

    /// Unconditional write. Only legal for immutable objects, whose paths are unique by
    /// construction (INV-2) — never for the manifest or a WAL slot.
    ///
    /// Deliberately does **not** populate the read cache; see [`Store::put_admitting`].
    pub async fn put(&self, path: &str, bytes: Vec<u8>) -> Result<Option<Etag>> {
        self.put_bytes(path, Bytes::from(bytes)).await
    }

    async fn put_bytes(&self, path: &str, bytes: Bytes) -> Result<Option<Etag>> {
        let len = bytes.len() as u64;
        let result = self
            .inner
            .put(&Path::from(path), PutPayload::from_bytes(bytes))
            .await?;
        if let Some(m) = &self.store_metrics {
            m.record_put(len);
        }
        Ok(result.e_tag.map(Etag))
    }

    /// Unconditional write that also **admits the bytes to the read cache**, so the writer
    /// does not have to re-fetch what it just wrote.
    ///
    /// This exists for the inline-fold path: since `wait_for_index`, a `serve` replica can
    /// fold a generation itself, writing every cluster, vector block and SSTable — and then
    /// miss the cache on all of them when the next query arrives, having had the bytes in
    /// hand and dropped them. (The background `index` deployment is a different process, so
    /// only the inline path is affected.)
    ///
    /// It is opt-in on purpose, and [`Store::put`] stays read-through. A fold writes the
    /// *whole* generation, and no imminent query probes most of it, so admitting all of it
    /// trades a warm working set for cold data — and because the cache is a FIFO ring, a
    /// single fold that writes more than the disk budget laps the ring and leaves nothing
    /// behind but its own tail. Which call sites should use this — plausibly only the small
    /// objects every query is certain to read (centroids, footers, `pk.idx`) and not the
    /// per-cluster bulk — is a follow-up decision to make against the hit-ratio numbers,
    /// not a default. Nothing in the tree calls it yet.
    pub async fn put_admitting(&self, path: &str, bytes: Vec<u8>) -> Result<Option<Etag>> {
        let bytes = Bytes::from(bytes);
        let etag = self.put_bytes(path, bytes.clone()).await?;
        if let Some(cache) = &self.cache {
            // Exactly the key `get_immutable` will look up. Anything else would admit bytes
            // no read can ever find, which is worse than not admitting at all.
            cache.put(crate::cache::CacheKey::new("", path, "immutable"), bytes);
        }
        Ok(etag)
    }

    /// Create an object only if it does not already exist (`If-None-Match: *`).
    ///
    /// This is how a WAL sequence number is claimed: concurrent writers racing for the
    /// same slot all issue this, exactly one wins, and the losers get
    /// [`Error::AlreadyExists`] and retry at the next sequence.
    pub async fn put_if_absent(&self, path: &str, bytes: Vec<u8>) -> Result<Option<Etag>> {
        let len = bytes.len() as u64;
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
            Ok(r) => {
                if let Some(m) = &self.store_metrics {
                    m.record_put(len);
                }
                Ok(r.e_tag.map(Etag))
            }
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
        let len = bytes.len() as u64;
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
            Ok(r) => {
                if let Some(m) = &self.store_metrics {
                    m.record_put(len);
                }
                Ok(r.e_tag.map(Etag))
            }
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
        if let Some(m) = &self.store_metrics {
            m.record_delete();
        }
        match self.inner.delete(&Path::from(path)).await {
            Ok(()) => Ok(()),
            // Deletion is idempotent: GC may race another node doing the same work.
            Err(object_store::Error::NotFound { .. }) => Ok(()),
            Err(e) => Err(e.into()),
        }
    }

    /// List object paths under a prefix, sorted ascending.
    pub async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        if let Some(m) = &self.store_metrics {
            m.record_list();
        }
        let mut paths: Vec<String> = self
            .inner
            .list(Some(&Path::from(prefix)))
            .map_ok(|meta| meta.location.to_string())
            .try_collect()
            .await?;
        paths.sort();
        Ok(paths)
    }

    /// List object paths under a prefix together with each object's size, sorted ascending.
    ///
    /// The size comes from the listing itself, so surfacing it costs the same one LIST as
    /// [`Store::list`] — the alternative, a HEAD per object, would turn a log view into N
    /// roundtrips.
    pub async fn list_with_size(&self, prefix: &str) -> Result<Vec<(String, u64)>> {
        if let Some(m) = &self.store_metrics {
            m.record_list();
        }
        let mut out: Vec<(String, u64)> = self
            .inner
            .list(Some(&Path::from(prefix)))
            .map_ok(|meta| (meta.location.to_string(), meta.size as u64))
            .try_collect()
            .await?;
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    /// List object paths under a prefix together with each object's last-modified time, so
    /// GC can apply an age-based grace window before deleting.
    pub async fn list_with_age(
        &self,
        prefix: &str,
    ) -> Result<Vec<(String, chrono::DateTime<chrono::Utc>)>> {
        if let Some(m) = &self.store_metrics {
            m.record_list();
        }
        let mut out: Vec<(String, chrono::DateTime<chrono::Utc>)> = self
            .inner
            .list(Some(&Path::from(prefix)))
            .map_ok(|meta| (meta.location.to_string(), meta.last_modified))
            .try_collect()
            .await?;
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
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
