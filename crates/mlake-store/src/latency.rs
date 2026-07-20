//! Latency-injecting `ObjectStore` wrapper.
//!
//! Local storage answers in microseconds, which would make every roundtrip-budget
//! benchmark meaningless — a design that issues twenty roundtrips still looks fast. This
//! wrapper imposes the S3-like timings from SPEC §10.4 so CI measures the shape of the
//! access pattern rather than the speed of the developer's SSD.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::BoxStream;
use object_store::path::Path;
use object_store::{
    GetOptions, GetResult, ListResult, MultipartUpload, ObjectMeta, ObjectStore, PutMultipartOpts,
    PutOptions, PutPayload, PutResult, Result as OsResult,
};

/// Simulated timings. Defaults match SPEC §10.4's CI profile.
#[derive(Clone, Copy, Debug)]
pub struct LatencyProfile {
    pub get: Duration,
    pub put: Duration,
    pub list: Duration,
}

impl Default for LatencyProfile {
    fn default() -> Self {
        Self {
            get: Duration::from_millis(80),
            put: Duration::from_millis(120),
            list: Duration::from_millis(80),
        }
    }
}

impl LatencyProfile {
    /// No delay — for tests that care about behaviour rather than timing.
    pub fn none() -> Self {
        Self {
            get: Duration::ZERO,
            put: Duration::ZERO,
            list: Duration::ZERO,
        }
    }
}

/// Wraps any `ObjectStore`, delaying each operation by the configured amount.
pub struct LatencyStore {
    inner: Arc<dyn ObjectStore>,
    profile: LatencyProfile,
}

impl LatencyStore {
    pub fn new(inner: Arc<dyn ObjectStore>, profile: LatencyProfile) -> Self {
        Self { inner, profile }
    }
}

impl fmt::Display for LatencyStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LatencyStore({})", self.inner)
    }
}

impl fmt::Debug for LatencyStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LatencyStore({:?})", self.profile)
    }
}

#[async_trait]
impl ObjectStore for LatencyStore {
    async fn put_opts(
        &self,
        location: &Path,
        payload: PutPayload,
        opts: PutOptions,
    ) -> OsResult<PutResult> {
        tokio::time::sleep(self.profile.put).await;
        self.inner.put_opts(location, payload, opts).await
    }

    async fn put_multipart_opts(
        &self,
        location: &Path,
        opts: PutMultipartOpts,
    ) -> OsResult<Box<dyn MultipartUpload>> {
        tokio::time::sleep(self.profile.put).await;
        self.inner.put_multipart_opts(location, opts).await
    }

    async fn get_opts(&self, location: &Path, options: GetOptions) -> OsResult<GetResult> {
        tokio::time::sleep(self.profile.get).await;
        self.inner.get_opts(location, options).await
    }

    async fn get_ranges(&self, location: &Path, ranges: &[Range<usize>]) -> OsResult<Vec<Bytes>> {
        // Deliberately one delay for the whole call: coalesced ranges are one roundtrip,
        // which is exactly the property the budget is meant to reward.
        tokio::time::sleep(self.profile.get).await;
        self.inner.get_ranges(location, ranges).await
    }

    async fn head(&self, location: &Path) -> OsResult<ObjectMeta> {
        tokio::time::sleep(self.profile.get).await;
        self.inner.head(location).await
    }

    async fn delete(&self, location: &Path) -> OsResult<()> {
        tokio::time::sleep(self.profile.put).await;
        self.inner.delete(location).await
    }

    fn list(&self, prefix: Option<&Path>) -> BoxStream<'_, OsResult<ObjectMeta>> {
        self.inner.list(prefix)
    }

    async fn list_with_delimiter(&self, prefix: Option<&Path>) -> OsResult<ListResult> {
        tokio::time::sleep(self.profile.list).await;
        self.inner.list_with_delimiter(prefix).await
    }

    async fn copy(&self, from: &Path, to: &Path) -> OsResult<()> {
        tokio::time::sleep(self.profile.put).await;
        self.inner.copy(from, to).await
    }

    async fn copy_if_not_exists(&self, from: &Path, to: &Path) -> OsResult<()> {
        tokio::time::sleep(self.profile.put).await;
        self.inner.copy_if_not_exists(from, to).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Store;
    use std::time::Instant;

    #[tokio::test]
    async fn injects_the_configured_delay() {
        let inner = Arc::new(object_store::memory::InMemory::new());
        let store = Store::new(Arc::new(LatencyStore::new(
            inner,
            LatencyProfile {
                get: Duration::from_millis(50),
                put: Duration::from_millis(50),
                list: Duration::ZERO,
            },
        )));
        store.put("a", b"x".to_vec()).await.unwrap();
        let start = Instant::now();
        store.get("a", None).await.unwrap();
        assert!(start.elapsed() >= Duration::from_millis(45));
    }

    #[tokio::test]
    async fn coalesced_ranges_cost_one_delay_not_n() {
        let inner = Arc::new(object_store::memory::InMemory::new());
        let store = Store::new(Arc::new(LatencyStore::new(
            inner,
            LatencyProfile {
                get: Duration::from_millis(50),
                put: Duration::ZERO,
                list: Duration::ZERO,
            },
        )));
        store.put("a", vec![7u8; 4096]).await.unwrap();
        let start = Instant::now();
        let parts = store
            .get_ranges("a", &[0..16, 100..200, 3000..3100], None)
            .await
            .unwrap();
        assert_eq!(parts.len(), 3);
        // Three ranges, one roundtrip: well under two delays.
        assert!(start.elapsed() < Duration::from_millis(95));
    }
}
