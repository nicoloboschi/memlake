//! S3 cost model — the pitch is "lowball Postgres", so cost is a first-class metric.
//!
//! Prices are AWS S3 Standard, us-east-1 (edit `Pricing` for another provider/region).
//! Costs are computed from *counted* store operations (`StoreSnapshot`) and the stored byte
//! footprint, so an indexer change that doubles PUTs shows up directly as a pricing change.

use mlake_store::StoreSnapshot;

/// S3 pricing, per AWS S3 Standard us-east-1.
#[derive(Clone, Copy)]
pub struct Pricing {
    /// $ per 1,000 PUT/COPY/POST/LIST requests.
    pub put_per_1k: f64,
    /// $ per 1,000 GET/SELECT requests.
    pub get_per_1k: f64,
    /// $ per GB-month of Standard storage.
    pub storage_gb_month: f64,
}

impl Default for Pricing {
    fn default() -> Self {
        Self {
            put_per_1k: 0.005,
            get_per_1k: 0.0004,
            storage_gb_month: 0.023,
        }
    }
}

impl Pricing {
    /// Request cost of a measured phase (writes count PUTs+LISTs; reads count GETs).
    pub fn request_cost(&self, s: &StoreSnapshot) -> f64 {
        let put_like = (s.puts + s.lists + s.deletes) as f64;
        let get_like = s.gets as f64;
        put_like / 1000.0 * self.put_per_1k + get_like / 1000.0 * self.get_per_1k
    }

    /// Monthly storage cost for `bytes` of Standard storage.
    pub fn storage_cost_month(&self, bytes: u64) -> f64 {
        bytes as f64 / 1e9 * self.storage_gb_month
    }
}

const GB: f64 = 1e9;

/// Cost summary for one write phase.
pub struct WriteCost {
    pub ingest_requests_usd: f64,
    pub stored_bytes: u64,
    pub storage_usd_month: f64,
    /// $ to ingest 1M memories at this rate (request cost only).
    pub usd_per_million_ingested: f64,
}

pub fn write_cost(pricing: &Pricing, phase: &StoreSnapshot, stored_bytes: u64, memories: usize) -> WriteCost {
    let req = pricing.request_cost(phase);
    WriteCost {
        ingest_requests_usd: req,
        stored_bytes,
        storage_usd_month: pricing.storage_cost_month(stored_bytes),
        usd_per_million_ingested: if memories > 0 {
            req / memories as f64 * 1_000_000.0
        } else {
            0.0
        },
    }
}

/// Cost summary for a read workload.
pub struct ReadCost {
    /// $ per 1,000 queries at the measured GET rate.
    pub usd_per_1k_queries: f64,
    pub gets_per_query: f64,
    pub bytes_per_query: f64,
}

pub fn read_cost(pricing: &Pricing, phase: &StoreSnapshot, queries: usize) -> ReadCost {
    if queries == 0 {
        return ReadCost { usd_per_1k_queries: 0.0, gets_per_query: 0.0, bytes_per_query: 0.0 };
    }
    let per_query_req = pricing.request_cost(phase) / queries as f64;
    ReadCost {
        usd_per_1k_queries: per_query_req * 1000.0,
        gets_per_query: phase.gets as f64 / queries as f64,
        bytes_per_query: phase.get_bytes as f64 / queries as f64,
    }
}

/// Human-readable GB.
pub fn gb(bytes: u64) -> f64 {
    bytes as f64 / GB
}
