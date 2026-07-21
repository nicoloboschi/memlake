//! Memory identity.
//!
//! Ids are UUIDs on the wire and in the API, but stored as a fixed 16-byte array so
//! archived records stay zero-copy readable (`uuid::Uuid` is not an rkyv type).

use std::fmt;

use rkyv::{Archive, Deserialize, Serialize};
use uuid::Uuid;

#[derive(
    Archive, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
)]
#[archive(check_bytes)]
#[archive_attr(derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug))]
pub struct MemoryId(pub [u8; 16]);

impl MemoryId {
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub fn new_v4() -> Self {
        Self(*Uuid::new_v4().as_bytes())
    }

    /// Deterministic id from an external string key. Used by the benchmark harness so a
    /// BEIR document id maps to a stable MemoryId across runs.
    pub fn from_key(key: &str) -> Self {
        Self(*Uuid::new_v5(&Uuid::NAMESPACE_OID, key.as_bytes()).as_bytes())
    }

    pub fn as_uuid(&self) -> Uuid {
        Uuid::from_bytes(self.0)
    }
}

impl From<Uuid> for MemoryId {
    fn from(u: Uuid) -> Self {
        Self(*u.as_bytes())
    }
}

impl From<MemoryId> for Uuid {
    fn from(id: MemoryId) -> Self {
        Uuid::from_bytes(id.0)
    }
}

impl fmt::Debug for MemoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_uuid())
    }
}

impl fmt::Display for MemoryId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_uuid())
    }
}

// Written by hand rather than derived: `MemoryId` already derives rkyv's `Serialize` and
// `Deserialize`, and deriving serde's same-named traits alongside them makes every
// unqualified call ambiguous.
impl serde::Serialize for MemoryId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&self.as_uuid(), s)
    }
}

impl<'de> serde::Deserialize<'de> for MemoryId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        <Uuid as serde::Deserialize>::deserialize(d).map(Self::from)
    }
}

/// An entity identity. Like [`MemoryId`], a full 16-byte value (Hindsight's entity ids are
/// UUIDs) stored as a fixed array so archived records stay zero-copy readable. Kept a
/// distinct type from `MemoryId` so the two can never be confused in the graph arm.
#[derive(
    Archive, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
)]
#[archive(check_bytes)]
#[archive_attr(derive(PartialEq, Eq, PartialOrd, Ord, Hash, Debug))]
pub struct EntityId(pub [u8; 16]);

impl EntityId {
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    /// Deterministic id from an external string key (used by the benchmark harness).
    pub fn from_key(key: &str) -> Self {
        Self(*Uuid::new_v5(&Uuid::NAMESPACE_OID, key.as_bytes()).as_bytes())
    }

    pub fn as_uuid(&self) -> Uuid {
        Uuid::from_bytes(self.0)
    }
}

impl From<Uuid> for EntityId {
    fn from(u: Uuid) -> Self {
        Self(*u.as_bytes())
    }
}

impl fmt::Debug for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_uuid())
    }
}

impl serde::Serialize for EntityId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&self.as_uuid(), s)
    }
}

impl<'de> serde::Deserialize<'de> for EntityId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        <Uuid as serde::Deserialize>::deserialize(d).map(Self::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_key_is_deterministic() {
        assert_eq!(MemoryId::from_key("doc-1"), MemoryId::from_key("doc-1"));
        assert_ne!(MemoryId::from_key("doc-1"), MemoryId::from_key("doc-2"));
    }

    #[test]
    fn uuid_roundtrip() {
        let id = MemoryId::new_v4();
        assert_eq!(MemoryId::from(id.as_uuid()), id);
    }

    #[test]
    fn ordering_matches_byte_order() {
        // pk.idx relies on MemoryId sorting by raw bytes so it can be binary-searched.
        let a = MemoryId::from_bytes([0u8; 16]);
        let mut b_bytes = [0u8; 16];
        b_bytes[15] = 1;
        let b = MemoryId::from_bytes(b_bytes);
        assert!(a < b);
    }
}
