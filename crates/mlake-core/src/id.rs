//! Item identity.
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
pub struct ItemId(pub [u8; 16]);

impl ItemId {
    pub const fn from_bytes(bytes: [u8; 16]) -> Self {
        Self(bytes)
    }

    pub fn new_v4() -> Self {
        Self(*Uuid::new_v4().as_bytes())
    }

    /// Deterministic id from an external string key. Used by the benchmark harness so a
    /// BEIR document id maps to a stable ItemId across runs.
    pub fn from_key(key: &str) -> Self {
        Self(*Uuid::new_v5(&Uuid::NAMESPACE_OID, key.as_bytes()).as_bytes())
    }

    pub fn as_uuid(&self) -> Uuid {
        Uuid::from_bytes(self.0)
    }
}

impl From<Uuid> for ItemId {
    fn from(u: Uuid) -> Self {
        Self(*u.as_bytes())
    }
}

impl From<ItemId> for Uuid {
    fn from(id: ItemId) -> Self {
        Uuid::from_bytes(id.0)
    }
}

impl fmt::Debug for ItemId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_uuid())
    }
}

impl fmt::Display for ItemId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_uuid())
    }
}

// Written by hand rather than derived: `ItemId` already derives rkyv's `Serialize` and
// `Deserialize`, and deriving serde's same-named traits alongside them makes every
// unqualified call ambiguous.
impl serde::Serialize for ItemId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&self.as_uuid(), s)
    }
}

impl<'de> serde::Deserialize<'de> for ItemId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        <Uuid as serde::Deserialize>::deserialize(d).map(Self::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_key_is_deterministic() {
        assert_eq!(ItemId::from_key("doc-1"), ItemId::from_key("doc-1"));
        assert_ne!(ItemId::from_key("doc-1"), ItemId::from_key("doc-2"));
    }

    #[test]
    fn uuid_roundtrip() {
        let id = ItemId::new_v4();
        assert_eq!(ItemId::from(id.as_uuid()), id);
    }

    #[test]
    fn ordering_matches_byte_order() {
        // pk.idx relies on ItemId sorting by raw bytes so it can be binary-searched.
        let a = ItemId::from_bytes([0u8; 16]);
        let mut b_bytes = [0u8; 16];
        b_bytes[15] = 1;
        let b = ItemId::from_bytes(b_bytes);
        assert!(a < b);
    }
}
