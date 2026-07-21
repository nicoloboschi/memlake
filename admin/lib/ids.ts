/**
 * memlake ids (MemoryId, EntityId) are 16 raw bytes on the wire. Everywhere a
 * human sees one we render it as a canonical 8-4-4-4-12 UUID, and everywhere a
 * human types one we accept that same form (with or without dashes, any case).
 *
 * Pure Uint8Array — no Buffer — so this module is safe to import from a client
 * component as well as from a route handler.
 */

const HEX = "0123456789abcdef";

export class IdFormatError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "IdFormatError";
  }
}

/** 16 raw bytes -> "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx". */
export function bytesToUuid(bytes: Uint8Array | null | undefined): string {
  if (!bytes || bytes.length === 0) return "";
  if (bytes.length !== 16) {
    // Never throw on a display path: an unexpected width is worth showing, not
    // worth blanking the page for.
    return `0x${toHex(bytes)}`;
  }
  const h = toHex(bytes);
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(
    16,
    20,
  )}-${h.slice(20, 32)}`;
}

/** "xxxxxxxx-xxxx-..." (dashes optional) -> 16 raw bytes. Throws on bad input. */
export function uuidToBytes(uuid: string): Uint8Array {
  const hex = uuid.trim().replace(/-/g, "").toLowerCase();
  if (hex.length !== 32 || !/^[0-9a-f]{32}$/.test(hex)) {
    throw new IdFormatError(
      `not a 16-byte id: ${JSON.stringify(uuid)} (want 32 hex digits, optionally dashed)`,
    );
  }
  const out = new Uint8Array(16);
  for (let i = 0; i < 16; i++) {
    out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

/** True when `s` parses as a 16-byte id. Cheap enough for input validation. */
export function isUuid(s: string): boolean {
  return /^[0-9a-f]{32}$/.test(s.trim().replace(/-/g, "").toLowerCase());
}

/** Short form for dense tables: first and last 4 hex digits. */
export function shortId(uuid: string): string {
  if (uuid.length < 12) return uuid;
  return `${uuid.slice(0, 8)}…${uuid.slice(-4)}`;
}

function toHex(bytes: Uint8Array): string {
  let s = "";
  for (let i = 0; i < bytes.length; i++) {
    const b = bytes[i];
    s += HEX[(b >> 4) & 0xf] + HEX[b & 0xf];
  }
  return s;
}
