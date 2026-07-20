//! Minimal reader for NumPy `.npy` arrays.
//!
//! The benchmark's whole premise is that memlake and Qdrant score *identical* vectors, so
//! this reads the exact float32 arrays the Python harness cached rather than re-embedding.
//! Only the one case the cache produces is supported — 2-D, little-endian `<f4`, C-order —
//! and anything else is a hard error rather than a silent misread.

use std::io::Read;
use std::path::Path;

use anyhow::{bail, Context, Result};

/// A dense 2-D f32 matrix read from a `.npy` file.
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    /// Borrow row `i` as a slice.
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

/// Read a 2-D little-endian float32 C-order array.
pub fn read_f32_matrix(path: &Path) -> Result<Matrix> {
    let mut file = std::fs::File::open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic)?;
    if &magic[0..6] != b"\x93NUMPY" {
        bail!("{} is not a .npy file", path.display());
    }
    let major = magic[6];

    // Header length is u16 (v1) or u32 (v2+).
    let header_len = if major >= 2 {
        let mut b = [0u8; 4];
        file.read_exact(&mut b)?;
        u32::from_le_bytes(b) as usize
    } else {
        let mut b = [0u8; 2];
        file.read_exact(&mut b)?;
        u16::from_le_bytes(b) as usize
    };

    let mut header = vec![0u8; header_len];
    file.read_exact(&mut header)?;
    let header = String::from_utf8(header).context("npy header is not utf8")?;

    if header.contains("'fortran_order': True") {
        bail!("{}: fortran-order arrays are not supported", path.display());
    }
    if !(header.contains("'<f4'") || header.contains("\"<f4\"")) {
        bail!("{}: expected little-endian float32 (<f4)", path.display());
    }

    let (rows, cols) = parse_shape(&header)
        .with_context(|| format!("parsing shape from npy header: {header}"))?;

    let mut raw = Vec::new();
    file.read_to_end(&mut raw)?;
    let expected = rows * cols * 4;
    if raw.len() < expected {
        bail!(
            "{}: truncated data ({} bytes, expected {expected})",
            path.display(),
            raw.len()
        );
    }

    let mut data = Vec::with_capacity(rows * cols);
    for chunk in raw[..expected].chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(Matrix { rows, cols, data })
}

/// Parse `'shape': (R, C)` out of the header dict.
fn parse_shape(header: &str) -> Option<(usize, usize)> {
    let start = header.find("'shape':")? + "'shape':".len();
    let open = header[start..].find('(')? + start + 1;
    let close = header[open..].find(')')? + open;
    let inside = &header[open..close];
    let mut nums = inside
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>());
    let rows = nums.next()?.ok()?;
    let cols = nums.next()?.ok()?;
    Some((rows, cols))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_two_dim_shape() {
        let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (5183, 384), }";
        assert_eq!(parse_shape(header), Some((5183, 384)));
    }

    #[test]
    fn parses_shape_with_odd_spacing() {
        let header = "{'shape':(10,20),}";
        assert_eq!(parse_shape(header), Some((10, 20)));
    }
}
