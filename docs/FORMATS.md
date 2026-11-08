# Quantization formats

## Common layout

Each artifact directory contains:
- `packed.npy`: packed unsigned bytes in little-endian bit order.
- `scales.npy`: float32 scale for each contiguous flattened group.
- `manifest.json`: version, bit width, original shape/count, group size, scheme and storage accounting.

This is a research container, not a standardized interchange format. The manifest is required to decode the bytes.

## INT4 and INT2

Signed two's-complement values are packed least-significant bits first. The quantizer uses symmetric ranges {-7..7} for INT4 and {-1..1} for INT2, with one positive scale per group. The unused most-negative code is intentionally not emitted.

## INT1

INT1 stores binary sign values {-1,+1}; zero is not representable within a non-zero group. A completely zero group receives a zero scale and reconstructs as zero; isolated zero elements in a non-zero group map to the positive sign and reconstruct near the group's scale. This is highly lossy and must be treated as research-only unless model-level quality proves otherwise.

## Accounting

The effective bits/value calculation includes packed payload and float32 scales. It excludes filesystem/container overhead, JSON metadata and any runtime staging buffers; report those separately in production measurements.

## Endianness and versioning

Bit order is little-endian within each byte. Manifest version 1 is required. Future incompatible changes must increment the version. Do not infer a format from a filename or extension.
