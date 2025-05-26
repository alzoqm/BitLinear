# Overview

`bitLinear` is a PyTorch CUDA extension providing bit-level matrix multiplication and bit extraction kernels. It comprises two modules:

* **bitMatMul\_extension**: Implements bit-wise matrix multiplication between two tensors, supporting inputs stored as packed bits or standard numeric types.
* **bitCuda\_extension**: Provides bit-unpacking (checkBits) utilities to convert tensor elements into raw bit representations.

This extension allows efficient GPU execution of matrix operations at the bit granularity, useful for quantized neural networks, binary neural network research, or custom low-precision computations.

---

# Repository Structure

```text
.
├── setup.py                   # Build script for both extensions
├── bitMatMul.cu               # CUDA kernels and binding for bit-level matrix multiplication
├── bit_cuda.cu                # CUDA kernels and binding for bit-unpacking utilities (checkBits)
└── bit_cuda_kernel.cu         # Low-level CUDA kernel for raw bit extraction
```

---

# Requirements

* CUDA 12.0 or higher
* PyTorch (with matching CUDA toolkit) v1.7 or higher
* C++ compiler supporting C++14
* Python 3.8+

---

# Installation

From the root of this repository, install in editable or standard mode:

```bash
# Editable install (development)
pip install -e .

# Or standard build and install
python setup.py install
```

This will compile and install two CUDAExtension modules:

* `bitMatMul_extension`
* `bitCuda_extension`

---

# Usage

```python
import torch
# Import compiled extensions
import bitMatMul_extension
import bitCuda_extension

# Example: bit-unpacking
# Convert a float tensor to a CPU tensor of 32-bit flags (0/1) per bit
x = torch.tensor([1.5, -2.75, 0.0], device='cuda', dtype=torch.float32)
bits = bitCuda_extension.checkBits(x)  # returns shape [num_elements * 32]

# Example: bit-matrix multiplication
# Prepare two bit-packed or standard tensors
A = torch.randint(0, 2, (128, 256), dtype=torch.int8, device='cuda')  # packed as bits
B = torch.randint(0, 2, (256, 64), dtype=torch.int8, device='cuda')   # packed as bits
# Pre-allocate output tensor
C = torch.zeros(128, 64, dtype=torch.int32, device='cuda')

# Perform bit-matrix multiplication
# Arguments: (A, B, C, a_rows, a_cols, b_rows, b_cols, A_is_bit, B_is_bit)
bitMatMul_extension.bitMatMul(A, B, C, 128, 256, 256, 64, True, True)

# After call, `C[i,j]` holds the dot-product of A's i-th row and B's j-th column at bit granularity
```

---

# API Reference

## `bitCuda_extension.checkBits(input_tensor) -> Tensor`

Extract raw bits from `input_tensor`. Returns an `int32` tensor of length `num_elements * bit_width`, where `bit_width` matches the element size (e.g., 32 for `float32`).

## `bitMatMul_extension.bitMatMul(A, B, C, a_n, a_m, b_n, b_m, a_is_bit, b_is_bit)`

Performs bit-wise matrix multiplication:

* `A`, `B`: input tensors (packed bits or standard types)
* `C`: pre-allocated output tensor
* `a_n`, `a_m`: dimensions of `A` (rows, columns)
* `b_n`, `b_m`: dimensions of `B` (rows, columns)
* `a_is_bit`, `b_is_bit`: flags indicating if inputs are stored as bit-packed tensors

---

# Development and Testing

* To rebuild after changes: `python setup.py build_ext --inplace`
* Add tests under a `tests/` directory and run via `pytest`.

---

# License

MIT License
