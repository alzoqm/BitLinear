#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, typename cast_t>
__device__ int getBitFromElement(const scalar_t element, const int bitPosition) {
    const cast_t* elementAsCast = reinterpret_cast<const cast_t*>(&element);
    int bit = (*elementAsCast >> bitPosition) & 1;
    return bit;
}

template <typename scalar_t>
__global__ void Matrix_Mul_a_a(const scalar_t *a, const scalar_t *b, scalar_t *c, uint32_t bs, uint32_t n, uint32_t m, uint32_t p, int max_a, int elementBitLength) { 
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.z * blockDim.z + threadIdx.z; 
    
    if(l < bs && i < n && j < p) {
        scalar_t sum = 0;
        for(int k = 0; k < m; k++) {
            int elementIdx = ((l % max_a) * n * m + i * m + k) / elementBitLength;
            int bitPosition = elementBitLength - (((l % max_a) * n * m + i * m + k) % elementBitLength) - 1;
            // Access b as usual, but access a in bit-wise manner
            
            sum += getBitFromElement<scalar_t, int>(a[elementIdx], bitPosition) * b[l * m * p + k * p + j];
        }
        c[l*n*p + i * p + j] = sum;
    }
}


template <typename scalar_t>
__global__ void Matrix_Mul_a_b(const scalar_t *a, const scalar_t *b, scalar_t *c, uint32_t bs, uint32_t n, uint32_t m, uint32_t p, int max_a, int elementBitLength) { 
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.z * blockDim.z + threadIdx.z; 
    
    if(l < bs && i < n && j < p) {
        scalar_t sum = 0;
        for(int k = 0; k < m; k++) {
            int elementIdx = (l * m * p + k * p + j) / elementBitLength;
            int bitPosition = elementBitLength - ((l * m * p + k * p + j) % elementBitLength) - 1;
            // Access a as usual, but access b in bit-wise manner
            
            sum += a[(l % max_a) * n * m + i * m + k] * getBitFromElement<scalar_t, int>(b[elementIdx], bitPosition);
        }
        c[l*n*p + i * p + j] = sum;
    }
}

template <typename scalar_t>
__global__ void Matrix_Mul_a_all(const scalar_t *a, const scalar_t *b, scalar_t *c, uint32_t bs, uint32_t n, uint32_t m, uint32_t p, int max_a, int elementBitLength) { 
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.z * blockDim.z + threadIdx.z; 
    
    if(l < bs && i < n && j < p) {
        scalar_t sum = 0;
        for(int k = 0; k < m; k++) {
            int elementIdx_a = ((l % max_a) * n * m + i * m + k) / elementBitLength;
            int bitPosition_a = elementBitLength - (((l % max_a) * n * m + i * m + k) % elementBitLength) - 1;
            int elementIdx_b = (l * m * p + k * p + j) / elementBitLength;
            int bitPosition_b = elementBitLength - ((l * m * p + k * p + j) % elementBitLength) - 1;
            
            sum += getBitFromElement<scalar_t, int>(a[elementIdx_a], bitPosition_a) * getBitFromElement<scalar_t, int>(b[elementIdx_b], bitPosition_b);
        }
        c[l*n*p + i * p + j] = sum;
    }
}

template <typename scalar_t>
__global__ void Matrix_Mul_b_a(const scalar_t *a, const scalar_t *b, scalar_t *c, uint32_t bs, uint32_t n, uint32_t m, uint32_t p, int max_b, int elementBitLength) { 
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.z * blockDim.z + threadIdx.z; 
    
    if(l < bs && i < n && j < p) {
        scalar_t sum = 0;
        for(int k = 0; k < m; k++) {
            int elementIdx = (l*n*m + i * m + k) / elementBitLength;
            int bitPosition = elementBitLength - ((l*n*m + i * m + k) % elementBitLength) - 1;
            // Access b as usual, but access a in bit-wise manner
            
            sum += getBitFromElement<scalar_t, int>(a[elementIdx], bitPosition) * b[(l%max_b)*m*p + k * p + j];
        }
        c[l*n*p + i * p + j] = sum;
    }
}

template <typename scalar_t>
__global__ void Matrix_Mul_b_b(const scalar_t *a, const scalar_t *b, scalar_t *c, uint32_t bs, uint32_t n, uint32_t m, uint32_t p, int max_b, int elementBitLength) { 
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.z * blockDim.z + threadIdx.z; 
    
    if(l < bs && i < n && j < p) {
        scalar_t sum = 0;
        for(int k = 0; k < m; k++) {
            int elementIdx = ((l%max_b)*m*p + k * p + j) / elementBitLength;
            int bitPosition = elementBitLength - (((l%max_b)*m*p + k * p + j) % elementBitLength) - 1;
            // Access a as usual, but access b in bit-wise manner
            
            sum += a[l*n*m + i * m + k] * getBitFromElement<scalar_t, int>(b[elementIdx], bitPosition);
        }
        c[l*n*p + i * p + j] = sum;
    }
}

template <typename scalar_t>
__global__ void Matrix_Mul_b_all(const scalar_t *a, const scalar_t *b, scalar_t *c, uint32_t bs, uint32_t n, uint32_t m, uint32_t p, int max_b, int elementBitLength) { 
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.z * blockDim.z + threadIdx.z; 
    
    if(l < bs && i < n && j < p) {
        scalar_t sum = 0;
        for(int k = 0; k < m; k++) {
            int elementIdx_a = (l*n*m + i * m + k) / elementBitLength;
            int bitPosition_a = elementBitLength - ((l*n*m + i * m + k) % elementBitLength) - 1;
            int elementIdx_b = ((l%max_b)*m*p + k * p + j) / elementBitLength;
            int bitPosition_b = elementBitLength - (((l%max_b)*m*p + k * p + j) % elementBitLength) - 1;
            
            sum += getBitFromElement<scalar_t, int>(a[elementIdx_a], bitPosition_a) * getBitFromElement<scalar_t, int>(b[elementIdx_b], bitPosition_b);
        }
        c[l*n*p + i * p + j] = sum;
    }
}

// Original function that takes the output tensor as an argument.
void bitMatMul(const torch::Tensor& input_tensor_a, const torch::Tensor& input_tensor_b, torch::Tensor& output_tensor, bool check_a_b_dim, int max_a, int max_b, int a_n, int a_m, int b_n, int b_m, bool a_is_bit, bool b_is_bit) {
    const auto elementBitLength = output_tensor.element_size() * 8;

    // const int threadsPerBlock = 256;
    // const int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    if(check_a_b_dim == true){
        dim3 block(4, 16, 16);
        dim3 grid((max_b + block.x - 1) / block.x, (a_n + block.y - 1) / block.y, (b_m + block.z - 1) / block.z);
        size_t sharedMemorySize = a_m * sizeof(output_tensor.element_size());

        if(a_is_bit==true && b_is_bit==true){
            if (elementBitLength <= 16) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_16", ([&] {
                    Matrix_Mul_a_all<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_b, a_n, a_m, b_m, max_a, elementBitLength);
                }));
            } else if (elementBitLength <= 32) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_32", ([&] {
                    Matrix_Mul_a_all<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_b, a_n, a_m, b_m, max_a, elementBitLength);
                }));
            } else if (elementBitLength <= 64) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_64", ([&] {
                    Matrix_Mul_a_all<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_b, a_n, a_m, b_m, max_a, elementBitLength);
                }));
            }
        }
        else if(a_is_bit==true){
            if (elementBitLength <= 16) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_16", ([&] {
                    Matrix_Mul_a_a<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_b, a_n, a_m, b_m, max_a, elementBitLength);
                }));
            } else if (elementBitLength <= 32) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_32", ([&] {
                    Matrix_Mul_a_a<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_b, a_n, a_m, b_m, max_a, elementBitLength);
                }));
            } else if (elementBitLength <= 64) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_64", ([&] {
                    Matrix_Mul_a_a<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_b, a_n, a_m, b_m, max_a, elementBitLength);
                }));
            }
        }
        else if(b_is_bit==true){

            if (elementBitLength <= 16) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_16", ([&] {
                    Matrix_Mul_a_b<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_b, a_n, a_m, b_m, max_a, elementBitLength);
                }));
            } else if (elementBitLength <= 32) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_32", ([&] {
                    Matrix_Mul_a_b<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_b, a_n, a_m, b_m, max_a, elementBitLength);
                }));
            } else if (elementBitLength <= 64) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_64", ([&] {
                    Matrix_Mul_a_b<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_b, a_n, a_m, b_m, max_a, elementBitLength);
                }));
            }
        }
    }
    else{
        dim3 block(4, 16, 16);
        dim3 grid((max_a + block.x - 1) / block.x, (a_n + block.y - 1) / block.y, (b_m + block.z - 1) / block.z);
        size_t sharedMemorySize = a_m * sizeof(output_tensor.element_size());

        if(a_is_bit==true && b_is_bit==true){
            if (elementBitLength <= 16) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_16", ([&] {
                    Matrix_Mul_b_all<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_a, a_n, a_m, b_m, max_b, elementBitLength);
                }));
            } else if (elementBitLength <= 32) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_32", ([&] {
                    Matrix_Mul_b_all<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_a, a_n, a_m, b_m, max_b, elementBitLength);
                }));
            } else if (elementBitLength <= 64) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_64", ([&] {
                    Matrix_Mul_b_all<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_a, a_n, a_m, b_m, max_b, elementBitLength);
                }));
            }
        }
        else if(a_is_bit==true){
            if (elementBitLength <= 16) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_16", ([&] {
                    Matrix_Mul_b_a<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_a, a_n, a_m, b_m, max_b, elementBitLength);
                }));
            } else if (elementBitLength <= 32) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_32", ([&] {
                    Matrix_Mul_b_a<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_a, a_n, a_m, b_m, max_b, elementBitLength);
                }));
            } else if (elementBitLength <= 64) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_64", ([&] {
                    Matrix_Mul_b_a<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_a, a_n, a_m, b_m, max_b, elementBitLength);
                }));
            }
        }
        else if(b_is_bit==true){
            if (elementBitLength <= 16) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_16", ([&] {
                    Matrix_Mul_b_b<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_a, a_n, a_m, b_m, max_b, elementBitLength);
                }));
            } else if (elementBitLength <= 32) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_32", ([&] {
                    Matrix_Mul_b_b<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_a, a_n, a_m, b_m, max_b, elementBitLength);
                }));
            } else if (elementBitLength <= 64) {
                AT_DISPATCH_ALL_TYPES(output_tensor.scalar_type(), "bitMatMul_64", ([&] {
                    Matrix_Mul_b_b<scalar_t><<<grid, block, sharedMemorySize>>>(input_tensor_a.data_ptr<scalar_t>(), input_tensor_b.data_ptr<scalar_t>(), output_tensor.data_ptr<scalar_t>(), max_a, a_n, a_m, b_m, max_b, elementBitLength);
                }));
            }
        }
    }
    // cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bitMatMul", (void (*)(const torch::Tensor&, const torch::Tensor&, torch::Tensor&, bool, int, int, int, int, int, int, bool, bool)) &bitMatMul, "Dispatch function for bitMatMul Kernel taking output tensor as argument.");
}