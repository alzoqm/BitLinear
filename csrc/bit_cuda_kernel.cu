#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void checkBits(const float *input_tensor, int *output_bits, int totalBits) 
{ 
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < totalBits) 
    { 
        int elementIdx = idx / 32; // 원소 인덱스 
        int bitIdx = idx % 32; // 비트 인덱스 
        float val = input_tensor[elementIdx]; 
        unsigned int intPtr = (unsigned int)&val; // float을 unsigned int 포인터로 캐스팅하여 비트 접근 
        int bit = (*intPtr >> (31 - bitIdx)) & 1; // MSB부터 LSB로 비트 추출 
        output_bits[idx] = bit; 
    } 
}