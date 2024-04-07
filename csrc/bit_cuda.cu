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

template <typename scalar_t, typename cast_t>
__global__ void checkBitsKernel(const scalar_t* input, int* output, int numElements, int elementBitLength) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < numElements) {
        int elementIdx = idx / elementBitLength;
        int bitPosition = elementBitLength - (idx % elementBitLength) - 1;
        scalar_t element = input[elementIdx];
        int bit = getBitFromElement<scalar_t, cast_t>(element, bitPosition);
        output[idx] = bit;
    }
}

// Original function that takes the output tensor as an argument.
void checkBitsTorch(const torch::Tensor& input_tensor, torch::Tensor& output_tensor) {

    if (input_tensor.device() != output_tensor.device()) {
        throw std::runtime_error("Input and output tensors must be same device");
    }
    const auto elementBitLength = input_tensor.element_size() * 8;
    const auto numElements = input_tensor.numel() * elementBitLength;

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    if (elementBitLength <= 16) {
        AT_DISPATCH_ALL_TYPES(input_tensor.scalar_type(), "checkBitsTorch_16", ([&] {
            checkBitsKernel<scalar_t, uint16_t><<<blocksPerGrid, threadsPerBlock>>>(input_tensor.data_ptr<scalar_t>(), output_tensor.data_ptr<int>(), numElements, elementBitLength);
        }));
    } else if (elementBitLength <= 32) {
        AT_DISPATCH_ALL_TYPES(input_tensor.scalar_type(), "checkBitsTorch_32", ([&] {
            checkBitsKernel<scalar_t, uint32_t><<<blocksPerGrid, threadsPerBlock>>>(input_tensor.data_ptr<scalar_t>(), output_tensor.data_ptr<int>(), numElements, elementBitLength);
        }));
    } else if (elementBitLength <= 64) {
        AT_DISPATCH_ALL_TYPES(input_tensor.scalar_type(), "checkBitsTorch_64", ([&] {
            checkBitsKernel<scalar_t, uint64_t><<<blocksPerGrid, threadsPerBlock>>>(input_tensor.data_ptr<scalar_t>(), output_tensor.data_ptr<int>(), numElements, elementBitLength);
        }));
    }
}

torch::Tensor checkBitsTorch(const torch::Tensor& input_tensor) {
    const auto elementBitLength = input_tensor.element_size() * 8;
    const auto numElements = input_tensor.numel() * elementBitLength;
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(input_tensor.device());
    auto output_tensor = torch::empty({numElements}, options);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    if (elementBitLength <= 16) {
        AT_DISPATCH_ALL_TYPES(input_tensor.scalar_type(), "checkBitsTorch_16", ([&] {
            checkBitsKernel<scalar_t, uint16_t><<<blocksPerGrid, threadsPerBlock>>>(input_tensor.data_ptr<scalar_t>(), output_tensor.data_ptr<int>(), numElements, elementBitLength);
        }));
    } else if (elementBitLength <= 32) {
        AT_DISPATCH_ALL_TYPES(input_tensor.scalar_type(), "checkBitsTorch_32", ([&] {
            checkBitsKernel<scalar_t, uint32_t><<<blocksPerGrid, threadsPerBlock>>>(input_tensor.data_ptr<scalar_t>(), output_tensor.data_ptr<int>(), numElements, elementBitLength);
        }));
    } else if (elementBitLength <= 64) {
        AT_DISPATCH_ALL_TYPES(input_tensor.scalar_type(), "checkBitsTorch_64", ([&] {
            checkBitsKernel<scalar_t, uint64_t><<<blocksPerGrid, threadsPerBlock>>>(input_tensor.data_ptr<scalar_t>(), output_tensor.data_ptr<int>(), numElements, elementBitLength);
        }));
    }

    return output_tensor;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("checkBits", (void (*)(const torch::Tensor&, torch::Tensor&)) &checkBitsTorch, "Dispatch function for Check Bits Kernel taking output tensor as argument.");
    m.def("checkBits", (torch::Tensor (*)(const torch::Tensor&)) &checkBitsTorch, "Dispatch function for Check Bits Kernel returning output tensor.");
}