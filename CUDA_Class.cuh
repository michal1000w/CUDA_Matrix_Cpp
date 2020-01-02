#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cmath>

using std::cout;
using std::endl;

////////////////////////////////KERNELS////////////////////////////////
template <typename Y>
__global__
void addKernel(Y* a, const Y* b) {
    int i = threadIdx.x;
    a[i] += b[i];
}

template <typename Y>
__global__
void addNKernel(Y* a, const Y* b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        a[i] += b[i];
}

template <typename Y>
__global__
void subtractNKernel(Y* a, const Y* b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        a[i] -= b[i];
}

template <typename Y>
__global__
void multiplyNKernel(Y* a, const Y* b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        a[i] *= b[i];
}

template <typename Y>
__global__
void cmultiplyNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        a[i] *= b;
}

template <typename Y>
__global__
void cdivideNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        a[i] /= b;
}




////////////////////////////////CLASS///////////////////////////////////////

template <typename Y>
class CUDA_Class {
public:
    CUDA_Class<Y>() {}

    cudaError_t add(Y* a, Y* b, unsigned int size);
    cudaError_t subtract(Y* a, Y* b, unsigned int size);
    cudaError_t multiply(Y* a, Y* b, unsigned int size);
    cudaError_t cmultiply(Y* a, Y b, unsigned int size);
    cudaError_t cdivide(Y* a, Y b, unsigned int size);

private:
    unsigned int THREADS_PER_BLOCK = 512;
};

//////////////////////////LAUNCH FUNCTIONS//////////////////

template <typename Y>
cudaError_t CUDA_Class<Y>::add(Y* a, Y* b, unsigned int size) {
    //variables
    cudaError_t cudaStatus;
    unsigned int total_size = size * sizeof(Y);

    Y* dev_a = 0;
    Y* dev_b = 0;

    //set the device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) cout << "Setting device failed" << endl;

    //allocate memory
    cudaStatus = cudaMalloc((void**)&dev_a, total_size);
    if (cudaStatus != cudaSuccess) cout << "Memory alloc failed" << endl;
    cudaStatus = cudaMalloc((void**)&dev_b, total_size);
    if (cudaStatus != cudaSuccess) cout << "Memory alloc failed" << endl;

    //copy vectors to GPU
    cudaStatus = cudaMemcpy(dev_a, a, total_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) cout << "Copying to device failed" << endl;
    cudaStatus = cudaMemcpy(dev_b, b, total_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) cout << "Copying to device failed" << endl;

    //launch kernel
    //addKernel << <1, size >> > (dev_a, dev_b);
    addNKernel << <(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (dev_a, dev_b,size);

    //check if errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) cout << "Launching kernel failed" << endl;


    //synchronize devices
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) cout << "Synchronizing failed" << endl;

    //copy output to host
    cudaStatus = cudaMemcpy(a, dev_a, total_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) cout << "Copying to host failed" << endl;

    //free memory
    cudaFree(dev_a);
    cudaFree(dev_b);

    //reset the device
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) cout << "Resetting device failed" << endl;

    return cudaStatus;
}

template <typename Y>
cudaError_t CUDA_Class<Y>::subtract(Y* a, Y* b, unsigned int size) {
    //variables
    cudaError_t cudaStatus;
    unsigned int total_size = size * sizeof(Y);

    Y* dev_a = 0;
    Y* dev_b = 0;

    //set the device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) cout << "Setting device failed" << endl;

    //allocate memory
    cudaStatus = cudaMalloc((void**)&dev_a, total_size);
    if (cudaStatus != cudaSuccess) cout << "Memory alloc failed" << endl;
    cudaStatus = cudaMalloc((void**)&dev_b, total_size);
    if (cudaStatus != cudaSuccess) cout << "Memory alloc failed" << endl;

    //copy vectors to GPU
    cudaStatus = cudaMemcpy(dev_a, a, total_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) cout << "Copying to device failed" << endl;
    cudaStatus = cudaMemcpy(dev_b, b, total_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) cout << "Copying to device failed" << endl;

    //launch kernel
    //addKernel << <1, size >> > (dev_a, dev_b);
    subtractNKernel << <(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (dev_a, dev_b, size);

    //check if errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) cout << "Launching kernel failed" << endl;


    //synchronize devices
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) cout << "Synchronizing failed" << endl;

    //copy output to host
    cudaStatus = cudaMemcpy(a, dev_a, total_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) cout << "Copying to host failed" << endl;

    //free memory
    cudaFree(dev_a);
    cudaFree(dev_b);

    //reset the device
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) cout << "Resetting device failed" << endl;

    return cudaStatus;
}

template <typename Y>
cudaError_t CUDA_Class<Y>::multiply(Y* a, Y* b, unsigned int size) {
    //variables
    cudaError_t cudaStatus;
    unsigned int total_size = size * sizeof(Y);

    Y* dev_a = 0;
    Y* dev_b = 0;

    //set the device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) cout << "Setting device failed" << endl;

    //allocate memory
    cudaStatus = cudaMalloc((void**)&dev_a, total_size);
    if (cudaStatus != cudaSuccess) cout << "Memory alloc failed" << endl;
    cudaStatus = cudaMalloc((void**)&dev_b, total_size);
    if (cudaStatus != cudaSuccess) cout << "Memory alloc failed" << endl;

    //copy vectors to GPU
    cudaStatus = cudaMemcpy(dev_a, a, total_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) cout << "Copying to device failed" << endl;
    cudaStatus = cudaMemcpy(dev_b, b, total_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) cout << "Copying to device failed" << endl;

    //launch kernel
    //addKernel << <1, size >> > (dev_a, dev_b);
    multiplyNKernel << <(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (dev_a, dev_b, size);

    //check if errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) cout << "Launching kernel failed" << endl;


    //synchronize devices
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) cout << "Synchronizing failed" << endl;

    //copy output to host
    cudaStatus = cudaMemcpy(a, dev_a, total_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) cout << "Copying to host failed" << endl;

    //free memory
    cudaFree(dev_a);
    cudaFree(dev_b);

    //reset the device
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) cout << "Resetting device failed" << endl;

    return cudaStatus;
}

template <typename Y>
cudaError_t CUDA_Class<Y>::cmultiply(Y* a, Y b, unsigned int size) {
    //variables
    cudaError_t cudaStatus;
    unsigned int total_size = size * sizeof(Y);

    Y* dev_a = 0;

    //set the device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) cout << "Setting device failed" << endl;

    //allocate memory
    cudaStatus = cudaMalloc((void**)&dev_a, total_size);
    if (cudaStatus != cudaSuccess) cout << "Memory alloc failed" << endl;

    //copy vectors to GPU
    cudaStatus = cudaMemcpy(dev_a, a, total_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) cout << "Copying to device failed" << endl;

    //launch kernel
    //addKernel << <1, size >> > (dev_a, dev_b);
    cmultiplyNKernel << <(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (dev_a, b, size);

    //check if errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) cout << "Launching kernel failed" << endl;


    //synchronize devices
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) cout << "Synchronizing failed" << endl;

    //copy output to host
    cudaStatus = cudaMemcpy(a, dev_a, total_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) cout << "Copying to host failed" << endl;

    //free memory
    cudaFree(dev_a);

    //reset the device
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) cout << "Resetting device failed" << endl;

    return cudaStatus;
}

template <typename Y>
cudaError_t CUDA_Class<Y>::cdivide(Y* a, Y b, unsigned int size) {
    //variables
    cudaError_t cudaStatus;
    unsigned int total_size = size * sizeof(Y);

    Y* dev_a = 0;

    //set the device
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) cout << "Setting device failed" << endl;

    //allocate memory
    cudaStatus = cudaMalloc((void**)&dev_a, total_size);
    if (cudaStatus != cudaSuccess) cout << "Memory alloc failed" << endl;

    //copy vectors to GPU
    cudaStatus = cudaMemcpy(dev_a, a, total_size, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) cout << "Copying to device failed" << endl;

    //launch kernel
    //addKernel << <1, size >> > (dev_a, dev_b);
    cdivideNKernel << <(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (dev_a, b, size);

    //check if errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) cout << "Launching kernel failed" << endl;


    //synchronize devices
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) cout << "Synchronizing failed" << endl;

    //copy output to host
    cudaStatus = cudaMemcpy(a, dev_a, total_size, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) cout << "Copying to host failed" << endl;

    //free memory
    cudaFree(dev_a);

    //reset the device
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) cout << "Resetting device failed" << endl;

    return cudaStatus;
}