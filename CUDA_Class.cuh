#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cmath>

using std::cout;
using std::endl;

////////////////////////////////KERNELS////////////////////////////////
template <typename Y>
__global__
void setNKernel(Y* a, const Y* b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        a[i] = b[i];
}

template <typename Y>
__global__
void csetNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        a[i] = b;
}

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
void caddNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        a[i] += b;
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
void csubtractNKernel(Y* a, const Y b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        a[i] -= b;
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

///////////////
template <typename Y>
__global__ 
void sigmoidKernel(Y* a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        a[i] = 1.0 / (1.0 + std::exp(-1 * a[i]));
}

template <typename Y>
__global__
void sigmoid_derivativeKernel(Y* a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        a[i] = a[i] * (1 - a[i]);
}

template <typename Y>
__global__
void squareKernel(Y* a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
        a[i] *= a[i];
}




////////////////////////////////CLASS///////////////////////////////////////

template <typename Y>
class CUDA_Class {
public:
    CUDA_Class<Y>() {}

    cudaError_t set(Y* a, Y* b, unsigned int size);
    cudaError_t cset(Y* a, Y b, unsigned int size);
    cudaError_t add(Y* a, Y* b, unsigned int size);
    cudaError_t cadd(Y* a, Y b, unsigned int size);
    cudaError_t subtract(Y* a, Y* b, unsigned int size);
    cudaError_t csubtract(Y* a, Y b, unsigned int size);
    cudaError_t multiply(Y* a, Y* b, unsigned int size);
    cudaError_t cmultiply(Y* a, Y b, unsigned int size);
    cudaError_t cdivide(Y* a, Y b, unsigned int size);

    //matematyczne
    cudaError_t sigmoid(Y* a, unsigned int size);
    cudaError_t sigmoid_derivative(Y* a, unsigned int size);
    cudaError_t square(Y* a, unsigned int size);

    //setters
    void set_threads_per_block(unsigned int threads = 512) { this->THREADS_PER_BLOCK = threads; }

private:
    unsigned int THREADS_PER_BLOCK = 512;
};

//////////////////////////LAUNCH FUNCTIONS//////////////////
template <typename Y>
cudaError_t CUDA_Class<Y>::set(Y* a, Y* b, unsigned int size) {
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
    setNKernel << <(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (dev_a, dev_b, size);

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
cudaError_t CUDA_Class<Y>::cset(Y* a, Y b, unsigned int size) {
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
    csetNKernel << <(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (dev_a, b, size);

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
cudaError_t CUDA_Class<Y>::cadd(Y* a, Y b, unsigned int size) {
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
    caddNKernel << <(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (dev_a, b, size);

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
cudaError_t CUDA_Class<Y>::csubtract(Y* a, Y b, unsigned int size) {
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
    csubtractNKernel << <(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (dev_a, b, size);

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

template <typename Y>
cudaError_t CUDA_Class<Y>::sigmoid(Y* a, unsigned int size) {
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
    sigmoidKernel << <(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (dev_a, size);

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
cudaError_t CUDA_Class<Y>::sigmoid_derivative(Y* a, unsigned int size) {
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
    sigmoid_derivativeKernel << <(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (dev_a, size);

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
cudaError_t CUDA_Class<Y>::square(Y* a, unsigned int size) {
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
    squareKernel << <(size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> > (dev_a, size);

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