
#include "cuda_check.cu"
#include "cuda_runtime.h"
#include "linear_regression_cuda.hpp"

#include <iostream>



// calculate numerator and denominator which are then used to calculate slope and intercept
static __global__ void calculateNumeratorAndDenominator(const float *x, const float *y, const float x_mean, const float y_mean, float *num, float *dem, const std::size_t n) {
    
    extern __shared__ float shared_diff[];

    int tid = threadIdx.x;
    int idx = tid + blockDim.x * blockIdx.x;

    // split shared memory to calculate the sum of the differences
    // (xi - x_mean) * (yi - y_mean) - numerator
    // (xi - x_mean)^2               - denominator
    float *shared_num = shared_diff; 
    float *shared_dem = shared_diff + blockDim.x;

    shared_num[tid] = (idx < n) ? (x[idx] - x_mean) * (y[idx] - y_mean) : 0.0f;
    shared_dem[tid] = (idx < n) ? (x[idx] - x_mean) * (x[idx] - x_mean) : 0.0f;

    __syncthreads();


    // block-wise reduction to calculate the sums
    for(int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2*stride*tid;

        if(index < blockDim.x) {
            shared_num[tid] = shared_num[tid + stride];
            shared_dem[tid] = shared_dem[tid + stride];
        }
        __syncthreads();

    }

    // Atomic operations to get the sum of the nums and dems from all blocks
    if(tid == 0) {
        atomicAdd(num, shared_num[0]);
        atomicAdd(dem, shared_dem[0]);
    }

}

// kernel to calculate the sums of xi and yi which are then used to calculate means
static __global__ void calculateSums(float *x, float *y, float *x_sums, float *y_sums, const std::size_t n) {

    extern __shared__ float mean_shared_error[];

    int tid = threadIdx.x;
    int idx = tid + blockIdx.x * blockDim.x;

    // split shared memory for x and y
    float *shared_x = mean_shared_error;
    float *shared_y = mean_shared_error + blockDim.x;

    shared_x[tid] = (idx < n) ? x[idx] : 0;
    shared_y[tid] = (idx < n) ? y[idx] : 0;

    __syncthreads();

    // block-wise reduction to calculate partial sums of x and y
    for(int stride = 1; stride < blockDim.x; stride *= 2) {

        int index = 2 * stride * tid;

        if(index < blockDim.x) {
            shared_x[index] += shared_x[index + stride];
            shared_y[index] += shared_y[index + stride];
        }
        __syncthreads();    

    }

    // atomic operation to sum partial sums from blocks
    if(tid == 0) {
        atomicAdd(x_sums, shared_x[0]);
        atomicAdd(y_sums, shared_y[0]);
    }


}

// kernel to calculate squared error which are then used to calculate root-mean-squared-error (RMSE)
static __global__ void calculateSquaredError(float *y, float *pred, float *squared_error, const std::size_t n) {
    
    extern __shared__ float mse_shared_mem[];
    
    int tid = threadIdx.x;
    int idx = tid + blockIdx.x * blockDim.x;

    mse_shared_mem[tid] = 0.0f;
    __syncthreads();

    if(idx < n) {
        float diff = y[idx] - pred[idx];
        mse_shared_mem[tid] = diff * diff;
    }

    __syncthreads();

    // block-wisereduction to get the sum of the different between y and y_pred squared
    for(int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = stride * tid * 2;

        if(index < blockDim.x) {
            mse_shared_mem[index] = mse_shared_mem[index + stride];
        }

        __syncthreads();
    }


    if(tid == 0) {
        atomicAdd(squared_error, mse_shared_mem[0]);
    }

}

static __global__ void makePredictions(const float *x, float *predictions, const float slope, const float intercept, const std::size_t n) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if(idx < n) {
        predictions[idx] = slope * x[idx] + intercept; // a default form to calculate pred_y = k*x + b (only for one x)
    }



}


LinearModel::LinearModel(
    float *x_train,
    float *y_train,
    float *x_test,
    float *y_test,
    const std::size_t train_size,
    const std::size_t test_size
    ) {

    this->is_trained = false;

    // CUDA auxiliary variables
    this->block_size = 256;
    this->train_size = train_size;
    this->test_size = test_size;
    this->n = test_size + train_size;
    this->grid_size_train = (train_size + block_size - 1) / block_size;
    this->grid_size_test = (test_size + block_size - 1) / block_size;
    this->shared_mem_size = 2 * block_size *sizeof(int);

    //GPU memory
    CUDA_CHECK(cudaMalloc(&d_x_train, train_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_train, train_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x_test, test_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_test, test_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_predictions, test_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x_train, x_train, train_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_train, y_train, train_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x_test, x_test, test_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_test, y_test, test_size * sizeof(float), cudaMemcpyHostToDevice));
        

}

LinearModel::~LinearModel() {

    // freeing CPU-memory
    delete[] h_predictions;

    // freeing GPU-memory
    CUDA_CHECK(cudaFree(d_x_train));
    CUDA_CHECK(cudaFree(d_y_train));
    CUDA_CHECK(cudaFree(d_x_test));
    CUDA_CHECK(cudaFree(d_y_test));
    CUDA_CHECK(cudaFree(d_predictions));
}

void LinearModel::fit(){
    
    float *d_x_mean, *d_y_mean;

    // x_mean and y_mean for kernel
    CUDA_CHECK(cudaMalloc((void**)&d_x_mean, sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y_mean, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_x_mean, 0.0f, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_y_mean, 0.0f, sizeof(float)));


    // calculate means
    calculateSums<<<grid_size_train, block_size, shared_mem_size>>>(d_x_train, d_y_train, d_x_mean, d_y_mean, train_size);

    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float x_mean, y_mean;

    CUDA_CHECK(cudaMemcpy(&x_mean, d_x_mean, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&y_mean, d_y_mean, sizeof(float), cudaMemcpyDeviceToHost));

    x_mean = x_mean / train_size;
    y_mean = y_mean / train_size;

    // calculate coefficients
    float *d_numerator, *d_denominator;
    CUDA_CHECK(cudaMalloc((void**)&d_numerator, sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_denominator, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_numerator, 0.0f, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_denominator, 0.0f, sizeof(float)));

    calculateNumeratorAndDenominator<<<grid_size_train, block_size, shared_mem_size>>>(d_x_train, d_y_train, x_mean, y_mean, d_numerator, d_denominator, train_size);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float numerator, denominator;

    CUDA_CHECK(cudaMemcpy(&numerator, d_numerator, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&denominator, d_denominator, sizeof(float), cudaMemcpyDeviceToHost));

    this->slope = numerator / denominator;
    this->intercept = y_mean - x_mean * slope;

    // make test prediction to calculate RMSE
    makePredictions<<<grid_size_test, block_size>>>(d_x_test, d_predictions, this->slope, this->intercept, test_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    

    this->is_trained = true;
}


float LinearModel::RMSE() {

    // if model doesnt fit, we cannot get a metric
    if(!is_trained) {
        return -1.0f;
    }

    // calculate squared error
    float *d_rmse;
    CUDA_CHECK(cudaMalloc((void**)&d_rmse, sizeof(float)));

    calculateSquaredError<<<grid_size_test, block_size, shared_mem_size>>>(d_y_test, d_predictions, d_rmse, test_size);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // calculate root-mean-squared-error
    float rmse;
    CUDA_CHECK(cudaMemcpy(&rmse, d_rmse, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_rmse));

    return sqrt(rmse / test_size);
}


void LinearModel::predict(float *x, float *dst, const std::size_t n) {

    // allocate memory for new predictions
    float *d_x_new;
    float *d_predictions_new;
    CUDA_CHECK(cudaMalloc((void**)&d_x_new, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_predictions_new, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x_new, x, n * sizeof(float), cudaMemcpyHostToDevice));

    // new grid size for new list of x
    int grid_size = (n + block_size - 1) / block_size; 
    
    makePredictions<<<grid_size, block_size, shared_mem_size>>>(d_x_new, d_predictions_new, slope, intercept, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dst, d_predictions_new, n * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x_new));
    CUDA_CHECK(cudaFree(d_predictions_new));
}

