
#ifndef CUDA_CHECK
#define CUDA_CHECK

#include "cuda_runtime.h"

#include <iostream>
#include <cassert>

// CUDA error checker
// return:
// 1. information about error
// 2. file with error
// 3. line with error
#define CUDA_CHECK(ans) {cudaAssert(ans, __FILE__, __LINE__);}

inline void cudaAssert(cudaError_t ans, const char *file, unsigned long line, bool is_abort=true) {
    
    if(ans != cudaSuccess) {

        std::cerr << "GPU error: " << cudaGetErrorString(ans) << ' ' << file << ' ' << line << '\n';

        assert(is_abort);

    }

}

#endif // CUDA_CHECK_H