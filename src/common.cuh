/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*/

#ifndef __COMMON_CUH
#define __COMMON_CUH

#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL(call)                                                                         \
{                                                                                                  \
    cudaError_t cudaStatus = call;                                                                 \
    if (cudaSuccess != cudaStatus)                                                                 \
        fprintf(stderr, "ERROR: CUDA RT call \"%s\" in line %d of file %s failed with %s (%d).\n", \
                        #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus);    \
}
#endif

typedef int size_type;

#ifdef USE_8B_KEYS
    typedef long long key_type;
    typedef long long value_type;
    using hasher = default_hash<long long>;
    constexpr size_type N_UNROLL = 8; 
    typedef longlong2 joined_type; 
    typedef longlong2 hash_map_type; 
#else
    typedef int key_type;
    typedef int value_type;
    using hasher = default_hash<int>;
    constexpr size_type N_UNROLL = 16; 
    typedef int2 joined_type; 
    typedef int2 hash_map_type; 
#endif

constexpr size_type DATA_SKEW_OVERALLOCATE_FACTOR = 2; 
constexpr size_type N_UNROLL_W_VAL = N_UNROLL / 2; 
constexpr size_type BIG_HISTO_BIT_SIZE = 4; 

__device__ __inline__
long long atomicCAS(long long* address, long long compare, long long val)
{
    return (long long)atomicCAS((unsigned long long*)address, (unsigned long long)compare, (unsigned long long)val);
}

#endif //COMMON_CUH
