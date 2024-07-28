/* Copyright 2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#ifndef GENERATE_INPUT_TABLES_CUH
#define GENERATE_INPUT_TABLES_CUH

#include <type_traits>

#include <curand.h>
#include <curand_kernel.h>

#ifndef CURAND_CALL
#define CURAND_CALL( call )                                                                                                            \
{                                                                                                                                      \
    curandStatus_t curandStatus = call;                                                                                                \
    if ( CURAND_STATUS_SUCCESS != curandStatus )                                                                                       \
        fprintf(stderr, "ERROR: cuRAND call \"%s\" in line %d of file %s failed with %d.\n", #call, __LINE__, __FILE__, curandStatus); \
}
#endif

#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/distance.h>
#include <thrust/sort.h>
#include <thrust/set_operations.h>

#include "nvtx_helper.cuh"
#include "common.cuh"

__global__ void init_curand(curandState * state, const int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i < n ) {
        curand_init( 1234ULL, i, 0, state+i );
    }
}

template<typename key_type, typename size_type>
__global__ void init_build_tbl(
    key_type* const build_tbl, const size_type build_tbl_size,
    const key_type rand_max,
    const bool uniq_build_tbl_keys,
    key_type* const lottery, const size_type lottery_size,
    curandState * state, const int num_states)
{
    static_assert(std::is_signed<key_type>::value, "key_type needs to be signed for lottery to work");
    const int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const key_type stride = blockDim.x * gridDim.x;
    assert( start_idx < num_states );
    
    curandState localState = state[start_idx];
    
    for ( size_type idx = start_idx; idx < build_tbl_size; idx += stride )
    {
        const float x = curand_uniform(&localState);
        if ( uniq_build_tbl_keys ) {
            size_type lottery_idx = x*lottery_size;
            key_type lottery_val = -1;
            while ( -1 == lottery_val )
            {
                lottery_val = lottery[lottery_idx];
                if ( -1 != lottery_val ) {
                    lottery_val = atomicCAS( lottery + lottery_idx, lottery_val, -1 );
                }
                lottery_idx=(lottery_idx+1)%lottery_size;
            }
            build_tbl[idx] = lottery_val;
        }
        else {
            build_tbl[idx] = x*rand_max;
        }
    }
    state[start_idx] = localState;
}

template<typename key_type, typename size_type>
__global__ void init_probe_tbl(
    key_type* const probe_tbl, const size_type probe_tbl_size,
    const key_type* const build_tbl, const size_type build_tbl_size,
    const key_type* const lottery, const size_type lottery_size,
    const float selectivity,
    curandState * state, const int num_states)
{
    const int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_type stride = blockDim.x * gridDim.x;
    assert( start_idx < num_states );
    
    curandState localState = state[start_idx];
    
    for ( size_type idx = start_idx; idx < probe_tbl_size; idx += stride )
    {
        key_type val;
        float x = curand_uniform(&localState);
        if ( x <= selectivity ) {
            x = curand_uniform(&localState);
            size_type build_tbl_idx = x*build_tbl_size;
	    if (build_tbl_idx >= build_tbl_size) build_tbl_idx = build_tbl_size-1;
            val = build_tbl[build_tbl_idx];
        }
        else {
            x = curand_uniform(&localState);
            size_type lottery_idx = x*lottery_size;
            val = lottery[lottery_idx];
        }
        probe_tbl[idx] = val;
    }
    
    state[start_idx] = localState;
}

/**
 * generate_input_tables generates random integer input tables for database benchmarks.
 * 
 * generate_input_tables generates two random integer input tables for database benchmark
 * mainly designed to benchmark join operations. The templates key_type and value_type needed
 * to be builtin integer types (e.g. short, int, longlong) and key_type needs to be signed
 * as the lottery used internally relies on beeing able to use negative values to mark drawn
 * numbers. The tables need to be preallocated in a memory region accessible by the GPU
 * (e.g. device memory, zero copy memory or unified memory). Each value in the build table
 * will be from [0,rand_max] and if uniq_build_tbl_keys is true it is ensured that each value
 * will be uniq in the build table. Each value in the probe table will be also in the build
 * table with a propability of selectivity and a random number from
 * [0,rand_max] \setminus \{build_tbl\} otherwise.
 *
 * @param[out] build_tbl            The build table to generate. Usually the smaller table used to
 *                                  "build" the hash table in a hash based join implementation.
 * @param[in] build_tbl_size        number of keys in the build table
 * @param[out] probe_tbl            The probe table to generate. Usually the larger table used to
 *                                  probe into the hash table created from the build table.
 * @param[in] build_tbl_size        number of keys in the build table
 * @param[in] selectivity           propability with which an element of the probe table is
 *                                  present in the build table.
 * @param[in] rand_max              maximum random number to generate. I.e. random numbers are
 *                                  integers from [0,rand_max].
 * @param[in] uniq_build_tbl_keys   if each key in the build table should appear exactly once.
 */
template<typename key_type, typename size_type>
void generate_input_tables(
    key_type* const build_tbl,
    const size_type build_tbl_size,
    key_type* const probe_tbl,
    const size_type probe_tbl_size,
    const float selectivity,
    const key_type rand_max,
    const bool uniq_build_tbl_keys)
{
    static_assert(std::is_signed<key_type>::value, "key_type needs to be signed for lottery to work");
    // With large values of rand_max the a lot of temporary storage is needed for the lottery. At the expense of not beeing that accurate
    // with applying the selectivity an especially more memory efficient implementations would be to partition the random numbers into two
    // intervals and then let one table choose random numbers from only one interval and the other only select with selectivity propability from
    // the same interval and from the other in the other cases.
    PUSH_RANGE("generate_input_tables",0)
    const int block_size = 128;
    //maximize exposed parallelism while minimizing storage for curand state
    int num_blocks_init_build_tbl = 1;
    CUDA_RT_CALL( cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &num_blocks_init_build_tbl, init_build_tbl<key_type,size_type> , block_size, 0 ) );
    int num_blocks_init_probe_tbl = 1;
    CUDA_RT_CALL( cudaOccupancyMaxActiveBlocksPerMultiprocessor ( &num_blocks_init_probe_tbl, init_probe_tbl<key_type,size_type>, block_size, 0 ) );
    
    int dev_id = 0;
    CUDA_RT_CALL( cudaGetDevice( &dev_id ) );
    int num_sms = 0;
    CUDA_RT_CALL( cudaDeviceGetAttribute( &num_sms, cudaDevAttrMultiProcessorCount, dev_id ) );

    const int num_states = num_sms*std::max(num_blocks_init_build_tbl,num_blocks_init_probe_tbl)*block_size;
    curandState * devStates;
    CUDA_RT_CALL( cudaMalloc( &devStates, num_states*sizeof(curandState) ) );
    init_curand<<<((num_states-1)/block_size)+1,block_size>>>( devStates, num_states );
    CUDA_RT_CALL( cudaGetLastError() );
    CUDA_RT_CALL( cudaDeviceSynchronize() );
    
    key_type* build_tbl_sorted;
    CUDA_RT_CALL( cudaMalloc( &build_tbl_sorted, build_tbl_size*sizeof(key_type) ) );
    
    size_type lottery_size = rand_max < ( std::numeric_limits<key_type>::max() - 1 ) ? rand_max + 1 : rand_max;
    key_type* lottery;
    bool lottery_in_device_memory = true;
    size_t free_gpu_mem = 0;
    size_t total_gpu_mem = 0;
    CUDA_RT_CALL( cudaMemGetInfo ( &free_gpu_mem, &total_gpu_mem ) );
    if ( free_gpu_mem > lottery_size*sizeof(key_type) )
    {        
        CUDA_RT_CALL( cudaMalloc( &lottery, lottery_size*sizeof(key_type) ) );
    }
    else
    {
        CUDA_RT_CALL( cudaMallocHost( &lottery, lottery_size*sizeof(key_type) ) );
        lottery_in_device_memory=false;
    }
    
    if ( uniq_build_tbl_keys ) {
        thrust::sequence(thrust::device, lottery, lottery+lottery_size, 0);
    }
    
    init_build_tbl<<<num_blocks_init_build_tbl,block_size>>>( build_tbl, build_tbl_size, rand_max, uniq_build_tbl_keys, lottery, lottery_size, devStates, num_states );
    CUDA_RT_CALL( cudaGetLastError() );
    CUDA_RT_CALL( cudaDeviceSynchronize() );
    
    CUDA_RT_CALL( cudaMemcpy( build_tbl_sorted, build_tbl, build_tbl_size*sizeof(key_type), cudaMemcpyDeviceToDevice ) );
    
    thrust::sort(thrust::device, build_tbl_sorted, build_tbl_sorted + build_tbl_size);
    
    thrust::counting_iterator<key_type> first_lottery_elem(0);
    thrust::counting_iterator<key_type> last_lottery_elem = first_lottery_elem + lottery_size;
    key_type * lottery_end = thrust::set_difference(thrust::device, first_lottery_elem, last_lottery_elem, build_tbl_sorted, build_tbl_sorted + build_tbl_size, lottery);
    
    lottery_size = thrust::distance(lottery, lottery_end);
    
    init_probe_tbl<<<num_blocks_init_build_tbl,block_size>>>( probe_tbl, probe_tbl_size, build_tbl, build_tbl_size, lottery, lottery_size, selectivity, devStates, num_states );
    CUDA_RT_CALL( cudaGetLastError() );
    CUDA_RT_CALL( cudaDeviceSynchronize() );
    
    if ( lottery_in_device_memory ) {
        CUDA_RT_CALL( cudaFree( lottery ) );
    } else {
        CUDA_RT_CALL( cudaFreeHost( lottery ) );
    }
    CUDA_RT_CALL( cudaFree( build_tbl_sorted ) );
    CUDA_RT_CALL( cudaFree( devStates ) );
    POP_RANGE
}

template<typename key_type, typename size_type>
__global__ void linear_sequence(key_type* tbl, const size_type size)
{
    for (size_type i = threadIdx.x + blockDim.x * blockIdx.x; i < size; i += blockDim.x * gridDim.x) {
        tbl[i] = i;
    }
}

template<typename key_type, typename size_type>
void generate_input_tables_simple(
    key_type* const build_tbl,
    const size_type build_tbl_size,
    key_type* const probe_tbl,
    const size_type probe_tbl_size,
    const float selectivity,
    const key_type rand_max,
    const bool uniq_build_tbl_keys)
{
#if 0
    int block = 256;
    int grid = 1024;
    linear_sequence<<<grid, block>>>(build_tbl, build_tbl_size);
    cudaDeviceSynchronize();
    assert(build_tbl_size == probe_tbl_size);
    cudaMemcpy(probe_tbl, build_tbl, probe_tbl_size * sizeof(key_type), cudaMemcpyDefault);
#else
    key_type *build_h = (key_type*)malloc(build_tbl_size * sizeof(key_type));

    FILE *f = fopen("/dev/urandom", "r");
    if (f==NULL) {fputs("File error", stderr); exit(1);}
    key_type res;

    res = fread(build_h, sizeof(key_type), build_tbl_size, f);
    if (res != build_tbl_size) exit(1);
    cudaMemcpy(build_tbl, build_h, build_tbl_size * sizeof(key_type), cudaMemcpyDefault);
    assert(build_tbl_size == probe_tbl_size);
    cudaMemcpy(probe_tbl, build_h, probe_tbl_size * sizeof(key_type), cudaMemcpyDefault);

    free(build_h);
#endif
}

#endif //GENERATE_INPUT_TABLES_CUH
