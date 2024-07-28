/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*/

#ifndef __PARTITION_KERNELS_CUH
#define __PARTITION_KERNELS_CUH

#include <iostream>
#include <algorithm>
#include <cub/cub.cuh>
#include "hash_functions.cuh"
#include "common.cuh"
#include "../int_fastdiv/int_fastdiv.h"

constexpr size_type BLOCK_SIZE_PARTITION = 512;
constexpr size_type ELEM_PER_THREAD = 1024/BLOCK_SIZE_PARTITION;
constexpr size_type WARP_SIZE = 32; 

__host__ __device__ int divup(int a, int b) {
    return (a + b - 1) / b;
}

template<uint BIN_BITS>
struct bin_traits_t {
    static const uint WORD_BITS = 32;
    static const uint WORD_BINS = WORD_BITS / BIN_BITS;
    static const uint BIN_MASK = (1 << BIN_BITS) - 1;
};

template<uint BIN_BITS>
__device__ void inc_bit_bin(uint* sbins, int* bins, int nbins, int bin) {
    typedef bin_traits_t<BIN_BITS> bin_traits;
    uint iword = bin / bin_traits::WORD_BINS, ibin = bin % bin_traits::WORD_BINS;
    uint sh = ibin * BIN_BITS;
    uint old_word = atomicAdd(sbins + iword, 1 << sh), new_word = old_word + (1 << sh);
    if ((new_word >> sh & bin_traits::BIN_MASK) != 0) return;
    // overflow
    atomicAdd(&bins[bin], bin_traits::BIN_MASK + 1);
    for (uint dbin = 1; ibin + dbin < bin_traits::WORD_BINS && bin + dbin < nbins;
        ++dbin) {
        uint sh1 = (ibin + dbin) * BIN_BITS;
        if ((new_word >> sh1 & bin_traits::BIN_MASK) == 0) {
        // overflow
        atomicAdd(&bins[bin + dbin], bin_traits::BIN_MASK);
        } else {
        // correction
        atomicAdd(&bins[bin + dbin], -1);
        break;
        }
    }
}

template<int BIN_BITS>
struct bit_hist_op {
    typedef bin_traits_t<BIN_BITS> bin_traits;
    static __device__ int nwords(int nbins) {
        return divup(nbins, bin_traits::WORD_BINS);
    }
    static __device__  void update_bin
    (uint* sbins, int* bins, int nbins, int bin) {
        inc_bit_bin<BIN_BITS>(sbins, bins, nbins, bin);
    }
    static __device__ int bin_value(uint* sbins, int bin) {
        return (sbins[bin / bin_traits::WORD_BINS] >>
                (bin % bin_traits::WORD_BINS * BIN_BITS)) & bin_traits::BIN_MASK;
    }
};

template<>
struct bit_hist_op<32> {
    typedef bin_traits_t<32> bin_traits;
    static __device__ int nwords(int nbins) { return nbins; }
    static __device__  void update_bin
    (uint* sbins, int* bins, int nbins, int bin) {
        atomicAdd(&sbins[bin], 1);
    }
    static __device__ int bin_value(uint* sbins, int bin) { return sbins[bin]; }
};

template<typename hist_op, int n_unroll>
__global__ void big_histogram(int* bins, int nbins, const key_type* data, size_t n) {
    extern __shared__ int shared_bins[];
    uint* sbins = (uint*)&shared_bins;
    int nwords = hist_op::nwords(nbins);

    for (int j = threadIdx.x; j < nwords; j += blockDim.x) {
        sbins[j] = 0;
    }
    __syncthreads();
    
    key_type rdata[n_unroll];
    int nthreads = gridDim.x * blockDim.x;
    int stride = nthreads * n_unroll;
    int extra = nthreads * (n_unroll - 1);
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    // unrolled
    for (; i + extra < n; i += stride) {
    #pragma unroll
        for (int j = 0; j < n_unroll; ++j) {
            rdata[j] = data[i + nthreads * j];
        }
    #pragma unroll
        for (int j = 0; j < n_unroll; ++j) {
            int bin = (nbins - 1) & rdata[j]; 
            hist_op::update_bin(sbins, bins, nbins, bin);
        }
    }
    // remainder
    for (; i < n; i += nthreads) {
        int bin = (nbins - 1) & data[i]; 
        hist_op::update_bin(sbins, bins, nbins, bin);
    }
    __syncthreads();
    
    for (int j = threadIdx.x; j < nbins; j += blockDim.x) {
        int count = hist_op::bin_value(sbins, j);
        if (count > 0) atomicAdd(&bins[j], count);
    }
}

template<typename size_type>
__global__ void retrieve_histogram_of_pass_one(
    size_type* __restrict__ pass_one_histo, 
    const int num_partitions_p2, 
    const size_type* __restrict__ global_histo, 
    const int num_partitions)
{
    size_type start_i = threadIdx.x + blockIdx.x * blockDim.x; 
    
    for (size_type i = start_i; i < num_partitions; i += gridDim.x * blockDim.x) {
        atomicAdd(&pass_one_histo[i / num_partitions_p2], global_histo[i]);
    }
}

template<typename key_type, typename size_type, typename value_type, int n_unroll>
__global__ void histogram_pass_one(
    size_type* __restrict__ global_histo, 
    const key_type* __restrict__ table, 
    const size_type table_size, 
    const int num_partitions, 
    const int num_bit_p2)
{
    extern __shared__ size_type shared_histo[];

    // initialize shared histogram to 0 
    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
        shared_histo[i] = 0; 
    }

    __syncthreads();

    for (int i = threadIdx.x + blockIdx.x * blockDim.x * n_unroll; i < table_size; 
        i += gridDim.x * blockDim.x * n_unroll) {

        key_type ttable[n_unroll]; 

        #pragma unroll 
        for (int u = 0; u < n_unroll; ++u) {
            int j = i + u * blockDim.x;
            ttable[u] = j < table_size ? table[j] : ~0;
        }

        int prev_h = (((num_partitions - 1) << num_bit_p2) & ttable[0]) >> num_bit_p2; 
        int local_agg = 0; 

        #pragma unroll 
        for (int u = 0; u < n_unroll; ++u) {
            int j = i + u * blockDim.x;

            if (j >= table_size) break;

            int h = (((num_partitions - 1) << num_bit_p2) & ttable[u]) >> num_bit_p2; 
            if (h != prev_h) {
                atomicAdd(&shared_histo[prev_h], local_agg);
                local_agg = 1;
                prev_h = h; 
            } else {
                ++local_agg;  
            }
        }
        atomicAdd(&shared_histo[prev_h], local_agg);
    }

    __syncthreads();

    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
        atomicAdd(&global_histo[i], shared_histo[i]);
    }
}

template<typename key_type, typename size_type, typename value_type, int n_unroll, bool with_value>
__global__ void partition_pass_one(
    key_type* __restrict__ table_out,
    const key_type* __restrict__ table,
    value_type* __restrict__ value_out,
    const value_type* __restrict__ value,
    const size_type table_size, 
    size_type* __restrict__ global_histo_scanned,
    const int num_partitions, 
    int_fastdiv nth_partition, 
    const int num_bit_p2)
{
    extern __shared__ char shared_mem[];
    key_type *table_shared = (key_type*)shared_mem; 
    size_type *histo_shared_in = (size_type*)(table_shared + n_unroll * BLOCK_SIZE_PARTITION); 
    size_type *histo_shared_out = (size_type*)(histo_shared_in + num_partitions); 
    value_type *value_shared = nullptr;
    if (with_value) value_shared = (value_type*)(histo_shared_out + num_partitions);

    key_type ttable[n_unroll]; 
    value_type tvalue[n_unroll]; 

    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
        histo_shared_in[i] = 0; 
    }

    __syncthreads(); 
    
    // load into shared memory and construct per block histogram 
    int idx = threadIdx.x + blockIdx.x * blockDim.x * n_unroll;
    if (idx < table_size) {
        #pragma unroll 
        for (int u = 0; u < n_unroll; ++u) {
            int j = idx + u * blockDim.x;
            ttable[u] = j < table_size ? table[j] : ~0;
            if (with_value) {
                tvalue[u] = j < table_size ? value[j] : ~0;
            }
        }

        int prev_h = (((num_partitions - 1) << num_bit_p2) & ttable[0]) >> num_bit_p2; 
        int local_agg = 0; 

        #pragma unroll 
        for (int u = 0; u < n_unroll; ++u) {
            int j = idx + u * blockDim.x;
            if (j >= table_size) break;

            int h = (((num_partitions - 1) << num_bit_p2) & ttable[u]) >> num_bit_p2; 
            if (h != prev_h) {
                atomicAdd(&histo_shared_in[prev_h], local_agg);
                local_agg = 1;
                prev_h = h; 
            } else {
                ++local_agg;  
            }
        }
        atomicAdd(&histo_shared_in[prev_h], local_agg);
    }

    __syncthreads();

    typedef cub::BlockScan<size_type, BLOCK_SIZE_PARTITION> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    // each thread handles ELEM_PER_THREAD elements to support upto 1024 partitions 
    size_type temp_histo[ELEM_PER_THREAD];
 
    for (int i = 0; i < ELEM_PER_THREAD; ++i) {
        if (ELEM_PER_THREAD * threadIdx.x + i < num_partitions) {
            temp_histo[i] = histo_shared_in[ELEM_PER_THREAD * threadIdx.x + i]; 
        } else {
            temp_histo[i] = 0;
        }
    }

    __syncthreads();

    BlockScan(temp_storage).ExclusiveSum(temp_histo, temp_histo);

    __syncthreads();

    for (int i = 0; i < ELEM_PER_THREAD; ++i) {
        if (ELEM_PER_THREAD * threadIdx.x + i < num_partitions) {
            histo_shared_out[ELEM_PER_THREAD * threadIdx.x + i] = temp_histo[i]; 
        } 
    }

    __syncthreads();

    if (idx < table_size) {
        #pragma unroll 
        for (int u = 0; u < n_unroll; ++u) {
            int j = idx + u * blockDim.x;
            if (j >= table_size) break;

            int h = (((num_partitions - 1) << num_bit_p2) & ttable[u]) >> num_bit_p2; 

            int pos = atomicAdd(&histo_shared_out[h], 1);
            table_shared[pos] = ttable[u];
            if (with_value) {
                value_shared[pos] = tvalue[u];
            }
        }
    }

    // reuse histo_shared_in storing start location for write back to partitioned table 
    for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
        int num_elem = histo_shared_in[i]; 
        histo_shared_in[i] = atomicAdd(&global_histo_scanned[i], num_elem);
    }

    __syncthreads(); 

    // flush shared memory to global memory
    for (int i = threadIdx.x / nth_partition; i < num_partitions; i += BLOCK_SIZE_PARTITION / nth_partition) {
        int row_off = threadIdx.x % nth_partition;
        int shared_pos_end = histo_shared_out[i];
        int shared_pos_start = i == 0 ? 0 : histo_shared_out[i-1];
        int global_pos_start = histo_shared_in[i];
        int nelem = shared_pos_end - shared_pos_start; 

        while (row_off < nelem) {
            table_out[global_pos_start + row_off] = table_shared[shared_pos_start + row_off];
            if (with_value) {
                value_out[global_pos_start + row_off] = value_shared[shared_pos_start + row_off];
            }
            row_off += nth_partition;
        }
    }
}

template<typename key_type, typename size_type, typename value_type, int n_unroll>
__global__ void histogram_pass_two(
    size_type* __restrict__ histo, 
    const key_type* __restrict__ table, 
    const size_type table_size, 
    const int num_partitions_p1, 
    const int num_partitions_p2, 
    const size_type* __restrict__ global_histo_p1, 
    const size_type* __restrict__ num_blocks_each_partition_s, 
    const size_type* __restrict__ partition_index, 
    const int base_num_elem_each_block)
{
    __shared__ int partition_start_shared;
    __shared__ int partition_end_shared;

    int partition_1 = partition_index[blockIdx.x];
    extern __shared__ size_type shared_histo[];

    // initialize shared histogram to 0 
    for (int i = threadIdx.x; i < num_partitions_p2; i += blockDim.x) {
        shared_histo[i] = 0; 
    }

    if (threadIdx.x == 0) {
        partition_start_shared = global_histo_p1[partition_1]; 
        partition_end_shared = (partition_1) == num_partitions_p1 - 1
                                ? table_size : global_histo_p1[partition_1 + 1];
        
        int num_partition_prev_blk = partition_1 == 0 ? 0 : num_blocks_each_partition_s[partition_1-1]; 
        int partition_offset = blockIdx.x - num_partition_prev_blk;
        partition_start_shared += partition_offset * base_num_elem_each_block; 
        int partition_end_shared_temp = partition_start_shared + base_num_elem_each_block; 
        if (partition_end_shared_temp < partition_end_shared) {
            partition_end_shared = partition_end_shared_temp; 
        }                        
    }
    
    __syncthreads();

    int partition_start = partition_start_shared;
    int partition_end = partition_end_shared;

    int idx = partition_start + threadIdx.x;
    if (idx < partition_end) {
        key_type ttable[n_unroll]; 

        #pragma unroll 
        for (int u = 0; u < n_unroll; ++u) {
            int j = idx + u * blockDim.x;
            ttable[u] = j < partition_end ? table[j] : ~0;
        }

        int prev_h = (num_partitions_p2 - 1) & ttable[0]; 
        int local_agg = 0; 

        #pragma unroll 
        for (int u = 0; u < n_unroll; ++u) {
            int j = idx + u * blockDim.x;

            if (j >= partition_end) break;

            int h = (num_partitions_p2 - 1) & ttable[u]; 
            if (h != prev_h) {
                atomicAdd(&shared_histo[prev_h], local_agg);
                local_agg = 1;
                prev_h = h; 
            } else {
                ++local_agg;  
            }
        }
        atomicAdd(&shared_histo[prev_h], local_agg);
    }

    __syncthreads();

    for (int i = threadIdx.x; i < num_partitions_p2; i += blockDim.x) {
        atomicAdd(&histo[partition_1 * num_partitions_p2 + i], shared_histo[i]);
    }
}

template<typename key_type, typename size_type, typename value_type, int n_unroll, bool with_value>
__global__ void partition_pass_two(
    key_type* __restrict__ table_out,
    const key_type* __restrict__ table,
    value_type* __restrict__ value_out,
    const value_type* __restrict__ value,
    const size_type table_size, 
    const size_type* __restrict__ global_histo_p1, 
    size_type* __restrict__ global_histo_scanned, 
    const int num_partitions_p1, 
    const int num_partitions_p2,
    int_fastdiv nth_partition, 
    const size_type* __restrict__ num_blocks_each_partition_s, 
    const size_type* __restrict__ partition_index, 
    const int base_num_elem_each_block)
{
    extern __shared__ char shared_mem[];
    key_type *table_shared = (key_type*)shared_mem; 
    size_type *histo_shared_in = (size_type*)(table_shared + n_unroll * BLOCK_SIZE_PARTITION); 
    size_type *histo_shared_out = (size_type*)(histo_shared_in + num_partitions_p2); 
    value_type *value_shared = nullptr;
    if (with_value) value_shared = (value_type*)(histo_shared_out + num_partitions_p2);

    key_type ttable[n_unroll]; 
    value_type tvalue[n_unroll]; 

    __shared__ int partition_start_shared;
    __shared__ int partition_end_shared; 
    int partition_1 = partition_index[blockIdx.x];

    for (int i = threadIdx.x; i < num_partitions_p2; i += blockDim.x) {
        histo_shared_in[i] = 0; 
    }

    if (threadIdx.x == 0) {
        partition_start_shared = global_histo_p1[partition_1]; 
        partition_end_shared = (partition_1) == num_partitions_p1 - 1 ? table_size : global_histo_p1[partition_1 + 1];
        
        int num_partition_prev_blk = partition_1 == 0 ? 0 : num_blocks_each_partition_s[partition_1-1]; 
        int partition_offset = blockIdx.x - num_partition_prev_blk;
        partition_start_shared += partition_offset * base_num_elem_each_block; 
        int partition_end_shared_temp = partition_start_shared + base_num_elem_each_block; 
        if (partition_end_shared_temp < partition_end_shared) {
            partition_end_shared = partition_end_shared_temp; 
        }              
    }

    __syncthreads(); 
    
    int partition_start = partition_start_shared;
    int partition_end = partition_end_shared;

    int idx = partition_start + threadIdx.x; 
    if (idx < partition_end) {
        #pragma unroll 
        for (int u = 0; u < n_unroll; ++u) {
            int j = idx + u * blockDim.x;
            ttable[u] = j < partition_end ? table[j] : ~0;
            if (with_value) {
                tvalue[u] = j < partition_end ? value[j] : ~0;
            }
        }

        int prev_h = (num_partitions_p2 - 1) & ttable[0]; 
        int local_agg = 0; 

        #pragma unroll 
        for (int u = 0; u < n_unroll; ++u) {
            int j = idx + u * blockDim.x;
            if (j >= partition_end) break;

            int h = (num_partitions_p2 - 1) & ttable[u]; 
            if (h != prev_h) {
                atomicAdd(&histo_shared_in[prev_h], local_agg);
                local_agg = 1;
                prev_h = h; 
            } else {
                ++local_agg;  
            }
        }
        atomicAdd(&histo_shared_in[prev_h], local_agg);
    }

    __syncthreads(); 

    typedef cub::BlockScan<size_type, BLOCK_SIZE_PARTITION> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    // each thread handles ELEM_PER_THREAD elements to support upto 1024 partitions 
    size_type temp_histo[ELEM_PER_THREAD];

    for (int i = 0; i < ELEM_PER_THREAD; ++i) {
        if (ELEM_PER_THREAD * threadIdx.x + i < num_partitions_p2) {
            temp_histo[i] = histo_shared_in[ELEM_PER_THREAD * threadIdx.x + i]; 
        } else {
            temp_histo[i] = 0;
        }
    }

    __syncthreads();

    BlockScan(temp_storage).ExclusiveSum(temp_histo, temp_histo);

    __syncthreads();

    for (int i = 0; i < ELEM_PER_THREAD; ++i) {
        if (ELEM_PER_THREAD * threadIdx.x + i < num_partitions_p2) {
            histo_shared_out[ELEM_PER_THREAD * threadIdx.x + i] = temp_histo[i]; 
        } 
    }
    
    __syncthreads();

    if (idx < partition_end) {
        #pragma unroll 
        for (int u = 0; u < n_unroll; ++u) {
            int j = idx + u * blockDim.x;
            if (j >= partition_end) break; 

            int h = (num_partitions_p2 - 1) & ttable[u]; 

            int pos = atomicAdd(&histo_shared_out[h], 1);
            table_shared[pos] = ttable[u];
            if (with_value) {
                value_shared[pos] = tvalue[u];
            }
        }
    }

    __syncthreads(); 

    // reuse histo_shared_in storing start location for write back to partitioned table 
    for (int i = threadIdx.x; i < num_partitions_p2; i += blockDim.x) {
        int num_elem = histo_shared_in[i]; 
        histo_shared_in[i] = atomicAdd(&global_histo_scanned[partition_1 * num_partitions_p2 + i], num_elem);
    }

    __syncthreads(); 

    // flush shared memory to global memory
    for (int i = threadIdx.x / nth_partition; i < num_partitions_p2; i += BLOCK_SIZE_PARTITION / nth_partition) {
        int row_off = threadIdx.x % nth_partition;
        int shared_pos_end = histo_shared_out[i];
        int shared_pos_start = i == 0 ? 0 : histo_shared_out[i-1];
        int global_pos_start = histo_shared_in[i];
        int nelem = shared_pos_end - shared_pos_start; 

        while (row_off < nelem) {
            table_out[global_pos_start + row_off] = table_shared[shared_pos_start + row_off];
            if (with_value) {
                value_out[global_pos_start + row_off] = value_shared[shared_pos_start + row_off];
            }
            row_off += nth_partition;
        }
    }
}

#endif //PARTITION_KERNLES_CUH