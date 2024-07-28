/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*/

#ifndef __SKEWED_DATA_HANDKER_CUH
#define __SKEWED_DATA_HANDKER_CUH

#include <iostream>
#include <algorithm>
#include <cub/cub.cuh>
#include "common.cuh"

constexpr size_type BLOCK_SIZE = 512;

template<typename size_type>
__global__ void get_number_of_blocks_each_partition(
    size_type* num_blocks_each_partition,
    const size_type* build_global_histo, 
    const size_type build_table_size,
    const size_type num_partitions,
    const size_type hash_table_size
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    
    if (idx < num_partitions) {
        int partition_start = build_global_histo[idx];
        int partition_end = idx == num_partitions - 1 ? build_table_size : build_global_histo[idx+1];
        int partition_size = partition_end - partition_start; 

        if (partition_size == 0) {
            num_blocks_each_partition[idx] = 1; // use at least one block if the partition is empty 

        } else {
            num_blocks_each_partition[idx] = (partition_size - 1) / hash_table_size + 1; 
        }

    }
}

template<typename size_type>
__global__ void populate_index_array(
    const size_type* num_blocks_each_partition_scanned,
    const size_type num_partitions,
    size_type* partition_index_alt
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    
    if (idx < num_partitions) {
        partition_index_alt[num_blocks_each_partition_scanned[idx]] = 1; 
    }
}

template<typename size_type>
void handle_skewed_data(
    const size_type* build_global_histo, 
    const size_type build_tbl_size, 
    const size_type base_build_tbl_size_each_part, 
    const size_type num_partitions, 
    size_type* partition_index, 
    size_type* partition_index_alt,
    size_type* num_blocks_each_partition,
    size_type* num_blocks_each_partition_s,
    size_type* num_block_join_h, 
    void* temp_storage, 
    size_t temp_storage_bytes
) {
    // get number of blocks needed for each partition
    size_type num_blocks_get_block_size = ceil(1.0 * num_partitions / BLOCK_SIZE);  
    get_number_of_blocks_each_partition<size_type><<<num_blocks_get_block_size, BLOCK_SIZE>>>(
        num_blocks_each_partition, build_global_histo, build_tbl_size, num_partitions, base_build_tbl_size_each_part 
    );

    // scan number of blocks for each partition 
    cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes,
        num_blocks_each_partition, num_blocks_each_partition_s, num_partitions + 1
    );

    CUDA_RT_CALL( cudaMemcpy(num_block_join_h, num_blocks_each_partition_s + num_partitions, sizeof(size_type), cudaMemcpyDefault) );

    // populate partition index array each block to access
    populate_index_array<size_type><<<num_blocks_get_block_size, BLOCK_SIZE>>>(
        num_blocks_each_partition_s, num_partitions, partition_index_alt 
    );

    cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes, partition_index_alt, partition_index, *num_block_join_h);    
}

#endif //SKEWED_DATA_HANDKER_CUH