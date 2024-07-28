/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*/

#ifndef __PARTITIONED_JOIN_CUH
#define __PARTITIONED_JOIN_CUH

#include <iostream>
#include <algorithm>
#include <cub/cub.cuh>
#include "hash_functions.cuh"
#include "common.cuh"
#include "int_fastdiv.h"
#include "hash_join_kernels.cuh"
#include "partition_kernels.cuh"

void combined_histogram_pass(
    size_type* __restrict__ histo, 
    size_type* __restrict__ histo_pass_one, 
    const key_type* __restrict__ input_table, 
    const size_type table_size, 
    const int num_partitions, 
    const int num_partitions_p1, 
    const int num_partitions_p2, 
    const bool radix_partition) 
{
    typedef bin_traits_t<BIG_HISTO_BIT_SIZE> bin_traits;
    int shm_sz = divup(num_partitions, bin_traits::WORD_BINS) * sizeof(uint);
    int num_block = ceil(1.0 * table_size / (BLOCK_SIZE_PARTITION * N_UNROLL));

    big_histogram<bit_hist_op<BIG_HISTO_BIT_SIZE>, N_UNROLL>
    <<<num_block, BLOCK_SIZE_PARTITION, shm_sz>>>(
        histo, num_partitions, input_table, table_size
    );

    if (radix_partition) {
        int num_block_retrieve_histogram = ceil(1.0 * num_partitions / BLOCK_SIZE_PARTITION);
        retrieve_histogram_of_pass_one
        <<<num_block_retrieve_histogram, BLOCK_SIZE_PARTITION>>>(
            histo_pass_one, num_partitions_p2, histo, num_partitions
        ); 
    }
}

template<typename key_type, typename size_type, typename value_type>
void partition_input_tables_pass_one(
    key_type* __restrict__ partitioned_table, 
    value_type* __restrict__ partitioned_value, 
    size_type* __restrict__ histo, 
    size_type* __restrict__ histo_scanned, 
    const size_type histo_size, 
    const key_type* __restrict__ input_table, 
    const value_type* __restrict__ input_value,
    const size_type table_size, 
    const int num_partitions, 
    void* __restrict__ scan_temp_storage, 
    size_t temp_storage_bytes, 
    const bool with_value, 
    const int num_bits_p2, 
    const bool skip_histogram_kernel)
{
    int num_unroll = with_value ? N_UNROLL_W_VAL : N_UNROLL; 
    int num_block = ceil(1.0 * table_size / (BLOCK_SIZE_PARTITION * num_unroll));

    if (!skip_histogram_kernel) {
        if (with_value) {
            histogram_pass_one<key_type, size_type, value_type, N_UNROLL_W_VAL>
            <<<num_block, BLOCK_SIZE_PARTITION, num_partitions*sizeof(size_type)>>>
            (
                histo, input_table, table_size, num_partitions, num_bits_p2
            );
        } else {
            histogram_pass_one<key_type, size_type, value_type, N_UNROLL>
            <<<num_block, BLOCK_SIZE_PARTITION, num_partitions*sizeof(size_type)>>>
            (
                histo, input_table, table_size, num_partitions, num_bits_p2
            );
        }
    }

    cub::DeviceScan::ExclusiveSum(scan_temp_storage, temp_storage_bytes, histo, histo_scanned, num_partitions);
    // save a copy for global histogram, will be used in join kernel 
    cudaMemcpy(histo, histo_scanned, num_partitions * sizeof(size_type), cudaMemcpyDeviceToDevice); 

    if (with_value) {
        partition_pass_one<key_type, size_type, value_type, N_UNROLL_W_VAL, true>
        <<<num_block, BLOCK_SIZE_PARTITION, BLOCK_SIZE_PARTITION*N_UNROLL_W_VAL*sizeof(key_type) +
        BLOCK_SIZE_PARTITION*N_UNROLL_W_VAL*sizeof(value_type) + 2*num_partitions*sizeof(size_type)>>>
        (
            partitioned_table, input_table, partitioned_value, input_value, 
            table_size, histo_scanned, num_partitions, WARP_SIZE, num_bits_p2
        );
    } else {
        partition_pass_one<key_type, size_type, value_type, N_UNROLL, false>
        <<<num_block, BLOCK_SIZE_PARTITION, BLOCK_SIZE_PARTITION*N_UNROLL*sizeof(key_type) + 
        2*num_partitions*sizeof(size_type)>>>
        (
            partitioned_table, input_table, nullptr, nullptr, 
            table_size, histo_scanned, num_partitions, WARP_SIZE, num_bits_p2
        );
    }
}

template<typename key_type, typename size_type, typename value_type>
void partition_input_tables_pass_two(
    key_type* __restrict__ partitioned_table, 
    value_type* __restrict__ partitioned_value, 
    size_type* __restrict__ histo, 
    size_type* __restrict__ histo_scanned, 
    const size_type histo_size, 
    const key_type* __restrict__ input_table, 
    const value_type* __restrict__ input_value,
    const size_type table_size, 
    const int num_partitions_1, 
    const int num_partitions_2, 
    const size_type* __restrict__ num_blocks_each_partition_s, 
    const size_type* __restrict__ partition_index, 
    const size_type num_blocks, 
    void* __restrict__ scan_temp_storage, 
    size_t temp_storage_bytes, 
    size_type* __restrict__ global_histo_p1, 
    const bool with_value, 
    const bool skip_histogram_kernel)
{
    int num_partitions = num_partitions_1 * num_partitions_2; 

    if (!skip_histogram_kernel) {
        if (with_value) {
            histogram_pass_two<key_type, size_type, value_type, N_UNROLL_W_VAL>
            <<<num_blocks, BLOCK_SIZE_PARTITION, num_partitions_2*sizeof(size_type)>>>
            (
                histo, input_table, table_size, num_partitions_1, 
                num_partitions_2, global_histo_p1, 
                num_blocks_each_partition_s, partition_index, BLOCK_SIZE_PARTITION * N_UNROLL_W_VAL
            );    
        } else {
            histogram_pass_two<key_type, size_type, value_type, N_UNROLL>
            <<<num_blocks, BLOCK_SIZE_PARTITION, num_partitions_2*sizeof(size_type)>>>
            (
                histo, input_table, table_size, num_partitions_1, 
                num_partitions_2, global_histo_p1,
                num_blocks_each_partition_s, partition_index, BLOCK_SIZE_PARTITION * N_UNROLL
            );
        }
    }

    cub::DeviceScan::ExclusiveSum(scan_temp_storage, temp_storage_bytes, histo, histo_scanned, num_partitions);
    // save a copy for global histogram, will be used in join kernel 
    cudaMemcpy(histo, histo_scanned, num_partitions * sizeof(size_type), cudaMemcpyDeviceToDevice); 

    if (with_value) {
        partition_pass_two<key_type, size_type, value_type, N_UNROLL_W_VAL, true>
        <<<num_blocks, BLOCK_SIZE_PARTITION, BLOCK_SIZE_PARTITION*N_UNROLL_W_VAL*sizeof(key_type) + 
        BLOCK_SIZE_PARTITION*N_UNROLL_W_VAL*sizeof(value_type) + 2*num_partitions_2*sizeof(size_type)>>>
        (
            partitioned_table, input_table, partitioned_value, input_value, table_size, global_histo_p1, 
            histo_scanned, num_partitions_1, num_partitions_2, WARP_SIZE, 
            num_blocks_each_partition_s, partition_index, BLOCK_SIZE_PARTITION * N_UNROLL_W_VAL
        );
    } else {
        partition_pass_two<key_type, size_type, value_type, N_UNROLL, false>
        <<<num_blocks, BLOCK_SIZE_PARTITION, BLOCK_SIZE_PARTITION*N_UNROLL*sizeof(key_type) + 
        2*num_partitions_2*sizeof(size_type)>>>
        (
            partitioned_table, input_table, nullptr, nullptr, table_size, global_histo_p1, 
            histo_scanned, num_partitions_1, num_partitions_2, WARP_SIZE, 
            num_blocks_each_partition_s, partition_index, BLOCK_SIZE_PARTITION * N_UNROLL
        );
    }
}

template<typename key_type, typename size_type, typename value_type, typename joined_type>
void shared_memory_hash_join(
    joined_type* __restrict__ joined_table, 
    const key_type* __restrict__ build_table, 
    const value_type* __restrict__ build_table_val,
    const size_type build_table_size, 
    const key_type* __restrict__ probe_table, 
    const value_type* __restrict__ probe_table_val,
    const size_type probe_table_size, 
    const size_type base_build_tbl_size_each_part,  
    size_type* __restrict__ global_index, 
    const size_type* __restrict__ num_blocks_each_partition_s, 
    const size_type* __restrict__ partition_index, 
    const size_type join_num_block, 
    const int num_partitions, 
    const bool uniq_build_tbl_keys, 
    const size_type* __restrict__ build_acc_loc, 
    const size_type* __restrict__ probe_acc_loc, 
    const bool with_value, 
    const size_type joined_size) 
{
    size_type hash_table_size = ceil(1.0 * base_build_tbl_size_each_part / HASH_TABLE_OCC); 
    size_type output_shared_mem_size = ceil(1.0 * joined_size / num_partitions);
    hasher hash_function;

    if (uniq_build_tbl_keys) {
        if (with_value) {
            hash_join<key_type, size_type, value_type, joined_type, hasher, true, true>
            <<<join_num_block, BLOCK_SIZE_JOIN, 
            hash_table_size*sizeof(hash_map_type)+output_shared_mem_size*sizeof(joined_type)>>>
            (
                joined_table, build_table, build_table_val, build_table_size, 
                probe_table, probe_table_val, probe_table_size,
                num_partitions, base_build_tbl_size_each_part, global_index, 
                num_blocks_each_partition_s, partition_index, hash_function, 
                hash_table_size, build_acc_loc, probe_acc_loc, output_shared_mem_size
            );
        } else { 
            hash_join<key_type, size_type, value_type, joined_type, hasher, true, false>
            <<<join_num_block, BLOCK_SIZE_JOIN, 
            hash_table_size*sizeof(hash_map_type)+output_shared_mem_size*sizeof(joined_type)>>>
            (
                joined_table, build_table, build_table_val, build_table_size, 
                probe_table, probe_table_val, probe_table_size,
                num_partitions, base_build_tbl_size_each_part, global_index, 
                num_blocks_each_partition_s, partition_index, hash_function, 
                hash_table_size, build_acc_loc, probe_acc_loc, output_shared_mem_size
            );
        }               
    } else {
        if (with_value) {
            hash_join<key_type, size_type, value_type, joined_type, hasher, false, true>
            <<<join_num_block, BLOCK_SIZE_JOIN, 
            hash_table_size*sizeof(hash_map_type)+output_shared_mem_size*sizeof(joined_type)>>>
            (
                joined_table, build_table, build_table_val, build_table_size, 
                probe_table, probe_table_val, probe_table_size,
                num_partitions, base_build_tbl_size_each_part, global_index, 
                num_blocks_each_partition_s, partition_index, hash_function, 
                hash_table_size, build_acc_loc, probe_acc_loc, output_shared_mem_size
            );
        } else { 
            hash_join<key_type, size_type, value_type, joined_type, hasher, false, false>
            <<<join_num_block, BLOCK_SIZE_JOIN, 
            hash_table_size*sizeof(hash_map_type)+output_shared_mem_size*sizeof(joined_type)>>>
            (
                joined_table, build_table, build_table_val, build_table_size, 
                probe_table, probe_table_val, probe_table_size,
                num_partitions, base_build_tbl_size_each_part, global_index, 
                num_blocks_each_partition_s, partition_index, hash_function, 
                hash_table_size, build_acc_loc, probe_acc_loc, output_shared_mem_size
            );
        }    
    }
}

void set_shared_memory_config(
    const bool uniq_build_tbl_keys, 
    const bool with_value) 
{
    const int max_shm_precentage = 100; // set shared memory size to 96KiB (100% of maximum smem)

    CUDA_RT_CALL( cudaFuncSetAttribute(
        big_histogram<bit_hist_op<BIG_HISTO_BIT_SIZE>, N_UNROLL>, 
        cudaFuncAttributePreferredSharedMemoryCarveout, max_shm_precentage)
    );

    CUDA_RT_CALL( cudaFuncSetAttribute(
        histogram_pass_one<key_type, size_type, value_type, N_UNROLL>, 
        cudaFuncAttributePreferredSharedMemoryCarveout, max_shm_precentage)
    );

    if (with_value) {
        CUDA_RT_CALL( cudaFuncSetAttribute(
            partition_pass_one<key_type, size_type, value_type, N_UNROLL, true>, 
            cudaFuncAttributePreferredSharedMemoryCarveout, max_shm_precentage)
        );
    } else {
        CUDA_RT_CALL( cudaFuncSetAttribute(
            partition_pass_one<key_type, size_type, value_type, N_UNROLL, false>, 
            cudaFuncAttributePreferredSharedMemoryCarveout, max_shm_precentage)
        );
    }

    CUDA_RT_CALL( cudaFuncSetAttribute(
        histogram_pass_two<key_type, size_type, value_type, N_UNROLL>, 
        cudaFuncAttributePreferredSharedMemoryCarveout, max_shm_precentage)
    );

    if (with_value) {
        CUDA_RT_CALL( cudaFuncSetAttribute(
            partition_pass_two<key_type, size_type, value_type, N_UNROLL, true>, 
            cudaFuncAttributePreferredSharedMemoryCarveout, max_shm_precentage)
        );
    } else {
        CUDA_RT_CALL( cudaFuncSetAttribute(
            partition_pass_two<key_type, size_type, value_type, N_UNROLL, false>, 
            cudaFuncAttributePreferredSharedMemoryCarveout, max_shm_precentage)
        );
    }

    if (uniq_build_tbl_keys) {
        if (with_value) {
            CUDA_RT_CALL( cudaFuncSetAttribute(
                hash_join<key_type, size_type, value_type, joined_type, hasher, true, true>, 
                cudaFuncAttributePreferredSharedMemoryCarveout, 100)
            );
        } else { 
            CUDA_RT_CALL( cudaFuncSetAttribute(
                hash_join<key_type, size_type, value_type, joined_type, hasher, true, false>, 
                cudaFuncAttributePreferredSharedMemoryCarveout, 100)
            );
        }
    } else {
        if (with_value) {
            CUDA_RT_CALL( cudaFuncSetAttribute(
                hash_join<key_type, size_type, value_type, joined_type, hasher, false, true>, 
                cudaFuncAttributePreferredSharedMemoryCarveout, 100)
            );
        } else { 
            CUDA_RT_CALL( cudaFuncSetAttribute(
                hash_join<key_type, size_type, value_type, joined_type, hasher, false, false>, 
                cudaFuncAttributePreferredSharedMemoryCarveout, 100)
            );
        }    
    }

    const int dynamic_max_shm = 96000; // set max dynamic shared memory size to 96KB for hash join kernel 

    CUDA_RT_CALL( cudaFuncSetAttribute(
        big_histogram<bit_hist_op<BIG_HISTO_BIT_SIZE>, N_UNROLL>, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_max_shm)
    );

    if (uniq_build_tbl_keys) {
        if (with_value) {
            CUDA_RT_CALL( cudaFuncSetAttribute(
                hash_join<key_type, size_type, value_type, joined_type, hasher, true, true>, 
                cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_max_shm)
            );
        } else { 
            CUDA_RT_CALL( cudaFuncSetAttribute(
                hash_join<key_type, size_type, value_type, joined_type, hasher, true, false>, 
                cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_max_shm)
            );
        }               
    } else {
        if (with_value) {
            CUDA_RT_CALL( cudaFuncSetAttribute(
                hash_join<key_type, size_type, value_type, joined_type, hasher, false, true>, 
                cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_max_shm)
            );
        } else { 
            CUDA_RT_CALL( cudaFuncSetAttribute(
                hash_join<key_type, size_type, value_type, joined_type, hasher, false, false>, 
                cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic_max_shm)
            );
        }    
    }

}

#endif //PARTITIONED_JOIN_CUH
