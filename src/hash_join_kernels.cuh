/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*/

#ifndef __HASH_JOIN_KERNELS_CUH
#define __HASH_JOIN_KERNELS_CUH

#include <iostream>
#include <algorithm>
#include "hash_functions.cuh"
#include "common.cuh"
#include "../int_fastdiv/int_fastdiv.h"

constexpr size_type BLOCK_SIZE_JOIN = 512;
constexpr size_type UNUSED_VAL = -1;
constexpr float HASH_TABLE_OCC = 0.5;

template<typename key_type, typename size_type, typename value_type, typename joined_type, typename hasher, bool uniq_key, bool with_value>
__global__ void hash_join(
    joined_type* __restrict__ joined_table,
    const key_type* __restrict__ build_table, 
    const value_type* __restrict__ build_value,
    const size_type build_table_size, 
    const key_type* __restrict__ probe_table, 
    const value_type* __restrict__ probe_value,
    const size_type probe_table_size, 
    const int num_partitions, 
    const size_type base_build_tbl_size_each_part, 
    size_type* __restrict__ global_index, 
    const size_type* __restrict__ num_blocks_each_partition_s, 
    const size_type* __restrict__ partition_index, 
    const hasher hf,
    int_fastdiv hash_table_size,
    const size_type* __restrict__ build_global_histo, 
    const size_type* __restrict__ probe_global_histo, 
    const size_type write_back_buffer_capacity) 
{
    extern __shared__ char shared_mem[];
    hash_map_type *hash_table = (hash_map_type*)shared_mem;
    joined_type *write_back_buffer = (joined_type*)(hash_table + hash_table_size);
    __shared__ int write_back_buffer_size;
    __shared__ int gloabl_write_start_index; 

    __shared__ int build_partition_size_start_shared;
    __shared__ int probe_partition_size_start_shared;
    __shared__ int build_partition_size_end_shared;
    __shared__ int probe_partition_size_end_shared;

    int block_idx = blockIdx.x; 
    
    if (threadIdx.x == 0) {
        write_back_buffer_size = 0;
        int partition_idx = partition_index[block_idx];
        build_partition_size_start_shared = build_global_histo[partition_idx]; 
        probe_partition_size_start_shared = probe_global_histo[partition_idx]; 
        build_partition_size_end_shared = partition_idx == num_partitions - 1 ? build_table_size : build_global_histo[partition_idx+1];
        probe_partition_size_end_shared = partition_idx == num_partitions - 1 ? probe_table_size : probe_global_histo[partition_idx+1];

        int num_partition_prev_blk = partition_idx == 0 ? 0 : num_blocks_each_partition_s[partition_idx-1]; 
        int partition_offset = block_idx - num_partition_prev_blk;
        build_partition_size_start_shared += partition_offset * base_build_tbl_size_each_part; 
        int build_partition_size_end_shared_temp = build_partition_size_start_shared + base_build_tbl_size_each_part; 
        if (build_partition_size_end_shared_temp < build_partition_size_end_shared) {
            build_partition_size_end_shared = build_partition_size_end_shared_temp; 
        }
    }

    __syncthreads();

    int build_partition_size_start = build_partition_size_start_shared;
    int probe_partition_size_start = probe_partition_size_start_shared;
    int build_partition_size = build_partition_size_end_shared - build_partition_size_start;
    int probe_partition_size = probe_partition_size_end_shared - probe_partition_size_start;

    if (build_partition_size == 0 || probe_partition_size == 0) {
        return; 
    }

    for (int i = threadIdx.x; i < hash_table_size; i += blockDim.x) {
        // define unused elements = UNUSED_VAL
        hash_table[i].x = UNUSED_VAL;
        hash_table[i].y = 0;
    }

    __syncthreads();
    
    // build hash table 
    for (int i = threadIdx.x; i < build_partition_size; i += blockDim.x) {
        key_type key = build_table[build_partition_size_start + i];
        value_type value = 0; 
        if (with_value) {
            value = build_value[build_partition_size_start + i]; 
        }

        int hash_tbl_idx_temp = hf(key) % hash_table_size;
        int hash_tbl_idx = hash_tbl_idx_temp < 0 ? -hash_tbl_idx_temp : hash_tbl_idx_temp; 

        int attempt_counter = 0;

        while (attempt_counter < hash_table_size) {
            const int old = atomicCAS(&hash_table[hash_tbl_idx].x, UNUSED_VAL, key);

            if (old == UNUSED_VAL) {
                // inserion success, insert payload as well 
                if (with_value) {
                    hash_table[hash_tbl_idx].y = value;
                }
                break; 
            } else {
                hash_tbl_idx = (hash_tbl_idx + 1);
                if (hash_tbl_idx == hash_table_size) {
                    hash_tbl_idx = 0; // start over from begining 
                }

                attempt_counter++;

                if (attempt_counter == hash_table_size) {
                    // comment out the printf to avoid long kernel launch issue 
                    // this printf is useful for debugging though 
                    // printf("inserision fails\n");
                }
            }
        }
    }

    __syncthreads();

    // probe hash table
    for (int i = threadIdx.x; i < probe_partition_size; i += blockDim.x) {
        key_type key = probe_table[probe_partition_size_start + i];
        value_type value = 0; 
        if (with_value) {
            value = probe_value[probe_partition_size_start + i]; 
        }

        int hash_tbl_idx_temp = hf(key) % hash_table_size;
        int hash_tbl_idx = hash_tbl_idx_temp < 0 ? -hash_tbl_idx_temp : hash_tbl_idx_temp; 
        int hash_tbl_idx_save = hash_tbl_idx;

        while (true) {
            // return with no hit if find a unused value 
            if (hash_table[hash_tbl_idx].x == UNUSED_VAL) {
                break;
            }
            
            if (hash_table[hash_tbl_idx].x == key) {
                int shared_offset = atomicAdd(&write_back_buffer_size, 1);

                if (shared_offset < write_back_buffer_capacity) {
                    write_back_buffer[shared_offset].x = key;
                    if (with_value) {
                        write_back_buffer[shared_offset].y = value;
                    }
                } else {
                    int output_offset = atomicAdd(&global_index[0], 1);
                    joined_table[output_offset].x = key;
                    if (with_value) {
                        joined_table[output_offset].y = value;
                    }
                }

                if (uniq_key) {
                    break; 
                }
            }

            hash_tbl_idx = (hash_tbl_idx + 1);
            if (hash_tbl_idx == hash_table_size) {
                hash_tbl_idx = 0; // start over from begining 
            }

            if (hash_tbl_idx_save == hash_table_size) {
                break; 
            }
        }
    }

    __syncthreads(); 

    // write back to gloabl memory 
    if (threadIdx.x == 0) {
        if (write_back_buffer_size > write_back_buffer_capacity) {
            write_back_buffer_size = write_back_buffer_capacity;
        }
        gloabl_write_start_index = atomicAdd(&global_index[0], write_back_buffer_size);
    }

    __syncthreads(); 

    for (int i = threadIdx.x; i < write_back_buffer_size; i += blockDim.x) {
        joined_table[gloabl_write_start_index + i].x = write_back_buffer[i].x;
        if (with_value) {
            joined_table[gloabl_write_start_index + i].y = write_back_buffer[i].y;
        }
    }
}

#endif //HASH_JOIN_KERNELS_CUH