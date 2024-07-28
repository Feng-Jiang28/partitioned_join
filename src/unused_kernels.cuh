#include "partitioned_join.cuh"

template<typename key_type, typename size_type, typename value_type, int n_unroll>
__global__ void compute_histogram_pass_two(size_type* histo, 
                                           key_type* table, 
                                           const size_type table_size, 
                                           const int num_partitions_p1, 
                                           const int num_partitions_p2, 
                                           size_type* global_histo_p1, 
                                           size_type* global_histo_p2, 
                                           int_fastdiv blk_per_partition,
                                           const int num_bit_p1)
{
    __shared__ int partition_start_shared;
    __shared__ int partition_end_shared;

    if (threadIdx.x == 0) {
        partition_start_shared = global_histo_p1[blockIdx.x / blk_per_partition]; 
        partition_end_shared = (blockIdx.x / blk_per_partition) == num_partitions_p1 - 1
                                ? table_size : global_histo_p1[blockIdx.x / blk_per_partition + 1];
    }

    __syncthreads();

    int partition_start = partition_start_shared;
    int partition_end = partition_end_shared;

    extern __shared__ size_type shared_histo[];
    int num_block = gridDim.x;

    // initialize shared histogram to 0 
    for (int i = threadIdx.x; i < num_partitions_p2; i += blockDim.x) {
        shared_histo[i] = 0; 
    }

    __syncthreads();

    for (int i = partition_start + threadIdx.x + (blockIdx.x % blk_per_partition) * blockDim.x * n_unroll; i < partition_end; 
        i += blk_per_partition * blockDim.x * n_unroll) {

        key_type ttable[n_unroll]; 

        #pragma unroll 
        for (int u = 0; u < n_unroll; ++u) {
            int j = i + u * blockDim.x;
            ttable[u] = j < partition_end ? table[j] : ~0;
        }

        int prev_h = (((num_partitions_p2 - 1) << num_bit_p1) & ttable[0]) >> num_bit_p1; 
        int local_agg = 0; 

        #pragma unroll 
        for (int u = 0; u < n_unroll; ++u) {
            int j = i + u * blockDim.x;

            if (j >= partition_end) break;

            int h = (((num_partitions_p2 - 1) << num_bit_p1) & ttable[u]) >> num_bit_p1; 
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
        size_type curr_partition_size = shared_histo[i];
        histo[blockIdx.x + i * num_block] = curr_partition_size;
    }
}

template<typename size_type> 
__global__ void compute_global_histogram(size_type* global_histo_p2, 
                                         size_type* histo_scanned, 
                                         const int num_partitions_p2,
                                         const int blk_per_partition) 
{
    int part_1 = blockIdx.x;
    int part_2 = threadIdx.x; 
    global_histo_p2[part_1 * num_partitions_p2 + part_2] = 
        histo_scanned[part_1 * num_partitions_p2 * blk_per_partition + part_2 * blk_per_partition];
}

template<typename key_type, typename size_type, typename value_type, int n_unroll, bool with_value>
__global__ void construct_partitioned_table_pass_two(key_type* table_out,
                                                     key_type* table,
                                                     value_type* value_out,
                                                     value_type* value,
                                                     const size_type table_size, 
                                                     size_type* histo, 
                                                     size_type* histo_scanned, 
                                                     size_type* global_histo_p1, 
                                                     size_type* global_histo_p2,
                                                     const int num_partitions_p1, 
                                                     const int num_partitions_p2,
                                                     int_fastdiv blk_per_partition, 
                                                     const int num_bit_p1, 
                                                     const int max_shared_mem, 
                                                     int_fastdiv nth_partition)
{
    extern __shared__ char shared_mem[];
    key_type *table_shared = (key_type*)shared_mem; 
    // use larger shared memory for write-back buffer for pass two. 
    size_type *histo_shared = (size_type*)(table_shared + max_shared_mem); 
    value_type *value_shared = nullptr;

    if (with_value) value_shared = (value_type*)(histo_shared + num_partitions_p2);

    __shared__ int partition_start_shared;
    __shared__ int partition_end_shared; 

    int num_block = gridDim.x;

    typedef cub::BlockScan<size_type, BLOCK_SIZE_PARTITION> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    // each thread handles ELEM_PER_THREAD elements to support upto 1024 partitions 
    size_type temp_histo[ELEM_PER_THREAD];

    for (int i = 0; i < ELEM_PER_THREAD; ++i) {
        if (ELEM_PER_THREAD * threadIdx.x + i < num_partitions_p2) {
            temp_histo[i] = histo[blockIdx.x + (ELEM_PER_THREAD * threadIdx.x + i) * gridDim.x]; 
        } else {
            temp_histo[i] = 0;
        }
    }

    __syncthreads();

    BlockScan(temp_storage).ExclusiveSum(temp_histo, temp_histo);

    __syncthreads();

    for (int i = 0; i < ELEM_PER_THREAD; ++i) {
        if (ELEM_PER_THREAD * threadIdx.x + i < num_partitions_p2) {
            histo_shared[ELEM_PER_THREAD * threadIdx.x + i] = temp_histo[i]; 
        } 
    }
    
    if (threadIdx.x == 0) {
        partition_start_shared = global_histo_p1[blockIdx.x / blk_per_partition]; 
        partition_end_shared = (blockIdx.x / blk_per_partition) == num_partitions_p1 - 1
                                ? table_size : global_histo_p1[blockIdx.x / blk_per_partition + 1];
    }

    __syncthreads();

    int partition_start = partition_start_shared;
    int partition_end = partition_end_shared;

    for (int i = partition_start + threadIdx.x + (blockIdx.x % blk_per_partition) * blockDim.x * n_unroll; i < partition_end; 
        i += blk_per_partition * blockDim.x * n_unroll) {

        key_type ttable[n_unroll]; 
        value_type tvalue[n_unroll]; 

        #pragma unroll 
        for (int u = 0; u < n_unroll; ++u) {
            int j = i + u * blockDim.x;
            ttable[u] = j < partition_end ? table[j] : ~0;
            if (with_value) {
                tvalue[u] = j < partition_end ? value[j] : ~0;
            }
        }

        #pragma unroll 
        for (int u = 0; u < n_unroll; ++u) {
            int j = i + u * blockDim.x;

            if (j >= partition_end) break;

            int h = (((num_partitions_p2 - 1) << num_bit_p1) & ttable[u]) >> num_bit_p1;

            int pos = atomicAdd(&histo_shared[h], 1);
            if (pos < max_shared_mem) {
                table_shared[pos] = ttable[u];
                if (with_value) {
                    value_shared[pos] = tvalue[u];
                }
            } else {
                int pos_g = atomicAdd(&histo_scanned[blockIdx.x + h * num_block], 1);
                table_out[pos_g] = ttable[u];
                if (with_value) {
                    value_out[pos_g] = tvalue[u];
                }
            }
        }
    }

    __syncthreads(); 

    // flush shared memory to global memory
    // https://github.com/rapidsai/cudf/blob/2206ae55a23ef4b3d497387402bbb7ff8b08ce6a/cpp/src/hash/hashing.cu#L360
    for (int i = threadIdx.x / nth_partition; i < num_partitions_p2; i += BLOCK_SIZE_PARTITION / nth_partition) {
        int row_off = threadIdx.x % nth_partition;
        int shared_pos_end = histo_shared[i] >= max_shared_mem ? max_shared_mem : histo_shared[i];
        int shared_pos_start = i == 0 ? 0 : histo_shared[i-1];
        if (shared_pos_start >= max_shared_mem) {
            break; 
        }
        int nelem = shared_pos_end - shared_pos_start; 
        int global_pos_start = histo_scanned[blockIdx.x + i * num_block];

        while (row_off < nelem) {
            table_out[global_pos_start + row_off] = table_shared[shared_pos_start + row_off];
            if (with_value) {
                value_out[global_pos_start + row_off] = value_shared[shared_pos_start + row_off];
            }
            row_off += nth_partition;
        }
    }
}

template<typename key_type, typename size_type, typename value_type>
void partition_input_tables_pass_two(key_type* partitioned_table, 
                                     value_type* partitioned_value, 
                                     size_type* histo, 
                                     size_type* histo_scanned, 
                                     const size_type histo_size, 
                                     key_type* input_table, 
                                     value_type* input_value,
                                     const size_type table_size, 
                                     const int num_partitions_1, 
                                     const int num_partitions_2, 
                                     void* scan_temp_storage, 
                                     size_t temp_storage_bytes, 
                                     size_type* global_histo_p1, 
                                     size_type* global_histo_p2, 
                                     const bool with_value, 
                                     const int num_bits_p1)
{
    cudaEvent_t start, stop;
    float histogram_elapsed_ms = 0, global_elapsed_ms = 0, partition_elapsed_ms = 0;

    CUDA_RT_CALL( cudaEventCreate(&start) );
    CUDA_RT_CALL( cudaEventCreate(&stop) );

    CUDA_RT_CALL( cudaEventRecord(start, 0) );

    int avg_elem_in_blk = ceil(1.0 * table_size / num_partitions_1);
    int num_blk_per_part = ceil(1.0 * avg_elem_in_blk / (BLOCK_SIZE_PARTITION * N_UNROLL));
    int num_block = num_partitions_1 * num_blk_per_part;

    compute_histogram_pass_two<key_type, size_type, value_type, N_UNROLL>
    <<<num_block, BLOCK_SIZE_PARTITION, num_partitions_2*sizeof(size_type)>>>
    (
        histo, input_table, table_size, num_partitions_1, 
        num_partitions_2, global_histo_p1, global_histo_p2, num_blk_per_part, num_bits_p1
    );

    CUDA_RT_CALL( cudaEventRecord(stop, 0) );
    CUDA_RT_CALL( cudaEventSynchronize(stop) );
    CUDA_RT_CALL( cudaEventElapsedTime(&histogram_elapsed_ms, start, stop) );

    cub::DeviceScan::ExclusiveSum(scan_temp_storage, temp_storage_bytes, histo, histo_scanned, histo_size);

    CUDA_RT_CALL( cudaEventRecord(start, 0) );
    compute_global_histogram<size_type><<<num_partitions_1, num_partitions_2>>>(
        global_histo_p2, histo_scanned, num_partitions_2, num_blk_per_part
    );
    CUDA_RT_CALL( cudaEventRecord(stop, 0) );
    CUDA_RT_CALL( cudaEventSynchronize(stop) );
    CUDA_RT_CALL( cudaEventElapsedTime(&global_elapsed_ms, start, stop) );

    CUDA_RT_CALL( cudaEventRecord(start, 0) );

    int max_shared_mem = ceil(PASS_2_SHRMEM_MULTIPLIER * BLOCK_SIZE_PARTITION * N_UNROLL); 

    if (with_value) {
        construct_partitioned_table_pass_two<key_type, size_type, value_type, N_UNROLL, true>
        <<<num_block, BLOCK_SIZE_PARTITION, max_shared_mem*sizeof(key_type) + max_shared_mem*sizeof(value_type) +
        num_partitions_2*sizeof(size_type)>>>
        (
            partitioned_table, input_table, partitioned_value, input_value, table_size, histo, histo_scanned, 
            global_histo_p1, global_histo_p2, num_partitions_1, num_partitions_2, num_blk_per_part, 
            num_bits_p1, max_shared_mem, nthread_writeback_partition
        );
    } else {
        construct_partitioned_table_pass_two<key_type, size_type, value_type, N_UNROLL, false>
        <<<num_block, BLOCK_SIZE_PARTITION, max_shared_mem*sizeof(key_type) + num_partitions_2*sizeof(size_type)>>>
        (
            partitioned_table, input_table, nullptr, nullptr, table_size, histo, histo_scanned, 
            global_histo_p1, global_histo_p2, num_partitions_1, num_partitions_2, num_blk_per_part, 
            num_bits_p1, max_shared_mem, nthread_writeback_partition
        );
    }
    CUDA_RT_CALL( cudaEventRecord(stop, 0) );
    CUDA_RT_CALL( cudaEventSynchronize(stop) );
    CUDA_RT_CALL( cudaEventElapsedTime(&partition_elapsed_ms, start, stop) );

    std::cout << histogram_elapsed_ms << "," << global_elapsed_ms << "," << partition_elapsed_ms << ","; 
}
