/* Copyright 2019 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*/

#include <iostream>
#include <iomanip>
#include <string>
#include <curand.h>
#include <getopt.h>
#include <cuda_profiler_api.h>

#include "src/partitioned_join.cuh"
#include "src/generate_input_tables.cuh"
#include "src/inner_join_cpu.hpp"
#include "src/common.cuh"
#include "src/skewed_data_handler.cuh"
#include "src/generate_zipf_input.cuh"
#include "src/nvtx_helper.cuh"

constexpr size_t GIGABYTE = 1000000000;

bool is_power_of_two(int x)
{
    return (x != 0) && ((x & (x - 1)) == 0);
}

/*
* 1. Initialization and Argument Parsing.
* The code starts by setting default values for various parameters, such as table sizes,
* selectivity, random seed, and flags for unique keys, verbosity, and partitioning methods.
* It parses comman-line arguments to override the default settings.
*
* 2. Setting Up Memory and Table.
* It allocates GPU memory for the build and probe tables, including keys and optional values
* If reading from files, it loads the data into the allocated memory.
* If generationg data, it creates the tables either uniformly or with skew, depending on the
* specified mode.
*
* 3. Partitioning and Histogram Calculation
* The code determines whether to use radix partitioning and calculates the number of partitions and
* bit required.
*
* 4. Copying Data to Host Memory for Debugging.
* The code copies data from GPU to host memory for debugging purposes if verbosity is high.
*
* 5. Partitioning Logic.
* It sets up structures for partitioning the tables. This involves setting up histogram arrays
* and calculating the memory required for partitioning.
* If values are present, it handles them similarly by allocating and initializing additioal memory.
*
* 6. Radix Partitioning(if enabled).
* The radix partitioning logic involves creating partitions based on hash values of keys, which can be
* further divided based on the number of bits specified.
*
* 7. Main Join Logic.
* Depending on the partitioning, the code performs the hash join operation by probing the build table using
* the probe table.
* It uses CUDA kernel lauches to perform the join operatin efficiently on the GPU.
*
* 8. Performance Measurement：
* The code measutes the execution time for various stages of the join process, including partitioning, building histogram
* and probing.
* It prints the performance metrics and settings used for the run.
*
* 9. Cleanup.
* Finally, it frees the allocated GPU and host memory to avoid memory leaks.
*
* */

int main (int argc, char** argv) {
    // default arguments
    size_type build_tbl_size = 1<<20; // Size of the build table (2^20 = 1,048,576)
    size_type probe_tbl_size = 1<<20;  // Size of the probe table (2^20 = 1,048,576)
    double selectivity = 0.03f;   // Selectivity for the join operation
    int rand_max = build_tbl_size<<1; // Maximum value for generated data (twice the build table size)
    bool uniq_build_tbl_keys = false;  // Flag for unique keys in the build table
    bool read_from_file = false;  // Flag to read data from file
    std::string build_tbl_file = "";  // Path to the build table file
    std::string probe_tbl_file = "";   // Path to the probe table file
    int verbosity = 0; // Verbosity level for output
    bool radix_partition = false;  // Flag for using radix partitioning
    int num_partitions_1 = -1; // Number of partitions in the first pass
    int num_partitions_2 = -1; // Number of partitions in the second pass
    long long device_memory_used = 0; // Amount of device memory used
    bool skip_cpu_run = false; // Flag to skip CPU reference run
    int output_size = -1;  // Output size if CPU run is skipped
    bool with_value = false;  // Flag to add value columns to input tables
    bool csv = false;  // Flag to output results in CSV format
    int skewed_data_mode = 0;  // Mode for skewed data generation
    float zipf_factor = -1.0;  // Zipf factor for skewed data
    float max_partition_size_factor = 3.0;  // Factor for max partition size
    bool do_combined_histo_pass = false; // Flag for combined histogram pass

    // command line options
    const option long_opts[] = {
        {"build_tbl_size", required_argument, nullptr, 'b'},
        {"probe_tbl_size", required_argument, nullptr, 'p'},
        {"selectivity", required_argument, nullptr, 's'},
        {"rand_max", required_argument, nullptr, 'r'},
        {"uniq_build_tbl_keys", no_argument, nullptr, 'u'},
        {"read_from_file", no_argument, nullptr, 'F'},
        {"build_tbl_file", required_argument, nullptr, 'B'},
        {"probe_tbl_file", required_argument, nullptr, 'P'},
        {"verbosity", required_argument, nullptr, 'v'},
        {"max_partition_size_factor", required_argument, nullptr, 'f'},
        {"radix_partition", no_argument, nullptr, 'R'},
        {"num_partitions_1", required_argument, nullptr, 'n'},
        {"num_partitions_2", required_argument, nullptr, 'N'},
        {"skip_cpu_run", required_argument, nullptr, 'S'},
        {"with_value", no_argument, nullptr, 'V'},
        {"csv", no_argument, nullptr, 'C'},
        {"skewed_data_mode", required_argument, nullptr, 'Z'},
        {"zipf_dist", required_argument, nullptr, 'z'},
        {"do_combined_histogram_pass", required_argument, nullptr, 'H'}, 
    };

    const std::string opts_desc[] = {
        "Build table size",
        "Probe table size",
        "Selectivity (ignored when -f is enabled)", 
        "Maximum value of generated data (ignored when -f is enabled)", 
        "Unique keys for build table", 
        "Read from file (need to set -b, -p, -u, -B, and -P according to the files)", 
        "Build table file path (ignored when -f is not enabled)", 
        "Probe table file path (ignored when -f is not enabled)", 
        "Verbosity level", 
        "Factor controls the max partition size to further divide the large subsets to smaller ones \n \
        size = <this factor> * ceil(build table size / number of partitions)",
        "Use two-pass radix partition algorithm",
        "Number of partitions for pass 1",
        "Number of partitions for pass 2",
        "Skip CPU reference run and sepecify the output size", 
        "Add value columns to input tables", 
        "Output results in csv format",
        "Skewed data on [0: none, 1: build, 2: probe, 3: both]",
        "Zipf factor for skewed data [0-1]", 
        "Use combined histogram pass. When number of partition is small, this option may speedup the application. Only implemented with 4B key"
    };

    const std::string opts_default[] = {
        std::to_string(build_tbl_size),
        std::to_string(probe_tbl_size),
        std::to_string(selectivity),
        std::to_string(rand_max),
        std::to_string(uniq_build_tbl_keys),
        std::to_string(read_from_file),
        build_tbl_file,
        probe_tbl_file,
        std::to_string(verbosity),
        std::to_string(max_partition_size_factor),
        std::to_string(radix_partition),
        std::to_string(num_partitions_1),
        std::to_string(num_partitions_2),
        std::to_string(output_size),
        std::to_string(with_value),
        std::to_string(csv),
        std::to_string(skewed_data_mode), 
        std::to_string(zipf_factor),
        std::to_string(do_combined_histo_pass),
    };

    // parse command line
    int opt;
    while ((opt = getopt_long(argc, argv, "b:p:s:r:uFB:P:v:f:Rn:N:S:VCZ:z:Hh", long_opts, nullptr)) != -1) {
        switch (opt) {
            case 'b': build_tbl_size = atoi(optarg); break;
            case 'p': probe_tbl_size = atoi(optarg); break;
            case 's': selectivity = atof(optarg); break;
            case 'r': rand_max = atoi(optarg); break;
            case 'u': uniq_build_tbl_keys = true; break;
            case 'F': read_from_file = true; break;
            case 'B': build_tbl_file = optarg; break;
            case 'P': probe_tbl_file = optarg; break;
            case 'v': verbosity = atoi(optarg); break;
            case 'f': max_partition_size_factor = atof(optarg); break;
            case 'R': radix_partition = true; break;
            case 'n': num_partitions_1 = atoi(optarg); break;
            case 'N': num_partitions_2 = atoi(optarg); break;
            case 'S': skip_cpu_run = true; output_size = atoi(optarg); break;
            case 'V': with_value = true; break;
            case 'C': csv = true; break;
            case 'Z': skewed_data_mode = atoi(optarg); break; 
            case 'z': zipf_factor = atof(optarg); break; 
            case 'H': do_combined_histo_pass = true; break; 
            case 'h': {
                std::cout << "Usage:" << std::endl;
                int num_opts = std::extent<decltype(opts_desc)>::value;
                for (int i = 0; i < num_opts; i++)
                if (long_opts[i].has_arg != no_argument)
                    std::cout << "  -" << (char)long_opts[i].val << ", --" << long_opts[i].name << " [arg]" << std::endl
                    << "    " << opts_desc[i] << " [default: " << opts_default[i] << "]" << std::endl;
                else
                    std::cout << "  -" << (char)long_opts[i].val << ", --" << long_opts[i].name << std::endl
                    << "    " << opts_desc[i] << " [default: " << std::boolalpha << opts_default[i] << "]" << std::endl;

                exit(EXIT_FAILURE);
            }
            case '?':
                std::cout << "Please use -h or --help for the list of options" << std::endl;
                exit(EXIT_FAILURE);
            default:
                break;
        }
    }

#ifdef USE_8B_KEYS 
    if (do_combined_histo_pass) {
        std::cerr << "Error: combined histo pass is not supported for 8B key. Disable -H." << std::endl;
        return 1; 
    }
#endif 


    int num_bits_1 = 0; // Number of bits for the first partition
    int num_bits_2 = 0; // Number of bits for the second partition

    // Understanding the Bits for Partitioning
    /*
    * Binary Representation and Power of Two:
    * A number of partitions that is a power of two can be easily represented using binary digits (bits).
    * if num_partitions_1  is 4，then we need 2 bits to represent the 4 partitions 00 01 10 11
    * std::log2(num_partitions_1)  function calculates the base-2 logarithm, which effectively gives us the
    * number of bits required to represent the number of partitions.
    * */

    /*
    * Radix Partitioning:
    * Radix partitioning is a method of partitioning data in multiple passes. It can be used to further
    * divide data into finer partitions.
    *
    * */

    // Determine the number of bits for partitioning based on whether radix partitioning is used
    if (!radix_partition && is_power_of_two(num_partitions_1)) {
        // If not using radix partitioning and num_partitions_1 is a power of two, calculate num_bits_1
        num_bits_1 = std::log2(num_partitions_1);
    }
    else if (radix_partition && is_power_of_two(num_partitions_1) && is_power_of_two(num_partitions_2)) {
        // If using radix partitioning and both num_partitions_1 and num_partitions_2 are powers of two, calculate num_bits_1 and num_bits_2
        num_bits_1 = std::log2(num_partitions_1);
        num_bits_2 = std::log2(num_partitions_2);
    } else {
        // If the conditions are not met, print an error message and exit
        std::cerr << "Error: num_partition must be power of two." << std::endl;
        return 1; 
    }

    // Calculate the total number of partitions
    int num_partitions = radix_partition ? num_partitions_1 * num_partitions_2 : num_partitions_1;

    // Determine the maximum number of partitions between the two
    int num_partitions_max = std::max(num_partitions_1, num_partitions_2);

    // input tables
    // Allocate memory for the build table on the GPU
    key_type* build_tbl = nullptr;
    key_type* probe_tbl = nullptr;
    // build_tbl_size is the number of elements in the build table
    // sizeof(key_type):  gives the size of each element in the table.
    // used to store the keys for the build table.
    CUDA_RT_CALL( cudaMalloc(&build_tbl, build_tbl_size * sizeof(key_type)) );
    device_memory_used += build_tbl_size * sizeof(key_type);


    // Allocate memory for the probe table on the GPU
    CUDA_RT_CALL( cudaMalloc(&probe_tbl, probe_tbl_size * sizeof(key_type)) );
    device_memory_used += probe_tbl_size * sizeof(key_type);

    value_type* build_tbl_val = nullptr; // Pointer for build table values
    value_type* probe_tbl_val = nullptr; // Pointer for probe table values
    value_type* build_tbl_val_p = nullptr; // Pointer for build table values for partitioning
    value_type* probe_tbl_val_p = nullptr; // Pointer for probe table values for partitioning

    // Check if values are to be included
    if (with_value) {
        // Allocate memory for build table values on the GPU
        CUDA_RT_CALL( cudaMalloc(&build_tbl_val, build_tbl_size * sizeof(value_type)) );
        device_memory_used += build_tbl_size * sizeof(value_type); 

        // Initialize build table values to zero
        CUDA_RT_CALL( cudaMemset(build_tbl_val, 0, build_tbl_size * sizeof(value_type)) );

        // Allocate memory for probe table values on the GPU
        CUDA_RT_CALL( cudaMalloc(&probe_tbl_val, probe_tbl_size * sizeof(value_type)) );
        device_memory_used += probe_tbl_size * sizeof(value_type);

        // Initialize probe table values to zero
        CUDA_RT_CALL( cudaMemset(probe_tbl_val, 0, probe_tbl_size * sizeof(value_type)) );

        // Allocate memory for build table values for partitioning on the GPU
        CUDA_RT_CALL( cudaMalloc((void**)&build_tbl_val_p, build_tbl_size * sizeof(value_type)) );
        device_memory_used += build_tbl_size * sizeof(value_type);

        // Allocate memory for probe table values for partitioning on the GPU
        CUDA_RT_CALL( cudaMalloc((void**)&probe_tbl_val_p, probe_tbl_size * sizeof(value_type)) );
        device_memory_used += probe_tbl_size * sizeof(value_type);    
    }

    if (read_from_file) {
        // Declare pointers for reading build and probe tables from file
        key_type* build_tbl_read = nullptr;
        key_type* probe_tbl_read = nullptr;

        // Allocate pinned (page-locked) host memory for build and probe tables
        CUDA_RT_CALL( cudaMallocHost(&build_tbl_read, build_tbl_size * sizeof(key_type)) );
        CUDA_RT_CALL( cudaMallocHost(&probe_tbl_read, probe_tbl_size * sizeof(key_type)) );

        // Check if CSV format is not used, print a message indicating tables are being read
        if (!csv) {
            std::cout << "[*] Reading input tables" << std::endl;
        }

        // readFromFile(build_tbl_file.c_str(), build_tbl_read, build_tbl_size);
        // readFromFile(probe_tbl_file.c_str(), probe_tbl_read, probe_tbl_size);
        // Allocate pinned host memory for build table payload
        int* build_payload_read = nullptr; 
        CUDA_RT_CALL( cudaMallocHost(&build_payload_read, build_tbl_size * sizeof(int)) );

        // script for loading TPC-H Q4 data 
        std::string lfile = "/scratch/local/rlan/sf10/lineitem.tbl"; 
        std::string ofile = "/scratch/local/rlan/sf10/orders.tbl"; 
        load_tpch_q4_data(lfile, ofile, probe_tbl_size, build_tbl_size, probe_tbl_read, build_tbl_read, build_payload_read); 

        // Copy build and probe tables from host to device memory
        CUDA_RT_CALL( cudaMemcpy(build_tbl, build_tbl_read, build_tbl_size * sizeof(key_type), cudaMemcpyDefault) );
        CUDA_RT_CALL( cudaMemcpy(probe_tbl, probe_tbl_read, probe_tbl_size * sizeof(key_type), cudaMemcpyDefault) );

        // Copy build table payload from host to device memory
        CUDA_RT_CALL( cudaMemcpy(build_tbl_val, build_payload_read, build_tbl_size * sizeof(int), cudaMemcpyDefault) );

        // Free the pinned host memory
        CUDA_RT_CALL( cudaFreeHost(build_tbl_read) ); 
        CUDA_RT_CALL( cudaFreeHost(probe_tbl_read) ); 
    } else {
        // If skewwd data mode is not zero
        if (skewed_data_mode != 0) {
            // enforce selectivity to 1 
            selectivity = 1;

            // Set unique build table keys based on skewed data mode
            uniq_build_tbl_keys = skewed_data_mode == 2 ? true : false; 
        }

        // Check if CSV format is not used, print a message indicating tables are being generated
        if (!csv) {
            std::cout << "[*] Generating input tables" << std::endl;
        }
        generate_input_tables<key_type, size_type>(build_tbl, build_tbl_size, probe_tbl, probe_tbl_size, selectivity, rand_max, uniq_build_tbl_keys);
        // linear_sequence<key_type, size_type><<<1024, 256>>>(build_tbl, build_tbl_size); 
        // linear_sequence<key_type, size_type><<<1024, 256>>>(probe_tbl, probe_tbl_size);     

        // if (skewed_data_mode == 1 || skewed_data_mode == 3) {
        //     if (!csv) {
        //         std::cout << "[*] Generating skewed build table" << std::endl;
        //     }
        //     key_type* skewed_temp = nullptr; 
        //     CUDA_RT_CALL( cudaMallocHost(&skewed_temp, build_tbl_size * sizeof(key_type)) );
        //     gen_zipf<key_type>(build_tbl_size, rand_max, zipf_factor, skewed_temp);
        //     CUDA_RT_CALL( cudaMemcpy(build_tbl, skewed_temp, build_tbl_size * sizeof(key_type), cudaMemcpyDefault) );
        //     CUDA_RT_CALL( cudaFreeHost(skewed_temp) );  
        // } 

        // if (skewed_data_mode == 2 || skewed_data_mode == 3) {
        //     if (!csv) {
        //         std::cout << "[*] Generating skewed probe table" << std::endl;
        //     }
        //     key_type* skewed_temp = nullptr; 
        //     CUDA_RT_CALL( cudaMallocHost(&skewed_temp, build_tbl_size * sizeof(key_type)) );
        //     gen_zipf<key_type>(probe_tbl_size, rand_max, zipf_factor, skewed_temp);
        //     CUDA_RT_CALL( cudaMemcpy(probe_tbl, skewed_temp, probe_tbl_size * sizeof(key_type), cudaMemcpyDefault) );
        //     CUDA_RT_CALL( cudaFreeHost(skewed_temp) );  
        // } 
    }

    float part_shared_memory_kb = 0; // Variable to store estimated partition shared memory size in KB.

    // If the output format is CSV
    if (csv) {
        int tuple_byte = with_value ? sizeof(key_type) + sizeof(value_type) : sizeof(key_type); // Calculate tuple byte size based on whether values are included.
        std::cout << build_tbl_size << "," << probe_tbl_size << "," << sizeof(key_type) << "," << tuple_byte << "," << selectivity << "," << uniq_build_tbl_keys << "," << 
            num_partitions_1 << "," << num_partitions_2 << "," << max_partition_size_factor << "," << skewed_data_mode << "," << zipf_factor << ",";
    } else {
        // Print the settings if not in CSV format.
        std::cout << "[*] Settings:" << std::endl; 
        std::cout << "    build tabel size     = " << build_tbl_size << std::endl;
        std::cout << "    probe tabel size     = " << probe_tbl_size << std::endl;
        std::cout << "    has value col        = " << std::boolalpha << with_value << std::endl;
        std::cout << "    skewed data mode     = " << skewed_data_mode << std::endl; 

        // If using skewed data, print the Zipf factor
        if (skewed_data_mode != 0) {
            std::cout << "    zipf factor          = " << zipf_factor << std::endl;     
        }

        // If reading from files, print the file paths
        if (read_from_file) {
            std::cout << "    build table path     = " << build_tbl_file << std::endl;
            std::cout << "    probe table path     = " << probe_tbl_file << std::endl;
        } else {
            // If generating tables, print selectivity and random max value
            std::cout << "    selectivity          = " << selectivity << std::endl;
            std::cout << "    rand max             = " << rand_max << std::endl;
        }
        std::cout << "    uniq build tbl keys  = " << std::boolalpha << uniq_build_tbl_keys << std::endl;
        std::cout << "    max part size factor = " << max_partition_size_factor << std::endl; 
        std::cout << "    radix partition      = " << std::boolalpha << radix_partition << std::endl;
        std::cout << "    num partitions P1    = " << num_partitions_1 << std::endl;
        std::cout << "    num bits P1          = " << num_bits_1 << std::endl;

        // If using radix partitioning, print details about the second level partitions
        if (radix_partition) {
            std::cout << "    num partitions P2    = " << num_partitions_2 << std::endl;
            std::cout << "    num bits P2          = " << num_bits_2 << std::endl;    
        }

        // Print the total number of partitions and data type sizes
        std::cout << "    num partitions total = " << num_partitions << std::endl;
        std::cout << "    sizeof(key_type)     = " << sizeof(key_type) << std::endl;
        std::cout << "    sizeof(size_type)    = " << sizeof(size_type) << std::endl;
        int partition_shared_mem_size = BLOCK_SIZE_PARTITION*N_UNROLL; // Calculate the partition shared memory size

        // Estimate the shared memory size required for partitioning
        if (with_value) {
            part_shared_memory_kb = (partition_shared_mem_size*sizeof(key_type)+
                                    partition_shared_mem_size*sizeof(value_type)+num_partitions_max*sizeof(size_type)) / 1e3;
            std::cout << "    sizeof(value_type)   = " << sizeof(value_type) << std::endl;
        } else {
            part_shared_memory_kb = (partition_shared_mem_size*sizeof(key_type)+num_partitions_max*sizeof(size_type)) / 1e3;
        }

        // Print the estimated shared memory sizes
        std::cout << "    partition shmem size = " << part_shared_memory_kb << " KB per block (estimated, lower bound)" << std::endl;
        std::cout << "    join shmem size      = " << ceil(max_partition_size_factor * build_tbl_size / num_partitions)*sizeof(int2) / (HASH_TABLE_OCC*1e3) 
                << " KB per block (estimated, lower bound)" << std::endl;
        std::cout << "    skip CPU run         = " << std::boolalpha << skip_cpu_run << std::endl;    
    }

    // Create vectors to hold the build and probe tables on the host
    std::vector<key_type> build_tbl_h(build_tbl_size);
    std::vector<key_type> probe_tbl_h(probe_tbl_size);

    // Copy the build table data from the device to the host vector
    CUDA_RT_CALL( cudaMemcpy(build_tbl_h.data(), build_tbl, build_tbl_size * sizeof(key_type), cudaMemcpyDefault) );

    // Copy the probe table data from the device to the host vector
    CUDA_RT_CALL( cudaMemcpy(probe_tbl_h.data(), probe_tbl, probe_tbl_size * sizeof(key_type), cudaMemcpyDefault) );

    // If verbosity level is greater than 2, print the contents of the build and probe tables
    if (verbosity > 2) {
        // Print the build input table
        std::cout << "====Build input table:" << std::endl;
        for (int i = 0; i < build_tbl_size; ++i) {
            std::cout << build_tbl_h[i] << " ";   // Print each element of the build table
        }
        std::cout << std::endl << "====End" << std::endl;

        // Print the probe input table
        std::cout << "====Probe input table:" << std::endl;
        for (int i = 0; i < probe_tbl_size; ++i) {
            std::cout << probe_tbl_h[i] << " ";  // Print each element of the probe table
        }
        std::cout << std::endl << "====End" << std::endl;
    }
    /*
    *  1. primary histograms (build_histo and probe_hisot)
    *  These histograms are used to perform the initial partitioning of the build and probe tables.
    *  They count the number of elements that fall into each partition based on their hash values.
    *
    *  2. Secondary Histograms
    *  These histograms serve as additional structures for more refined partitioning or optimization steps.
    *  Secondary histograms can be used to handle specific cases such as data skew, ensuring balanced partition
    *  sizes, or to optimize memory access patterns during the partitioning process.
    *
    *  3. Second-Level Histograms for Radix Partitioning
    *  In radix partitioning, which is a multi-level partitioning process, second-level histograms are used to manage the second stage of partitioning.
    *  After the initial partitioning is done using primary histograms, the data is further divided into more sub-partitions. These second-level histograms
    *  count the elements in each sub-partition during this second stage of partitioning.
    * */
    // Allocate device memory for the partitioned build and probe tables
    key_type* build_tbl_p = nullptr;
    key_type* probe_tbl_p = nullptr;

    // Allocate memory for the partitioned build table on the GPU
    CUDA_RT_CALL( cudaMalloc((void**)&build_tbl_p, build_tbl_size * sizeof(key_type)) );
    device_memory_used += build_tbl_size * sizeof(key_type);

    // Allocate memory for the partitioned probe table on the GPU
    CUDA_RT_CALL( cudaMalloc((void**)&probe_tbl_p, probe_tbl_size * sizeof(key_type)) );
    device_memory_used += probe_tbl_size * sizeof(key_type);

    // Allocate device memory for histograms
    size_type* build_histo = nullptr;
    size_type* probe_histo = nullptr;
    size_type build_histo_size = num_partitions_1;
    size_type probe_histo_size = num_partitions_1;

    // Allocate memory for the build table histogram on the GPU
    CUDA_RT_CALL( cudaMalloc((void**)&build_histo, build_histo_size * sizeof(size_type)) );
    device_memory_used += build_histo_size * sizeof(size_type);

    // Initialize build table histogram memory to zero
    CUDA_RT_CALL( cudaMemset(build_histo, 0, build_histo_size * sizeof(size_type)) );

    // Allocate memory for the probe table histogram on the GPU
    CUDA_RT_CALL( cudaMalloc((void**)&probe_histo, probe_histo_size * sizeof(size_type)) );
    device_memory_used += probe_histo_size * sizeof(size_type);

    // Initialize probe table histogram memory to zero
    CUDA_RT_CALL( cudaMemset(probe_histo, 0, probe_histo_size * sizeof(size_type)) );


    // Allocate device memory for secondary histograms
    size_type* build_histo_s = nullptr;
    size_type* probe_histo_s = nullptr;

    // Allocate memory for the secondary build table histogram on the GPU
    CUDA_RT_CALL( cudaMalloc((void**)&build_histo_s, build_histo_size * sizeof(size_type)) );
    device_memory_used += build_histo_size * sizeof(size_type);

    // Initialize secondary build table histogram memory to zero
    CUDA_RT_CALL( cudaMemset(build_histo_s, 0, build_histo_size * sizeof(size_type)) );

    // Allocate memory for the secondary probe table histogram on the GPU
    CUDA_RT_CALL( cudaMalloc((void**)&probe_histo_s, probe_histo_size * sizeof(size_type)) );
    device_memory_used += probe_histo_size * sizeof(size_type);

    // Initialize secondary probe table histogram memory to zero
    CUDA_RT_CALL( cudaMemset(probe_histo_s, 0, probe_histo_size * sizeof(size_type)) );

    // Allocate device memory for second-level histograms for radix partitioning
    size_type* build_histo_p2 = nullptr;
    size_type* probe_histo_p2 = nullptr;
    size_type* build_histo_s_p2 = nullptr;
    size_type* probe_histo_s_p2 = nullptr;
    size_type build_histo_size_p2 = num_partitions;
    size_type probe_histo_size_p2 = num_partitions;

    if (radix_partition) {
        // Allocate memory for the second-level build table histogram on the GPU
        CUDA_RT_CALL( cudaMalloc((void**)&build_histo_p2, build_histo_size_p2 * sizeof(size_type)) );
        device_memory_used += build_histo_size_p2 * sizeof(size_type);

        // Initialize second-level build table histogram memory to zero
        CUDA_RT_CALL( cudaMemset(build_histo_p2, 0, build_histo_size_p2 * sizeof(size_type)) );

        // Allocate memory for the second-level probe table histogram on the GPU
        CUDA_RT_CALL( cudaMalloc((void**)&probe_histo_p2, probe_histo_size_p2 * sizeof(size_type)) );
        device_memory_used += probe_histo_size_p2 * sizeof(size_type);

        // Initialize second-level probe table histogram memory to zero
        CUDA_RT_CALL( cudaMemset(probe_histo_p2, 0, probe_histo_size_p2 * sizeof(size_type)) );

        // Allocate memory for the secondary second-level build table histogram on the GPU
        CUDA_RT_CALL( cudaMalloc((void**)&build_histo_s_p2, build_histo_size_p2 * sizeof(size_type)) );
        device_memory_used += build_histo_size_p2 * sizeof(size_type);

        // Initialize secondary second-level build table histogram memory to zero
        CUDA_RT_CALL( cudaMemset(build_histo_s_p2, 0, build_histo_size_p2 * sizeof(size_type)) );

        // Allocate memory for the secondary second-level probe table histogram on the GPU
        CUDA_RT_CALL( cudaMalloc((void**)&probe_histo_s_p2, probe_histo_size_p2 * sizeof(size_type)) );
        device_memory_used += probe_histo_size_p2 * sizeof(size_type);

        // Initialize secondary second-level probe table histogram memory to zero
        CUDA_RT_CALL( cudaMemset(probe_histo_s_p2, 0, probe_histo_size_p2 * sizeof(size_type)) );
    }

    /*
    * Summary:
    * The code is preparing memory allocations and initializations for handling skewed data and
    * performing a second-level partitioning in a radix-based partitioning scheme for both build
    * and probe tables in a GPU environment.
    * 1. Declares pointers for various arrays that will store partition indices and block counts for the build and probe tables.
    * 2. Memory Allocation and Initialization for Partition Indices:
    * Allocates and initializes memory for the partition indices to handle the second partitioning pass.
    * 3. Allocates and initializes memory for tracking the number of blocks in each partition for both the primary and secondary passes
    * 4. Allocates pinned host memory for storing the number of blocks involved in the second partitioning pass for the build and probe tables
    *
    * */

    /* Reasons for Using Blocks
    * Parallel Processing: GPU Architecture: GPUs are designed to handle many threads running in parallel. These threads are organized into blocks.
    * By breaking down the data into blocks, the GPU can process multiple blocks in parallel, significantly speeding up computations.
    *
    * Scalability: Blocks allow the algorithm to scale across multiple Streaming Multiprocessors (SMs) on the GPU. Each block can be assigned to a
    * different SM, utilizing the full computational power of the GPU.
    * */

    // extra memory needed to handle skewed data for partition pass 2
    size_type* partition_index_p2_build = nullptr, * partition_index_p2_probe = nullptr; 
    size_type* partition_index_alt_p2_build = nullptr, * partition_index_alt_p2_probe = nullptr;
    size_type* num_block_p2_build_h = nullptr, * num_block_p2_probe_h;
    size_type* num_blocks_each_partition_p2_build = nullptr, * num_blocks_each_partition_p2_probe = nullptr;
    size_type* num_blocks_each_partition_s_p2_build = nullptr, * num_blocks_each_partition_s_p2_probe = nullptr;

    if (radix_partition) {
        // over allocate as number of blocks needed may be larger than number of partitions  
        int avg_elem_in_blk_build = ceil(1.0 * build_tbl_size / num_partitions_1);
        int num_blk_per_part_build = ceil(1.0 * avg_elem_in_blk_build / (BLOCK_SIZE_PARTITION * N_UNROLL));
        int num_block_p2_build = num_partitions_1 * num_blk_per_part_build;

        // Allocate memory for build table partition index for the second partitioning pass
        CUDA_RT_CALL( cudaMalloc((void**)&partition_index_p2_build, num_block_p2_build * sizeof(size_type) * DATA_SKEW_OVERALLOCATE_FACTOR) );
        device_memory_used += num_block_p2_build * sizeof(size_type) * DATA_SKEW_OVERALLOCATE_FACTOR;
        CUDA_RT_CALL( cudaMemset(partition_index_p2_build, 0, num_block_p2_build * sizeof(size_type)) );

        // Allocate memory for alternative build table partition index for the second partitioning pass
        CUDA_RT_CALL( cudaMalloc((void**)&partition_index_alt_p2_build, num_block_p2_build * sizeof(size_type) * DATA_SKEW_OVERALLOCATE_FACTOR) );
        device_memory_used += num_block_p2_build * sizeof(size_type) * DATA_SKEW_OVERALLOCATE_FACTOR;
        CUDA_RT_CALL( cudaMemset(partition_index_alt_p2_build, 0, num_block_p2_build * sizeof(size_type)) );

        // Perform similar calculations and allocations for the probe table
        int avg_elem_in_blk_probe = ceil(1.0 * probe_tbl_size / num_partitions_1);
        int num_blk_per_part_probe = ceil(1.0 * avg_elem_in_blk_probe / (BLOCK_SIZE_PARTITION * N_UNROLL));
        int num_block_p2_probe = num_partitions_1 * num_blk_per_part_probe;

        // Allocate memory for probe table partition index for the second partitioning pass
        CUDA_RT_CALL( cudaMalloc((void**)&partition_index_p2_probe, num_block_p2_probe * sizeof(size_type) * DATA_SKEW_OVERALLOCATE_FACTOR) );
        device_memory_used += num_block_p2_probe * sizeof(size_type) * DATA_SKEW_OVERALLOCATE_FACTOR;
        CUDA_RT_CALL( cudaMemset(partition_index_p2_probe, 0, num_block_p2_probe * sizeof(size_type)) );

        // Allocate memory for alternative probe table partition index for the second partitioning pass
        CUDA_RT_CALL( cudaMalloc((void**)&partition_index_alt_p2_probe, num_block_p2_probe * sizeof(size_type) * DATA_SKEW_OVERALLOCATE_FACTOR) );
        device_memory_used += num_block_p2_probe * sizeof(size_type) * DATA_SKEW_OVERALLOCATE_FACTOR;
        CUDA_RT_CALL( cudaMemset(partition_index_alt_p2_probe, 0, num_block_p2_probe * sizeof(size_type)) );

        // Allocate and initialize memory for the number of blocks in each partition for the build table
        CUDA_RT_CALL( cudaMalloc((void**)&num_blocks_each_partition_p2_build, (num_partitions_1 + 1) * sizeof(size_type)) );
        device_memory_used += num_partitions_1 * sizeof(size_type);
        CUDA_RT_CALL( cudaMemset(num_blocks_each_partition_p2_build, 0, (num_partitions_1 + 1) * sizeof(size_type)) );

        // Allocate and initialize memory for the number of blocks in each partition for the secondary build table
        CUDA_RT_CALL( cudaMalloc((void**)&num_blocks_each_partition_s_p2_build, (num_partitions_1 + 1) * sizeof(size_type)) );
        device_memory_used += num_partitions_1 * sizeof(size_type);
        CUDA_RT_CALL( cudaMemset(num_blocks_each_partition_s_p2_build, 0, (num_partitions_1 + 1) * sizeof(size_type)) );

        // Allocate and initialize memory for the number of blocks in each partition for the probe table
        CUDA_RT_CALL( cudaMalloc((void**)&num_blocks_each_partition_p2_probe, (num_partitions_1 + 1) * sizeof(size_type)) );
        device_memory_used += num_partitions_1 * sizeof(size_type);
        CUDA_RT_CALL( cudaMemset(num_blocks_each_partition_p2_probe, 0, (num_partitions_1 + 1) * sizeof(size_type)) );

        // Allocate and initialize memory for the number of blocks in each partition for the secondary probe table
        CUDA_RT_CALL( cudaMalloc((void**)&num_blocks_each_partition_s_p2_probe, (num_partitions_1 + 1) * sizeof(size_type)) );
        device_memory_used += num_partitions_1 * sizeof(size_type);
        CUDA_RT_CALL( cudaMemset(num_blocks_each_partition_s_p2_probe, 0, (num_partitions_1 + 1) * sizeof(size_type)) );

        // Allocate pinned host memory for the number of blocks in the second partitioning pass for both build and probe tables
        CUDA_RT_CALL( cudaMallocHost((void**)&num_block_p2_build_h, sizeof(size_type)) );
        CUDA_RT_CALL( cudaMallocHost((void**)&num_block_p2_probe_h, sizeof(size_type)) );
    }

    // extra memory needed to handle skewed data for join kernel 
    size_type* partition_index = nullptr; 
    size_type* partition_index_alt = nullptr;
    // over allocate as number of blocks needed may be larger than number of partitions  
    CUDA_RT_CALL( cudaMalloc((void**)&partition_index, num_partitions * sizeof(size_type) * DATA_SKEW_OVERALLOCATE_FACTOR) );
    device_memory_used += num_partitions * sizeof(size_type) * DATA_SKEW_OVERALLOCATE_FACTOR;
    CUDA_RT_CALL( cudaMemset(partition_index, 0, num_partitions * sizeof(size_type)) );

    // Allocate alternate partition index for the join kernel
    CUDA_RT_CALL( cudaMalloc((void**)&partition_index_alt, num_partitions * sizeof(size_type) * DATA_SKEW_OVERALLOCATE_FACTOR) );
    device_memory_used += num_partitions * sizeof(size_type) * DATA_SKEW_OVERALLOCATE_FACTOR;
    CUDA_RT_CALL( cudaMemset(partition_index_alt, 0, num_partitions * sizeof(size_type)) );

    // Allocate memory for the number of blocks in each partition
    size_type* num_blocks_each_partition = nullptr;
    size_type* num_blocks_each_partition_s = nullptr;
    CUDA_RT_CALL( cudaMalloc((void**)&num_blocks_each_partition, (num_partitions + 1) * sizeof(size_type)) );
    device_memory_used += num_partitions * sizeof(size_type);
    CUDA_RT_CALL( cudaMemset(num_blocks_each_partition, 0, (num_partitions + 1) * sizeof(size_type)) );

    // Allocate memory for the secondary number of blocks in each partition
    CUDA_RT_CALL( cudaMalloc((void**)&num_blocks_each_partition_s, (num_partitions + 1) * sizeof(size_type)) );
    device_memory_used += num_partitions * sizeof(size_type);
    CUDA_RT_CALL( cudaMemset(num_blocks_each_partition_s, 0, (num_partitions + 1) * sizeof(size_type)) );

    // Allocate pinned (page-locked) host memory for the number of blocks for the join kernel
    size_type* num_block_join_h = nullptr;
    CUDA_RT_CALL( cudaMallocHost((void**)&num_block_join_h, sizeof(size_type)) );

    // Estimate the size of the joined output based on the size of the probe table
    size_type joined_size = probe_tbl_size;
    // reference CPU run 
    std::vector<joined_type> joined_h;
    double runtime_reference = -1; 

    if (!skip_cpu_run) {
        if (!csv) {
            std::cout << "[*] Running on CPU" << std::endl;
        }
        joined_h.reserve(joined_size);
        runtime_reference = inner_join_cpu(build_tbl_h, uniq_build_tbl_keys, probe_tbl_h, joined_h);
        joined_size = std::max(1ul, joined_h.size());
    } else {
        if (!csv) {
            std::cout << "[!] Warning: CPU run is skipped, using an estimated output size which may cause error." << std::endl;
        }
        joined_size = output_size;
    }

    // Allocate memory for the joined output table on the GPU
    joined_type* joined_tbl = nullptr;
    CUDA_RT_CALL( cudaMalloc((void**)&joined_tbl, joined_size * sizeof(joined_type)) );
    device_memory_used += joined_size * sizeof(joined_type);

    // Initialize the joined table memory to zero
    CUDA_RT_CALL( cudaMemset(joined_tbl, 0, joined_size * sizeof(joined_type)) );

    // Allocate memory for a global index used during the join process
    size_type *global_index = nullptr;
    CUDA_RT_CALL( cudaMalloc((void**)&global_index, sizeof(size_type)) );
    device_memory_used += sizeof(size_type);

    // Initialize the global index memory to zero
    CUDA_RT_CALL( cudaMemset(global_index, 0, sizeof(size_type)) );

    // Declare a pointer for temporary storage and variables for its size
    void *temp_storage = NULL;
    size_t temp_storage_bytes = 0, temp_storage_bytes_temp = 0;

    // Perform exclusive prefix sum (scan) operations to prepare histograms
    // Determine the required temporary storage size for CUB operations
    cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_bytes_temp, build_histo, build_histo_s, build_histo_size);
    temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes_temp);
    cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_bytes_temp, probe_histo, probe_histo_s, probe_histo_size);
    temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes_temp);
    cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_bytes_temp, build_histo_p2, build_histo_s_p2, build_histo_size_p2);
    temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes_temp);
    cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_bytes_temp, probe_histo_p2, probe_histo_s_p2, probe_histo_size_p2);
    temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes_temp);

    // Perform inclusive prefix sum (scan) operations for block counts
    cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes_temp, num_blocks_each_partition, num_blocks_each_partition_s, num_partitions + 1);
    temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes_temp);
    cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes_temp, num_blocks_each_partition_p2_build, num_blocks_each_partition_s_p2_build, num_partitions_1 + 1);
    temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes_temp);
    cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes_temp, num_blocks_each_partition_p2_probe, num_blocks_each_partition_s_p2_probe, num_partitions_1 + 1);
    temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes_temp);

    // Perform inclusive prefix sum for partition indices
    cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes_temp, partition_index_alt, partition_index, num_partitions * DATA_SKEW_OVERALLOCATE_FACTOR);
    temp_storage_bytes = std::max(temp_storage_bytes, temp_storage_bytes_temp);

    // Allocate temporary storage for CUB operations based on the maximum required size
    CUDA_RT_CALL( cudaMalloc((void**)&temp_storage, temp_storage_bytes) );
    device_memory_used += temp_storage_bytes;

    // Print message indicating GPU execution, if not in CSV mode
    if (!csv) {
        std::cout << "[*] Running on GPU" << std::endl;
    }

    // Start CUDA profiler to measure performance
    CUDA_RT_CALL( cudaProfilerStart() );

    // Set shared memory configuration based on the uniqueness of build table keys and presence of values
    set_shared_memory_config(uniq_build_tbl_keys, with_value); 

    // Variables to measure elapsed time for various stages of the join operation
    float join_elapsed_ms = 0, combined_histo_elapsed_ms = 0, partition_one_elapsed_ms = 0, 
        partition_two_elapsed_ms = 0, data_skew_handle_elapased_ms = 0, total_elapsed_ms = 0;

    // Create CUDA events for timing
    cudaEvent_t start, stop;

    CUDA_RT_CALL( cudaEventCreate(&start) );
    CUDA_RT_CALL( cudaEventCreate(&stop) );

    // Determine which histograms to use based on whether radix partitioning is enabled
    size_type* join_build_global_histo = radix_partition ? build_histo_p2 : build_histo; 
    size_type* join_probe_global_histo = radix_partition ? probe_histo_p2 : probe_histo; 

    // If combined histogram pass is enabled, record the start time
    if (do_combined_histo_pass) {
        CUDA_RT_CALL( cudaEventRecord(start, 0) );

        // Perform combined histogram pass for the build table
        combined_histogram_pass(
            join_build_global_histo, build_histo, build_tbl, build_tbl_size, 
            num_partitions, num_partitions_1, num_partitions_2, radix_partition
        );

        // Perform combined histogram pass for the probe table
        combined_histogram_pass(
            join_probe_global_histo, probe_histo, probe_tbl, probe_tbl_size, 
            num_partitions, num_partitions_1, num_partitions_2, radix_partition
        );

        // Record the stop time and calculate the elapsed time for the combined histogram pass
        CUDA_RT_CALL( cudaEventRecord(stop, 0) );
        CUDA_RT_CALL( cudaEventSynchronize(stop) );
        CUDA_RT_CALL( cudaEventElapsedTime(&combined_histo_elapsed_ms, start, stop) );
    }

    // Record the start time for the first partition pass
    CUDA_RT_CALL( cudaEventRecord(start, 0) );

    // Perform the first partition pass for the build table
    partition_input_tables_pass_one<key_type, size_type, value_type>(
        build_tbl_p, build_tbl_val_p, build_histo, build_histo_s, build_histo_size, build_tbl, 
        build_tbl_val, build_tbl_size, num_partitions_1, temp_storage, temp_storage_bytes, with_value, num_bits_2, do_combined_histo_pass
    );

    // Perform the first partition pass for the probe table
    partition_input_tables_pass_one<key_type, size_type, value_type>(
        probe_tbl_p, probe_tbl_val_p, probe_histo, probe_histo_s, probe_histo_size, probe_tbl, 
        probe_tbl_val, probe_tbl_size, num_partitions_1, temp_storage, temp_storage_bytes, with_value, num_bits_2, do_combined_histo_pass
    );

    // Record the stop time and calculate the elapsed time for the first partition pass
    CUDA_RT_CALL( cudaEventRecord(stop, 0) );
    CUDA_RT_CALL( cudaEventSynchronize(stop) );
    CUDA_RT_CALL( cudaEventElapsedTime(&partition_one_elapsed_ms, start, stop) );

    if (radix_partition) {
        // partition pass two     
        CUDA_RT_CALL( cudaEventRecord(start, 0) );

        int num_unroll = with_value ? N_UNROLL_W_VAL : N_UNROLL; 

        handle_skewed_data<size_type>(
            build_histo, build_tbl_size, BLOCK_SIZE_PARTITION * num_unroll, 
            num_partitions_1, partition_index_p2_build, partition_index_alt_p2_build, 
            num_blocks_each_partition_p2_build, num_blocks_each_partition_s_p2_build, num_block_p2_build_h, 
            temp_storage, temp_storage_bytes
        ); 

        partition_input_tables_pass_two<key_type, size_type, value_type>(
            build_tbl, build_tbl_val, build_histo_p2, build_histo_s_p2, build_histo_size_p2, build_tbl_p, 
            build_tbl_val_p, build_tbl_size, num_partitions_1, num_partitions_2, num_blocks_each_partition_s_p2_build, 
            partition_index_p2_build, *num_block_p2_build_h, temp_storage, 
            temp_storage_bytes, build_histo, with_value, do_combined_histo_pass
        );

        handle_skewed_data<size_type>(
            probe_histo, probe_tbl_size, BLOCK_SIZE_PARTITION * num_unroll, 
            num_partitions_1, partition_index_p2_probe, partition_index_alt_p2_probe, 
            num_blocks_each_partition_p2_probe, num_blocks_each_partition_s_p2_probe, num_block_p2_probe_h, 
            temp_storage, temp_storage_bytes
        ); 

        partition_input_tables_pass_two<key_type, size_type, value_type>(
            probe_tbl, probe_tbl_val, probe_histo_p2, probe_histo_s_p2, probe_histo_size_p2, probe_tbl_p, 
            probe_tbl_val_p, probe_tbl_size, num_partitions_1, num_partitions_2, num_blocks_each_partition_s_p2_probe, 
            partition_index_p2_probe, *num_block_p2_probe_h, temp_storage, 
            temp_storage_bytes, probe_histo, with_value, do_combined_histo_pass
        );

        CUDA_RT_CALL( cudaEventRecord(stop, 0) );
        CUDA_RT_CALL( cudaEventSynchronize(stop) );
        CUDA_RT_CALL( cudaEventElapsedTime(&partition_two_elapsed_ms, start, stop) );
    }

    // handle skewed data 
    CUDA_RT_CALL( cudaEventRecord(start, 0) );
    key_type* join_build_tbl = radix_partition ? build_tbl : build_tbl_p; 
    key_type* join_probe_tbl = radix_partition ? probe_tbl : probe_tbl_p; 
    value_type* join_build_tbl_val = radix_partition ? build_tbl_val : build_tbl_val_p; 
    value_type* join_probe_tbl_val = radix_partition ? probe_tbl_val : probe_tbl_val_p; 

    // factor to make each block handle more build table data 
    size_type base_build_tbl_size_each_part = ceil(max_partition_size_factor * build_tbl_size / num_partitions); 

    handle_skewed_data<size_type>(
        join_build_global_histo, build_tbl_size, base_build_tbl_size_each_part, 
        num_partitions, partition_index, partition_index_alt, 
        num_blocks_each_partition, num_blocks_each_partition_s, num_block_join_h, 
        temp_storage, temp_storage_bytes
    );

    CUDA_RT_CALL( cudaEventRecord(stop, 0) );
    CUDA_RT_CALL( cudaEventSynchronize(stop) );
    CUDA_RT_CALL( cudaEventElapsedTime(&data_skew_handle_elapased_ms, start, stop) );

    // join 
    CUDA_RT_CALL( cudaEventRecord(start, 0) );
    shared_memory_hash_join<key_type, size_type, value_type, joined_type>(
        joined_tbl, join_build_tbl, join_build_tbl_val, build_tbl_size, 
        join_probe_tbl, join_probe_tbl_val, probe_tbl_size, 
        base_build_tbl_size_each_part, 
        global_index, num_blocks_each_partition_s, partition_index, *num_block_join_h,
        num_partitions, uniq_build_tbl_keys, join_build_global_histo, join_probe_global_histo, 
        with_value, joined_size
    );
    CUDA_RT_CALL( cudaEventRecord(stop, 0) );
    CUDA_RT_CALL( cudaEventSynchronize(stop) );
    CUDA_RT_CALL( cudaEventElapsedTime(&join_elapsed_ms, start, stop) );
    CUDA_RT_CALL( cudaProfilerStop() );

    total_elapsed_ms = join_elapsed_ms + combined_histo_elapsed_ms 
        + data_skew_handle_elapased_ms + partition_one_elapsed_ms + partition_two_elapsed_ms; 

    key_type* build_tbl_pinned = nullptr;
    key_type* probe_tbl_pinned = nullptr;
    joined_type* joined_tbl_pinned = nullptr;
        
    CUDA_RT_CALL( cudaMallocHost((void**)&build_tbl_pinned, build_tbl_size * sizeof(key_type)) );
    CUDA_RT_CALL( cudaMallocHost((void**)&probe_tbl_pinned, probe_tbl_size * sizeof(key_type)) );
    CUDA_RT_CALL( cudaMallocHost((void**)&joined_tbl_pinned, joined_size * sizeof(joined_type)) );

    CUDA_RT_CALL( cudaMemcpy(build_tbl_pinned, build_tbl, build_tbl_size * sizeof(key_type), cudaMemcpyDefault) );
    CUDA_RT_CALL( cudaMemcpy(probe_tbl_pinned, probe_tbl, probe_tbl_size * sizeof(key_type), cudaMemcpyDefault) );
    CUDA_RT_CALL( cudaMemcpy(joined_tbl_pinned, joined_tbl, joined_size * sizeof(joined_type), cudaMemcpyDefault) );

    int num_errors = -1;
    if (!skip_cpu_run) {
        num_errors = check_results(
            joined_tbl_pinned, joined_size, build_tbl_pinned, build_tbl_size, 
            probe_tbl_pinned, probe_tbl_size, joined_h, build_tbl_h, probe_tbl_h, verbosity
        );
    }

    // print result
    if (csv) {
        std::cout << joined_size << "," << combined_histo_elapsed_ms << "," 
        << partition_one_elapsed_ms << "," << partition_two_elapsed_ms << ","
        << data_skew_handle_elapased_ms << "," <<  join_elapsed_ms << "," 
        << total_elapsed_ms << "," << num_errors << std::endl;
    } else {
        std::cout << "[*] Results:" << std::endl;
        std::cout << "    joined size          = " << joined_size << std::endl;
        if (do_combined_histo_pass) {
            std::cout << "    combined histogram time    = " << std::setprecision(3) << combined_histo_elapsed_ms << " ms" << std::endl;
        }
        std::cout << "    partition P1 time    = " << std::setprecision(3) << partition_one_elapsed_ms << " ms" << std::endl;
        if (radix_partition) {
            std::cout << "    partition P2 time    = " << std::setprecision(3) << partition_two_elapsed_ms << " ms" << std::endl;
        }
        std::cout << "    skewed data time     = " << std::setprecision(3) << data_skew_handle_elapased_ms << " ms" << std::endl;
        std::cout << "    join phase time      = " << std::setprecision(3) << join_elapsed_ms << " ms" << std::endl;
        std::cout << "    total exec time      = " << std::setprecision(3) << total_elapsed_ms << " ms" << std::endl;
        long long input_size = with_value ? ((probe_tbl_size + build_tbl_size)*(sizeof(key_type) + sizeof(value_type)))
                                : ((probe_tbl_size + build_tbl_size)*sizeof(key_type));
        std::cout << "    effective BW         = " << input_size/(total_elapsed_ms/1000*GIGABYTE) << " GB/s" << std::endl;
        std::cout << "    device memory usage  = " << device_memory_used / 1e9 << " GB" << std::endl;
        if (!skip_cpu_run) {
            std::cout << "    reference CPU time   = " << std::setprecision(5) << 1000.0 * runtime_reference << " ms" << std::endl;
            std::cout << "    number of errors     = " << num_errors << std::endl;
        } else {
            std::cout << "    reference CPU time   = SKIPPED" << std::endl;
            std::cout << "    number of errors     = SKIPPED" << std::endl;
        }
    }

    // free
    CUDA_RT_CALL( cudaFree(build_tbl) );
    CUDA_RT_CALL( cudaFree(probe_tbl) );
    CUDA_RT_CALL( cudaFree(build_tbl_p) );
    CUDA_RT_CALL( cudaFree(probe_tbl_p) );
    CUDA_RT_CALL( cudaFree(build_histo) );
    CUDA_RT_CALL( cudaFree(probe_histo) );
    CUDA_RT_CALL( cudaFree(build_histo_s) );
    CUDA_RT_CALL( cudaFree(probe_histo_s) );
    if (radix_partition) {
        CUDA_RT_CALL( cudaFree(build_histo_p2) );
        CUDA_RT_CALL( cudaFree(probe_histo_p2) );
        CUDA_RT_CALL( cudaFree(build_histo_s_p2) );
        CUDA_RT_CALL( cudaFree(probe_histo_s_p2) );
        CUDA_RT_CALL( cudaFree(partition_index_p2_build) );
        CUDA_RT_CALL( cudaFree(partition_index_p2_probe) );
        CUDA_RT_CALL( cudaFree(partition_index_alt_p2_build) );
        CUDA_RT_CALL( cudaFree(partition_index_alt_p2_probe) );
        CUDA_RT_CALL( cudaFree(num_blocks_each_partition_p2_build) );
        CUDA_RT_CALL( cudaFree(num_blocks_each_partition_p2_probe) );
        CUDA_RT_CALL( cudaFree(num_blocks_each_partition_s_p2_build) );
        CUDA_RT_CALL( cudaFree(num_blocks_each_partition_s_p2_probe) );
        CUDA_RT_CALL( cudaFreeHost(num_block_p2_build_h) );
        CUDA_RT_CALL( cudaFreeHost(num_block_p2_probe_h) );
    }
    if (with_value) {
        CUDA_RT_CALL( cudaFree(build_tbl_val) );
        CUDA_RT_CALL( cudaFree(probe_tbl_val) );
        CUDA_RT_CALL( cudaFree(build_tbl_val_p) );
        CUDA_RT_CALL( cudaFree(probe_tbl_val_p) );
    }
    CUDA_RT_CALL( cudaFree(joined_tbl) );
    CUDA_RT_CALL( cudaFree(temp_storage) );
    CUDA_RT_CALL( cudaFree(global_index) );
    CUDA_RT_CALL( cudaFreeHost(build_tbl_pinned) );
    CUDA_RT_CALL( cudaFreeHost(probe_tbl_pinned) );
    CUDA_RT_CALL( cudaFreeHost(joined_tbl_pinned) );

    CUDA_RT_CALL( cudaFree(partition_index) );
    CUDA_RT_CALL( cudaFree(partition_index_alt) );
    CUDA_RT_CALL( cudaFree(num_blocks_each_partition) );
    CUDA_RT_CALL( cudaFree(num_blocks_each_partition_s) );
    CUDA_RT_CALL( cudaFreeHost(num_block_join_h) );

    return 0; 
}