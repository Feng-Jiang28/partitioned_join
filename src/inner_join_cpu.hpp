/* Copyright 2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#ifndef INNER_JOIN_CPU_HPP
#define INNER_JOIN_CPU_HPP
#include <vector>
#include <unordered_map>

#include "nvtx_helper.cuh"
#include <omp.h>
#include "common.cuh"
#include <thrust/sort.h>

// the inner_join_cpu function performs an inner operation on two tables using either
// unique keys or non-unique keys. check_results function verifies the correctness of
// the join operation by comparing the joined results with a reference result, printing
// errors if any mismatches are found. The use of OpenMP for parallel processing and sorting ensures efficient and accurate comparisons.
template<typename key_type, typename joined_type>
double inner_join_cpu(
    std::vector<key_type>& build_tbl,
    const bool uniq_keys,
    std::vector<key_type>& probe_tbl,
    std::vector<joined_type>& joined,
    const int num_threads = 0)
{ 
    PUSH_RANGE("inner_join_cpu",2)
    
    typedef typename std::vector<key_type>::size_type vec_size_t;
    if ( num_threads > 0 )
        omp_set_num_threads(num_threads);

    double start = 0.0;
    double stop  = 0.0;
    if ( uniq_keys )
    {
        std::unordered_map<key_type,vec_size_t> hash_tbl;
        hash_tbl.reserve(2*build_tbl.size());
        const auto end = hash_tbl.end();
        
        const auto joined_capacity = joined.capacity();
        joined.resize( joined_capacity );
        
        start = omp_get_wtime();
        PUSH_RANGE("inner_join_cpu_build",3)

        for ( vec_size_t i = 0; i < build_tbl.size(); ++i ) {
            hash_tbl.insert( std::make_pair( build_tbl[i], i ) );
        }
        POP_RANGE
        
        PUSH_RANGE("inner_join_cpu_probe",4)
        vec_size_t current_idx = 0;
        #pragma omp parallel for shared(current_idx,hash_tbl)
        for ( vec_size_t i = 0; i < probe_tbl.size(); ++i ) {
            auto it = hash_tbl.find(probe_tbl[i]);
            if ( end != it ) {
                joined_type joined_val;
                joined_val.x = it->first;
                joined_val.y = 0;   
                int my_current_idx;
                #pragma omp atomic capture
                my_current_idx=current_idx++;
                
                if ( my_current_idx >= joined_capacity ) {
                    std::cerr<<"Output array is too small: "<<joined_capacity<<std::endl;
                } else {
                    joined[my_current_idx] = joined_val;
                }
            }
        }
        POP_RANGE
        stop = omp_get_wtime();
        joined.resize( current_idx );
    }
    else
    {
        std::unordered_multimap<key_type,vec_size_t> hash_tbl;
        hash_tbl.reserve(2*build_tbl.size());
        
        start = omp_get_wtime();
        PUSH_RANGE("inner_join_cpu_build",3)
        for ( vec_size_t i = 0; i < build_tbl.size(); ++i ) {
            hash_tbl.insert( std::make_pair( build_tbl[i], i ) );
        }
        POP_RANGE
        
        PUSH_RANGE("inner_join_cpu_probe",3)
        for ( vec_size_t i = 0; i < probe_tbl.size(); ++i ) {
            auto range = hash_tbl.equal_range(probe_tbl[i]);
            for (auto it = range.first; it != range.second; ++it) {
                joined_type joined_val;
                joined_val.x = it->first;
                joined_val.y = 0;
                joined.push_back( joined_val );
            }
        }
        POP_RANGE
        stop = omp_get_wtime();
    }
    POP_RANGE
    return (stop-start);
}

template<typename key_type, typename size_type, typename joined_type>
size_type check_results(
    joined_type * const joined, const size_type joined_size,
    const key_type* const build_tbl, const size_type build_tbl_size,
    const key_type* const probe_tbl, const size_type probe_tbl_size,
    std::vector<joined_type>& joined_h,
    const std::vector<key_type>& build_tbl_h, const std::vector<key_type>& probe_tbl_h,
    int verbosity = 0)
{
    PUSH_RANGE("check_results",3)

    thrust::stable_sort(joined, joined+joined_size, [=] (const joined_type lhs, const joined_type rhs) {
        return lhs.x < rhs.x;
    });
    
    thrust::stable_sort(joined, joined+joined_size, [=] (const joined_type lhs, const joined_type rhs) {
        return lhs.y < rhs.y;
    });

    std::stable_sort(joined_h.begin(), joined_h.end(), [=] (const joined_type lhs, const joined_type rhs) {
        return lhs.x < rhs.x;
    });

    std::stable_sort(joined_h.begin(), joined_h.end(), [=] (const joined_type lhs, const joined_type rhs) {
        return lhs.y < rhs.y;
    });

    if (verbosity > 1) {
        std::cout << "====Joined result:" << std::endl;
        for (int i = 0; i < joined_size; ++i) {
            std::cout << joined[i].x << " " << joined[i].y << ", ";
        }
        std::cout << std::endl << "====End" << std::endl;
        std::cout << "====Reference result:" << std::endl;
        for (int i = 0; i < joined_h.size(); ++i) {
            std::cout << joined_h[i].x << " " << joined_h[i].y << ", ";
        }
        std::cout << std::endl << "====End" << std::endl;
    }

    if ( joined_size != joined_h.size() )
    {
        std::cerr<<"ERROR: joined size = "<<joined_size<<" != "<<joined_h.size()<<" (reference)"<<std::endl;
        return std::numeric_limits<size_type>::max();
    }
    
    size_type num_errors = 0;
    for ( size_type i = 0; i < joined_size; ++i ) {
        if (joined[i].y != joined_h[i].y ) {
            if ( 0 == num_errors && verbosity == 1 ) std::cerr<<"ERROR: joined["<<i<<"].y = "<<joined[i].y<<" != "<<joined_h[i].y<<" (reference)"<<std::endl;
            ++num_errors;
        }
        if ( joined[i].x != joined_h[i].x ) {
            if ( 0 == num_errors && verbosity == 1 ) std::cerr<<"ERROR: joined["<<i<<"].x = "<<joined[i].x<<" != "<<joined_h[i].x<<" (reference)"<<std::endl;
            ++num_errors;
        }
    }
    POP_RANGE
    return num_errors;
}

#endif //INNER_JOIN_CPU_HPP
