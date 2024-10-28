/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#ifndef _PARTITIONER_H_
#define _PARTITIONER_H_

// #ifndef _CUDA_COMPILER_
#include <iostream>
// #endif

// #if !defined(_CUDA_COMPILER_) && defined(CUDA_8_0)
#include <atomic>
// #endif

// Partitioner definition -----------------------------------------------------

typedef struct Partitioner {

    int n_tasks;
    int cut;
    int current;

    int thread_id;
    int n_threads;

    // Support for dynamic partitioning
    int strategy;
    std::atomic_int *worklist;

} Partitioner;

// Partitioning strategies
#define STATIC_PARTITIONING 0
#define DYNAMIC_PARTITIONING 1

// Create a partitioner -------------------------------------------------------

inline Partitioner partitioner_create(int n_tasks, float alpha, int thread_id, int n_threads, std::atomic_int *worklist) {
    Partitioner p;
    p.n_tasks = n_tasks;
    p.thread_id = thread_id;
    p.n_threads = n_threads;
    if(alpha >= 0.0 && alpha <= 1.0) {
        p.cut = p.n_tasks * alpha;
        p.strategy = STATIC_PARTITIONING;
    } else {
        p.strategy = DYNAMIC_PARTITIONING;
        p.worklist = worklist;
    }
    return p;
}

// Partitioner iterators: first() ---------------------------------------------

inline int cpu_first(Partitioner *p) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        p->current = p->worklist->fetch_add(1);
    } else {
        p->current = p->thread_id;
    }
    return p->current;
}

// Partitioner iterators: more() ----------------------------------------------

inline bool cpu_more(const Partitioner *p) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        return (p->current < p->n_tasks);
    } else {
        return (p->current < p->cut);
    }
}

// Partitioner iterators: next() ----------------------------------------------

inline int cpu_next(Partitioner *p) {
    if(p->strategy == DYNAMIC_PARTITIONING) {
        p->current = p->worklist->fetch_add(1);
    } else {
        p->current = p->current + p->n_threads;
    }
    return p->current;
}

#endif

