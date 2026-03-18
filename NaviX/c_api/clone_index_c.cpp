/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-
// I/O code for indexes

#include "clone_index_c.h"
#include <faiss_navix/clone_index.h>
#include "macros_impl.h"

using faiss_navix::Index;
using faiss_navix::IndexBinary;

int faiss_clone_index(const FaissIndex* idx, FaissIndex** p_out) {
    try {
        auto out = faiss_navix::clone_index(reinterpret_cast<const Index*>(idx));
        *p_out = reinterpret_cast<FaissIndex*>(out);
    }
    CATCH_AND_HANDLE
}

int faiss_clone_index_binary(
        const FaissIndexBinary* idx,
        FaissIndexBinary** p_out) {
    try {
        auto out = faiss_navix::clone_binary_index(
                reinterpret_cast<const IndexBinary*>(idx));
        *p_out = reinterpret_cast<FaissIndexBinary*>(out);
    }
    CATCH_AND_HANDLE
}
