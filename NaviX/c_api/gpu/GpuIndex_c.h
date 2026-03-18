/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_NAVIX_GPU_INDEX_C_H
#define FAISS_NAVIX_GPU_INDEX_C_H

#include "../faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

FAISS_NAVIX_DECLARE_CLASS(GpuIndexConfig)

FAISS_NAVIX_DECLARE_GETTER(GpuIndexConfig, int, device)

FAISS_NAVIX_DECLARE_CLASS_INHERITED(GpuIndex, Index)

#ifdef __cplusplus
}
#endif

#endif
