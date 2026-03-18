/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "GpuIndex_c.h"
#include <faiss_navix/gpu/GpuIndex.h>
#include "macros_impl.h"

using faiss_navix::gpu::GpuIndexConfig;

DEFINE_GETTER(GpuIndexConfig, int, device)
