/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss_navix/perf_tests/utils.h>
namespace faiss_navix::perf_tests {
std::map<std::string, faiss_navix::ScalarQuantizer::QuantizerType> sq_types() {
    static std::map<std::string, faiss_navix::ScalarQuantizer::QuantizerType>
            sq_types = {
                    {"QT_8bit", faiss_navix::ScalarQuantizer::QT_8bit},
                    {"QT_4bit", faiss_navix::ScalarQuantizer::QT_4bit},
                    {"QT_8bit_uniform",
                     faiss_navix::ScalarQuantizer::QT_8bit_uniform},
                    {"QT_4bit_uniform",
                     faiss_navix::ScalarQuantizer::QT_4bit_uniform},
                    {"QT_fp16", faiss_navix::ScalarQuantizer::QT_fp16},
                    {"QT_8bit_direct", faiss_navix::ScalarQuantizer::QT_8bit_direct},
                    {"QT_6bit", faiss_navix::ScalarQuantizer::QT_6bit},
                    {"QT_bf16", faiss_navix::ScalarQuantizer::QT_bf16},
                    {"QT_8bit_direct_signed",
                     faiss_navix::ScalarQuantizer::QT_8bit_direct_signed}};
    return sq_types;
}
} // namespace faiss_navix::perf_tests
