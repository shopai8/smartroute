/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <random>

#include "faiss_navix/Index.h"
#include "faiss_navix/IndexHNSW.h"
#include "faiss_navix/index_factory.h"
#include "faiss_navix/index_io.h"
#include "test_util.h"

pthread_mutex_t temp_file_mutex = PTHREAD_MUTEX_INITIALIZER;

TEST(IO, TestReadHNSWPQ_whenSDCDisabledFlagPassed_thenDisableSDCTable) {
    // Create a temp file name with a randomized component for stress runs
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<float> dist(0, 9999999);
    std::string temp_file_name =
            "/tmp/faiss_TestReadHNSWPQ" + std::to_string(int(dist(mt)));
    Tempfilename index_filename(&temp_file_mutex, temp_file_name);

    // Create a HNSW index with PQ encoding
    int d = 32, n = 256;
    std::default_random_engine rng(123);
    std::uniform_real_distribution<float> u(0, 100);
    std::vector<float> vectors(n * d);
    for (size_t i = 0; i < n * d; i++) {
        vectors[i] = u(rng);
    }

    // Build the index and write it to the temp file
    {
        std::unique_ptr<faiss_navix::Index> index_writer(
                faiss_navix::index_factory(d, "HNSW8,PQ4np", faiss_navix::METRIC_L2));
        index_writer->train(n, vectors.data());
        index_writer->add(n, vectors.data());

        faiss_navix::write_index(index_writer.get(), index_filename.c_str());
    }

    // Load index from disk. Confirm that the sdc table is equal to 0 when
    // disable sdc is set
    {
        std::unique_ptr<faiss_navix::IndexHNSWPQ> index_reader_read_write(
                dynamic_cast<faiss_navix::IndexHNSWPQ*>(
                        faiss_navix::read_index(index_filename.c_str())));
        std::unique_ptr<faiss_navix::IndexHNSWPQ> index_reader_sdc_disabled(
                dynamic_cast<faiss_navix::IndexHNSWPQ*>(faiss_navix::read_index(
                        index_filename.c_str(),
                        faiss_navix::IO_FLAG_PQ_SKIP_SDC_TABLE)));

        ASSERT_NE(
                dynamic_cast<faiss_navix::IndexPQ*>(index_reader_read_write->storage)
                        ->pq.sdc_table.size(),
                0);
        ASSERT_EQ(
                dynamic_cast<faiss_navix::IndexPQ*>(
                        index_reader_sdc_disabled->storage)
                        ->pq.sdc_table.size(),
                0);
    }
}
