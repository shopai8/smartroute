/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss_navix/Clustering.h>
#include <faiss_navix/IndexFlat.h>
#include <faiss_navix/impl/AuxIndexStructures.h>
#include <faiss_navix/impl/FaissException.h>
#include <faiss_navix/utils/random.h>

TEST(TestCallback, timeout) {
    int n = 1000;
    int k = 100;
    int d = 128;
    int niter = 1000000000;
    int seed = 42;

    std::vector<float> vecs(n * d);
    faiss_navix::float_rand(vecs.data(), vecs.size(), seed);

    auto index(new faiss_navix::IndexFlat(d));

    faiss_navix::ClusteringParameters cp;
    cp.niter = niter;
    cp.verbose = false;

    faiss_navix::Clustering kmeans(d, k, cp);

    faiss_navix::TimeoutCallback::reset(0.010);
    EXPECT_THROW(kmeans.train(n, vecs.data(), *index), faiss_navix::FaissException);
    delete index;
}
