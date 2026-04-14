#!/bin/bash
set -e

echo "=== 开始执行实验... ==="

# echo "$(date): [步骤 1] 运行 experiments-Reviews-1-2..."
# cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Reviews-1-2.json

echo "$(date): [步骤 2] 运行 experiments-Genome-navix-prefilter-big..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/20260331/experiments-Genome-navix-prefilter-big.json > output.log

echo "$(date): [步骤 0] 运行 experiments-Amazon-navix-prefilter-big..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/20260331/experiments-Amazon-navix-prefilter-big.json > output.log

echo "$(date): [步骤 1] 运行 experiments-BookReviews-navix-prefilter-big..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/20260331/experiments-BookReviews-navix-prefilter-big.json > output.log



echo "$(date): [步骤 3] 运行 experiments-Music-navix-prefilter-big..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/20260331/experiments-Music-navix-prefilter-big.json > output.log

echo "$(date): [步骤 4] 运行 experiments-Reviews-navix-prefilter-big..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/20260331/experiments-Reviews-navix-prefilter-big.json > output.log

echo "$(date): [步骤 5] 运行 experiments-Tiktok-navix-prefilter-big..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/20260331/experiments-Tiktok-navix-prefilter-big.json > output.log

echo "$(date): [步骤 6] 运行 experiments-VariousImg-navix-prefilter-big..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/20260331/experiments-VariousImg-navix-prefilter-big.json > output.log

echo "$(date): [步骤 7] 运行 experiments-Laion-navix-prefilter-big..."
cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
./exp.sh /home/sunyahui/ljk/FilterVector/FilterVectorCode/experiment_json/20260331/experiments-Laion-navix-prefilter-big.json > output.log

# echo "$(date): [步骤 8] 运行 experiments-Laion-ACORN-big..."
# cd /home/sunyahui/ljk/FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Laion-ACORN-big.json > output.log



echo "$(date): === 所有任务执行完毕。 ==="