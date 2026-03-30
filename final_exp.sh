#!/bin/bash
set -e

echo "=== 开始执行实验... ==="

# echo "$(date): [步骤 1] 运行 experiments-Reviews-1-2..."
# cd /home/fengxiaoyao/FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Reviews-1-2.json

echo "$(date): [步骤 2] 运行 experiments-Genome-SmartRoute-1000-1..."
cd /home/fengxiaoyao/FilterVector/FilterVectorCode
./exp.sh experiment_json/202603-SmartRoute/experiments-Genome-SmartRoute-1000-1.json > output.log

echo "$(date): [步骤 0] 运行 experiments-Amazon-SmartRoute-1000-1..."
cd /home/fengxiaoyao/FilterVector/FilterVectorCode
./exp.sh experiment_json/202603-SmartRoute/experiments-Amazon-SmartRoute-1000-1.json > output.log

echo "$(date): [步骤 1] 运行 experiments-BookReviews-SmartRoute-1000-1..."
cd /home/fengxiaoyao/FilterVector/FilterVectorCode
./exp.sh experiment_json/202603-SmartRoute/experiments-BookReviews-SmartRoute-1000-1.json > output.log



echo "$(date): [步骤 3] 运行 experiments-Music-SmartRoute-1000-1..."
cd /home/fengxiaoyao/FilterVector/FilterVectorCode
./exp.sh experiment_json/202603-SmartRoute/experiments-Music-SmartRoute-1000-1.json > output.log

echo "$(date): [步骤 4] 运行 experiments-Reviews-SmartRoute-1000-1..."
cd /home/fengxiaoyao/FilterVector/FilterVectorCode
./exp.sh experiment_json/202603-SmartRoute/experiments-Reviews-SmartRoute-1000-1.json > output.log

echo "$(date): [步骤 5] 运行 experiments-Tiktok-SmartRoute-1000-1..."
cd /home/fengxiaoyao/FilterVector/FilterVectorCode
./exp.sh experiment_json/202603-SmartRoute/experiments-Tiktok-SmartRoute-1000-1.json > output.log

echo "$(date): [步骤 6] 运行 experiments-VariousImg-SmartRoute-1000-1..."
cd /home/fengxiaoyao/FilterVector/FilterVectorCode
./exp.sh experiment_json/202603-SmartRoute/experiments-VariousImg-SmartRoute-1000-1.json > output.log

# echo "$(date): [步骤 7] 运行 experiments-Laion-SmartRoute-1000-1..."
# cd /home/fengxiaoyao/FilterVector/FilterVectorCode
# ./exp.sh experiment_json/202603-SmartRoute/experiments-Laion-SmartRoute-1000-1.json > output.log

echo "$(date): [步骤 8] 运行 experiments-Laion-ACORN-big..."
cd /home/fengxiaoyao/FilterVector/FilterVectorCode
./exp.sh experiment_json/experiments-Laion-ACORN-big.json > output.log



echo "$(date): === 所有任务执行完毕。 ==="