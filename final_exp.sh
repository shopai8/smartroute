#!/bin/bash
set -e

echo "=== 开始执行实验... ==="

# echo "$(date): [步骤 1] 运行 experiments-Reviews-1-2..."
# cd /home/fengxiaoyao/FilterVector/FilterVectorCode
# ./exp.sh experiment_json/experiments-Reviews-1-2.json

# echo "$(date): [步骤 0] 运行 experiments-Amazon-FastSmartRoute-1000-2..."
# cd /home/fengxiaoyao/FilterVector/FilterVectorCode
# ./exp.sh experiment_json/202603-FastSmartRoute-small-efs/experiments-Amazon-FastSmartRoute-1000-2.json > output.log

# echo "$(date): [步骤 1] 运行 experiments-BookReviews-FastSmartRoute-1000-2..."
# cd /home/fengxiaoyao/FilterVector/FilterVectorCode
# ./exp.sh experiment_json/202603-FastSmartRoute-small-efs/experiments-BookReviews-FastSmartRoute-1000-2.json > output.log

# echo "$(date): [步骤 2] 运行 experiments-Genome-FastSmartRoute-1000-2..."
# cd /home/fengxiaoyao/FilterVector/FilterVectorCode
# ./exp.sh experiment_json/202603-FastSmartRoute-small-efs/experiments-Genome-FastSmartRoute-1000-2.json > output.log

# echo "$(date): [步骤 3] 运行 experiments-Music-FastSmartRoute-1000-2..."
# cd /home/fengxiaoyao/FilterVector/FilterVectorCode
# ./exp.sh experiment_json/202603-FastSmartRoute-small-efs/experiments-Music-FastSmartRoute-1000-2.json > output.log

# echo "$(date): [步骤 4] 运行 experiments-Reviews-FastSmartRoute-1000-2..."
# cd /home/fengxiaoyao/FilterVector/FilterVectorCode
# ./exp.sh experiment_json/202603-FastSmartRoute-small-efs/experiments-Reviews-FastSmartRoute-1000-2.json > output.log

# echo "$(date): [步骤 5] 运行 experiments-Tiktok-FastSmartRoute-1000-2..."
# cd /home/fengxiaoyao/FilterVector/FilterVectorCode
# ./exp.sh experiment_json/202603-FastSmartRoute-small-efs/experiments-Tiktok-FastSmartRoute-1000-2.json > output.log

# echo "$(date): [步骤 6] 运行 experiments-VariousImg-FastSmartRoute-1000-2..."
# cd /home/fengxiaoyao/FilterVector/FilterVectorCode
# ./exp.sh experiment_json/202603-FastSmartRoute-small-efs/experiments-VariousImg-FastSmartRoute-1000-2.json > output.log

echo "$(date): [步骤 7] 运行 experiments-Laion-FastSmartRoute-1000-2..."
cd /home/fengxiaoyao/FilterVector/FilterVectorCode
./exp.sh experiment_json/202603-FastSmartRoute-small-efs/experiments-Laion-FastSmartRoute-1000-2.json > output.log



echo "$(date): === 所有任务执行完毕。 ==="