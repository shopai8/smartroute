#!/bin/bash
# 1. 替换头文件引入路径
find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cuh" -o -name "CMakeLists.txt" \) -exec sed -i 's/<faiss\//<faiss_navix\//g' {} +
find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cuh" -o -name "CMakeLists.txt" \) -exec sed -i 's/"faiss\//"faiss_navix\//g' {} +

# 2. 替换命名空间定义和使用
find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cuh" \) -exec sed -i 's/namespace faiss/namespace faiss_navix/g' {} +
find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cuh" \) -exec sed -i 's/faiss::/faiss_navix::/g' {} +

# 3. 替换宏定义，避免与 ACORN 的宏冲突
find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.cuh" \) -exec sed -i 's/FAISS_/FAISS_NAVIX_/g' {} +