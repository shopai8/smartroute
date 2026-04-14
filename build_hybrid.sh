#!/bin/bash

# ==============================================================================
# build_hybrid.sh - 统一构建 UNG、ACORN 和 NaviX 索引
#
# 功能:
# 1. 编译 UNG、ACORN 和 NaviX 的代码
# 2. 检查并转换 fvecs -> bin 数据格式
# 3. 根据 build_mode 构建索引(parallel, serial, ung_only, acorn_only, navix_only)
# ==============================================================================

set -e # 如果任何命令失败，则立即退出
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# --- Step 1: 解析命令行参数 ---
PARAMS=("$@")
while [[ $# -gt 0 ]]; do
    if [[ $1 == --* ]]; then
        key=$(echo "$1" | sed 's/--//' | tr '[:lower:]-' '[:upper:]_')
        
        # 处理模式参数
        if [[ $key == "BUILD_MODE" ]]; then
            BUILD_MODE="$2"
            shift 2
            continue
        fi

        # 处理其他参数
        if [ -z "$2" ]; then
            echo "错误: 参数 $1 缺少值"
            exit 1
        fi
        declare "$key"="$2"
        shift 2
    else
        echo "未知参数: $1"; exit 1
    fi
done

# 验证构建模式
case "$BUILD_MODE" in
    parallel|serial|ung_only|acorn_only|navix_only|all)
        echo "[INFO] Build mode set to: $BUILD_MODE"
        ;;
    *)
        echo "错误: 无效的 build_mode '$BUILD_MODE'。可用选项: parallel, serial, ung_only, acorn_only, navix_only, all"
        exit 1
        ;;
esac

# --- Step 2: 编译代码 ---

# rm -rf /home/sunyahui/ljk/FilterVector/build_para_Reviews/ung

# 1. 先编译 NaviX (因为 UNG 依赖它)
if [ -z "$NAVIX_BUILD_DIR" ]; then
    NAVIX_BUILD_DIR="${SCRIPT_DIR}/NaviX/build"
fi

# echo $NAVIX_BUILD_DIR
# whoami

if [ ! -f "${NAVIX_BUILD_DIR}/faiss_navix/libfaiss.a" ] && [ ! -f "${NAVIX_BUILD_DIR}/faiss_navix/libfaiss_avx2.a" ]; then
    echo "[INFO] NaviX library not found. Compiling..."
    rm -rf "$NAVIX_BUILD_DIR" 
    mkdir -p "$NAVIX_BUILD_DIR"
    cmake -S "${SCRIPT_DIR}/NaviX" -B "$NAVIX_BUILD_DIR" \
        -DFAISS_OPT_LEVEL=avx2 \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release
    make -C "$NAVIX_BUILD_DIR" -j
else
    echo "[INFO] NaviX library found."
fi

# 2. 再编译 UNG
UNG_EXECUTABLE="${UNG_BUILD_DIR}/apps/build_UNG_index"
UNG_SEARCH_EXECUTABLE="${UNG_BUILD_DIR}/apps/search_UNG_index"
if [ ! -f "$UNG_EXECUTABLE" ] || [ ! -f "$UNG_SEARCH_EXECUTABLE" ]; then
    echo "[INFO] UNG executable not found. Compiling..."
    mkdir -p "$UNG_BUILD_DIR"
    cmake -S "${SCRIPT_DIR}/UNG/codes" -B "$UNG_BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DNAVIX_BUILD_DIR="$NAVIX_BUILD_DIR"
    make -C "$UNG_BUILD_DIR" -j
else
    echo "[INFO] UNG executable found."
fi

# 3. 编译 ACORN
ACORN_EXECUTABLE="${ACORN_BUILD_DIR}/demos/test_acorn"
if [ ! -f "$ACORN_EXECUTABLE" ]; then
    echo "[INFO] ACORN executable not found. Compiling..."
    mkdir -p "$ACORN_BUILD_DIR"
    # cmake -S "${SCRIPT_DIR}/ACORN" -B "$ACORN_BUILD_DIR" -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF -DBUILD_TESTING=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release
     cmake -S "${SCRIPT_DIR}/ACORN" -B "$ACORN_BUILD_DIR" \
        -DFAISS_ENABLE_GPU=OFF \
        -DFAISS_ENABLE_PYTHON=OFF \
        -DBUILD_TESTING=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    make -C "$ACORN_BUILD_DIR" -j test_acorn
else
    echo "[INFO] ACORN executable found."
fi

# --- Step 2.5: 检查并转换数据格式 ---
echo "[INFO] Checking and converting data format (if necessary)..."
FVECS_TO_BIN_TOOL="${UNG_BUILD_DIR}/tools/fvecs_to_bin"
BASE_FVECS_FILE="${DATA_DIR}/${DATASET}_base.fvecs"
BASE_BIN_FILE="${DATA_DIR}/${DATASET}_base.bin"

if [ ! -x "$FVECS_TO_BIN_TOOL" ]; then
    echo "Error: Conversion tool not found after compilation at: ${FVECS_TO_BIN_TOOL}"
    exit 1
fi

if [ ! -f "$BASE_BIN_FILE" ]; then
    if [ -f "$BASE_FVECS_FILE" ]; then
        echo "Target file '${BASE_BIN_FILE}' not found. Starting conversion from '${BASE_FVECS_FILE}'..."
        "$FVECS_TO_BIN_TOOL" --data_type float --input_file "$BASE_FVECS_FILE" --output_file "$BASE_BIN_FILE"
        echo "Conversion complete."
    else
        echo "Warning: Source file '${BASE_FVECS_FILE}' not found. Skipping conversion. Build process might fail if base .bin is also missing."
    fi
else
    echo "Target file '${BASE_BIN_FILE}' already exists. Skipping conversion."
fi


# --- Step 3: 构造与 search.sh 兼容的输出目录 ---
if [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "all" ]]; then
    INDEX_BASE_DIR="Index_parallel"
else
    INDEX_BASE_DIR="Index"
fi
echo "[INFO] Index base directory set to: $INDEX_BASE_DIR"
INDEX_DIR_NAME="M${MAX_DEGREE}_LB${LBUILD}_alpha${ALPHA}_C${NUM_CROSS_EDGES}_EP${NUM_ENTRY_POINTS}_AN${ACORN_N}_AM${ACORN_M}_AMB${ACORN_M_BETA}_AG${ACORN_GAMMA}"
INDEX_OUTPUT_DIR="${EXP_OUTPUT_DIR}/${INDEX_BASE_DIR}/${INDEX_DIR_NAME}"

echo "[INFO] Preparing index directory (if not exists): $INDEX_OUTPUT_DIR"
mkdir -p "$INDEX_OUTPUT_DIR/index_files"
mkdir -p "$INDEX_OUTPUT_DIR/acorn_output"
mkdir -p "$INDEX_OUTPUT_DIR/navix_output"
mkdir -p "$INDEX_OUTPUT_DIR/others"

# 定义标记文件路径 ---
UNG_MARKER_FILE="$INDEX_OUTPUT_DIR/index_files/.ung_built"
ACORN_MARKER_FILE="$INDEX_OUTPUT_DIR/acorn_output/.acorn_built"
NAVIX_MARKER_FILE="$INDEX_OUTPUT_DIR/navix_output/.navix_built"

# --- Step 4: 定义构建函数 ---

build_ung() {
    if [ -f "$UNG_MARKER_FILE" ]; then
        echo "[UNG] Marker file found ('$UNG_MARKER_FILE'). Skipping UNG build."
        return 0
    fi
    
    echo "[UNG] Build process started."
    "$UNG_EXECUTABLE" \
        --data_type float --dist_fn L2 --num_threads 60 \
        --max_degree "$MAX_DEGREE" --Lbuild "$LBUILD" --alpha "$ALPHA" --num_cross_edges "$NUM_CROSS_EDGES" \
        --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
        --base_label_file "$DATA_DIR/${DATASET}_base_labels.txt" \
        --base_label_info_file "$DATA_DIR/${DATASET}_base_labels_info.log" \
        --base_label_tree_roots "$DATA_DIR/tree_roots.txt" \
        --index_path_prefix "$INDEX_OUTPUT_DIR/index_files/" \
        --result_path_prefix "$INDEX_OUTPUT_DIR/results/" \
        --scenario general --dataset "$DATASET" \
        > "$INDEX_OUTPUT_DIR/others/ung_build.log" 2>&1
    
    echo "[UNG] Build process finished."
    touch "$UNG_MARKER_FILE"
}

build_acorn() {
    if [ -f "$ACORN_MARKER_FILE" ]; then
        echo "[ACORN] Marker file found ('$ACORN_MARKER_FILE'). Skipping ACORN build."
        return 0
    fi

    echo "[ACORN] Build process started."
    local acorn_base_fvecs_path
    local acorn_base_label_path
    if [[ "$BUILD_MODE" == "serial" || "$BUILD_MODE" == "all" ]]; then
        acorn_base_fvecs_path="${INDEX_OUTPUT_DIR}/index_files/reordered_vecs.fvecs"
        acorn_base_label_path="${INDEX_OUTPUT_DIR}/index_files/reordered_labels.txt"
        echo "[ACORN] Using reordered base vectors: $acorn_base_fvecs_path"
    else
        acorn_base_fvecs_path="${DATA_DIR}/${DATASET}_base.fvecs"
        acorn_base_label_path="${DATA_DIR}/${DATASET}_base_labels.txt"
        if [[ "$BUILD_MODE" == "acorn_only" ]]; then 
            acorn_base_label_path="${DATA_DIR}/${DATASET}_base_labels_reorder_ori.txt"
        fi
        echo "[ACORN] Using original base vectors: $acorn_base_fvecs_path"
    fi
    
    if [ ! -f "$acorn_base_fvecs_path" ] || [ ! -f "$acorn_base_label_path" ]; then
        echo "[ACORN] 错误: 必需的输入文件未找到."
        exit 1
    fi

    "$ACORN_EXECUTABLE" build \
        "$ACORN_N" "$ACORN_GAMMA" "$DATASET" "$ACORN_M" "$ACORN_M_BETA" \
        "$acorn_base_fvecs_path" "$acorn_base_label_path" "dummy_query_path" \
        "dummy_csv_dir" "dummy_avg_csv_dir" "dummy_dis_path" \
        60 1 true "10" \
        "$INDEX_OUTPUT_DIR/acorn_output/acorn.index" \
        "$INDEX_OUTPUT_DIR/acorn_output/acorn1.index" \
        0 \
        > "$INDEX_OUTPUT_DIR/others/acorn_build.log" 2>&1
    
    echo "[ACORN] Build process finished."
    touch "$ACORN_MARKER_FILE"
}

build_navix() {
    if [ -f "$NAVIX_MARKER_FILE" ]; then
        echo "[NAVIX] Marker file found ('$NAVIX_MARKER_FILE'). Skipping NaviX build."
        return 0
    fi

    echo "[NAVIX] Build process started."
    "$UNG_BUILD_DIR/apps/build_navix_index" \
        --base_bin_file "$DATA_DIR/${DATASET}_base.bin" \
        --index_output "$INDEX_OUTPUT_DIR/navix_output/hnsw_base.index" \
        --M "$ACORN_M" \
        --num_threads 60 \
        > "$INDEX_OUTPUT_DIR/others/navix_build.log" 2>&1
    echo "[NAVIX] Build process finished."
    touch "$NAVIX_MARKER_FILE"
}

# --- Step 5: 根据构建模式执行任务 ---
start_time=$(date +%s)

if [[ "$BUILD_MODE" == "parallel" || "$BUILD_MODE" == "all" ]]; then
    echo "--- [Executing in PARALLEL/ALL mode] ---"
    build_ung &
    ung_pid=$!
    build_acorn &
    acorn_pid=$!
    # build_navix &
    # navix_pid=$!
    wait $ung_pid
    wait $acorn_pid
    # wait $navix_pid
elif [ "$BUILD_MODE" == "serial" ]; then
    echo "--- [Executing in SERIAL mode] ---"
    build_ung
    build_acorn
    # build_navix
elif [ "$BUILD_MODE" == "ung_only" ]; then
    echo "--- [Executing in UNG_ONLY mode] ---"
    build_ung
elif [ "$BUILD_MODE" == "acorn_only" ]; then
    echo "--- [Executing in ACORN_ONLY mode] ---"
    build_acorn
elif [ "$BUILD_MODE" == "navix_only" ]; then
    echo "--- [Executing in NAVIX_ONLY mode] ---"
    build_navix
fi

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "--- Build Summary ---"
echo "[SUCCESS] All build tasks complete. Total time: $duration seconds."
echo "Indexes are saved in: $INDEX_OUTPUT_DIR"