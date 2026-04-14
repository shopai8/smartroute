cmake_minimum_required(VERSION 3.15)
project(ANNS)

# cmake setting
set(CMAKE_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 将 ACORN/faiss 项目作为 UNG 项目的一个子目录添加进来
set(FAISS_ENABLE_GPU OFF CACHE BOOL "Disable Faiss GPU support")
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "Disable Faiss Python bindings")
set(BUILD_TESTING OFF CACHE BOOL "Disable Faiss tests")
add_subdirectory(../../ACORN ${CMAKE_BINARY_DIR}/acorn_build)

# Boost
find_package(Boost REQUIRED COMPONENTS program_options filesystem)

if(NOT Boost_FOUND)
    message(FATAL_ERROR "no Boost support")
endif()

message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost_LIBRARIES: ${Boost_LIBRARIES}")

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

# build options
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0 -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")
option(UNG_FORCE_GLIBCXX_ABI0 "Force -D_GLIBCXX_USE_CXX11_ABI=0 for legacy prebuilt dependencies" OFF)
if(UNG_FORCE_GLIBCXX_ABI0)
    add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)
    message(WARNING "UNG_FORCE_GLIBCXX_ABI0=ON: building with old libstdc++ ABI (0).")
else()
    message(STATUS "UNG_FORCE_GLIBCXX_ABI0=OFF: using default libstdc++ ABI (usually 1).")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma -msse2 -ftree-vectorize -fno-builtin-malloc -fno-builtin-calloc -fno-builtin-realloc -fno-builtin-free -fopenmp -fopenmp-simd -funroll-loops -Wfatal-errors -DUSE_AVX2")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -g -DNDEBUG -Ofast -march=native -mtune=native")

# include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/third_party/CRoaring/include)

# Find pre-built CRoaring library
set(ROARING_LIB_DIR "${PROJECT_SOURCE_DIR}/third_party/CRoaring/build/src")
find_library(ROARING_LIB
    NAMES roaring libroaring
    PATHS ${ROARING_LIB_DIR}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)
if(NOT ROARING_LIB)
    message(FATAL_ERROR "CRoaring library not found. Searched in ${ROARING_LIB_DIR} and subdirectories. Please check the build or path.")
else()
    message(STATUS "CRoaring library found at: ${ROARING_LIB}")
endif()

# ONNX
set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.16.3" CACHE PATH "Path to ONNX Runtime directory")
if(EXISTS ${ONNXRUNTIME_DIR})
    include_directories(${ONNXRUNTIME_DIR}/include)
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so"
    )
    message(STATUS "Found ONNX Runtime at: ${ONNXRUNTIME_DIR}")
else()
    message(FATAL_ERROR "ONNX Runtime not found at specified path: ${ONNXRUNTIME_DIR}. Please check the path in CMakeLists.txt")
endif()

# ==============================================================================
# 引入 NaviX (Faiss 1.10.0 独立版)
# ==============================================================================
set(NAVIX_ROOT "${PROJECT_SOURCE_DIR}/../../NaviX")
include_directories(${NAVIX_ROOT})

# 允许从外部构建脚本传入真实的 NAVIX_BUILD_DIR
if(NOT NAVIX_BUILD_DIR)
    set(NAVIX_BUILD_DIR "${NAVIX_ROOT}/build")
endif()

# 查找 libfaiss_avx2.a 或 libfaiss.a
find_library(NAVIX_LIB
    NAMES faiss_avx2 faiss
    PATHS "${NAVIX_BUILD_DIR}/faiss_navix" "${NAVIX_BUILD_DIR}/faiss"
    NO_DEFAULT_PATH
)

if(NOT NAVIX_LIB)
    message(FATAL_ERROR "NaviX library not found in ${NAVIX_BUILD_DIR}! 请先执行 NaviX 的编译。")
else()
    message(STATUS "NaviX library 成功找到，路径为: ${NAVIX_LIB}")
endif()
# ==============================================================================
# ==============================================================================

# add subdirectories
add_subdirectory(vamana)
add_subdirectory(src)
add_subdirectory(tools)
add_subdirectory(test)
add_subdirectory(apps)
