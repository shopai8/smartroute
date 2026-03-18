#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include <sys/stat.h>
#include <boost/program_options.hpp>
#include <omp.h> // 新增：引入 OpenMP 库以控制线程数

// 引入隔离后的 NaviX 核心组件
#include <faiss_navix/IndexHNSW.h>
#include <faiss_navix/index_io.h>

namespace po = boost::program_options;

// 辅助函数：获取生成文件的大小（字节）
long get_file_size(const std::string &filename) {
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

int main(int argc, char **argv) {
    std::string base_bin_file, index_output;
    int M = 32;
    int efConstruction = 200;
    int num_threads = 1; // 新增：默认线程数为 1

    // 参数解析
    try {
        po::options_description desc{"NaviX Build Arguments"};
        desc.add_options()("help,h", "Print information on arguments");
        desc.add_options()("base_bin_file", po::value<std::string>(&base_bin_file)->required(), "File containing the base vectors in .bin format");
        desc.add_options()("index_output", po::value<std::string>(&index_output)->required(), "Path to save the NaviX .index file");
        desc.add_options()("M", po::value<int>(&M)->default_value(32), "HNSW M parameter (default: 32)");
        desc.add_options()("efConstruction", po::value<int>(&efConstruction)->default_value(200), "HNSW efConstruction parameter (default: 200)");
        desc.add_options()("num_threads", po::value<int>(&num_threads)->default_value(1), "Number of threads to use for index building"); // 新增参数

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << "Argument error: " << ex.what() << std::endl;
        return -1;
    }

    // 1. 根据 Vamana .bin 格式规范读取二进制数据
    std::ifstream in(base_bin_file, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Cannot open base data file " << base_bin_file << std::endl;
        return -1;
    }

    uint32_t num_points, dim;
    in.read(reinterpret_cast<char*>(&num_points), sizeof(uint32_t));
    in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

    std::cout << "[NaviX Build] Loading " << num_points << " vectors of dimension " << dim << " from " << base_bin_file << std::endl;
    size_t total_elements = static_cast<size_t>(num_points) * dim; // 强制转换为 size_t (通常是64位) 后再相乘
    std::vector<float> data(total_elements);
    // 注意这里也要使用 total_elements
    in.read(reinterpret_cast<char*>(data.data()), total_elements * sizeof(float));
    in.close();

    // 2. 初始化 NaviX (HNSW) 索引
    std::cout << "[NaviX Build] Initializing faiss_navix::IndexHNSWFlat (M=" << M << ", efConstruction=" << efConstruction << ")..." << std::endl;
    faiss_navix::IndexHNSWFlat index(dim, M, faiss_navix::METRIC_L2);
    index.hnsw.efConstruction = efConstruction; 

    // 3. 设置 OpenMP 线程数并构建索引
    omp_set_num_threads(num_threads);
    std::cout << "[NaviX Build] Adding data points to the index using " << num_threads << " threads..." << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    index.add(num_points, data.data());
    auto end_time = std::chrono::high_resolution_clock::now();
    
    double build_time_s = std::chrono::duration<double>(end_time - start_time).count();
    std::cout << "[NaviX Build] Index built in " << build_time_s << " seconds." << std::endl;

    // 4. 序列化保存索引与生成 .meta 文件
    std::cout << "[NaviX Build] Saving index to " << index_output << "..." << std::endl;
    faiss_navix::write_index(&index, index_output.c_str());

    long index_size_bytes = get_file_size(index_output);
    std::ofstream meta_file(index_output + ".meta");
    meta_file << "build_time_s:" << build_time_s << "\n";
    meta_file << "total_size_bytes:" << index_size_bytes << "\n";
    meta_file.close();

    std::cout << "[NaviX Build] Success! Index size: " << (double)index_size_bytes / (1024.0 * 1024.0) << " MB." << std::endl;

    return 0;
}