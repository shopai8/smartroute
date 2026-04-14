import os
import numpy as np
import pandas as pd
import struct


# =========================
# 读取 FAISS bin 文件
# =========================
def read_faiss_bin(file_path):
    with open(file_path, 'rb') as f:
        n, d = struct.unpack('ii', f.read(8))
        data = np.fromfile(f, dtype=np.float32)
        vectors = data.reshape(n, d)
    return vectors


# =========================
# 写 FAISS bin 文件
# =========================
def write_faiss_bin(file_path, vectors):
    n, d = vectors.shape
    with open(file_path, 'wb') as f:
        f.write(struct.pack('ii', n, d))
        vectors.astype(np.float32).tofile(f)


# =========================
# 读取 fvecs
# =========================
def read_fvecs(file_path):
    vectors = []
    with open(file_path, 'rb') as f:
        while True:
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim = struct.unpack('i', dim_bytes)[0]
            vec = np.fromfile(f, dtype=np.float32, count=dim)
            vectors.append(vec)
    return np.array(vectors)


# =========================
# 写 fvecs
# =========================
def write_fvecs(file_path, vectors):
    with open(file_path, 'wb') as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack('i', dim))
            vec.astype(np.float32).tofile(f)


# =========================
# 主处理函数
# =========================
def process(
    bin_path,
    fvecs_path,
    label_path,
    output_bin,
    output_fvecs,
    output_txt,
    repeat_times=100,
    csv_path=None,
    output_algo_csv=None,
    algo_choices=[0, 2, 5],
    algo_weights=None
):
    # 1. 数据读取
    bin_vectors = read_faiss_bin(bin_path)
    fvecs_vectors = read_fvecs(fvecs_path)

    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f]

    # 校验
    assert len(bin_vectors) == len(labels) == len(fvecs_vectors), "数据长度不一致"

    n_queries = len(bin_vectors)
    print(f"总查询数: {n_queries}")

    # 2. 获取算法选择
    if csv_path is not None:
        # 从CSV读取
        print(f"从CSV读取算法选择: {csv_path}")
        df = pd.read_csv(csv_path)[['QueryID', 'Algo_Choice']]
    else:
        # 随机生成算法选择
        print("随机生成算法选择")
        if algo_weights is None:
            algo_weights = [1/len(algo_choices)] * len(algo_choices)
        
        random_algos = np.random.choice(algo_choices, size=n_queries, p=algo_weights)
        
        # 统计分布
        from collections import Counter
        algo_counts = Counter(random_algos)
        for algo, count in sorted(algo_counts.items()):
            print(f"Algo_Choice={algo}: {count} 个 ({count/n_queries*100:.1f}%)")
        
        # 创建DataFrame用于分组
        df = pd.DataFrame({
            'QueryID': range(n_queries),
            'Algo_Choice': random_algos
        })
        
        # 保存原始算法选择到CSV
        if output_algo_csv is not None:
            os.makedirs(os.path.dirname(output_algo_csv), exist_ok=True)
            print(f"保存原始算法选择到: {output_algo_csv}")
            df.to_csv(output_algo_csv, index=False)
    
    # 3. 按算法分组处理
    grouped = df.groupby('Algo_Choice')
    
    all_bin = []
    all_fvecs = []
    all_labels = []
    all_algos = []  # 记录重复后的算法选择
    
    for algo, group in grouped:
        print(f"处理 Algo_Choice={algo}, 数量={len(group)}")
        
        for _, row in group.iterrows():
            qid = int(row['QueryID'])
            
            vec_bin = bin_vectors[qid]
            vec_fvecs = fvecs_vectors[qid]
            label = labels[qid]
            
            for _ in range(repeat_times):
                all_bin.append(vec_bin)
                all_fvecs.append(vec_fvecs)
                all_labels.append(label)
                all_algos.append(algo)

    # 转 numpy
    all_bin = np.array(all_bin)
    all_fvecs = np.array(all_fvecs)

    # （可选）shuffle
    # indices = np.random.permutation(len(all_bin))
    # all_bin = all_bin[indices]
    # all_fvecs = all_fvecs[indices]
    # all_labels = [all_labels[i] for i in indices]
    # if 'all_algos' in locals():
    #     all_algos = [all_algos[i] for i in indices]

    # 4. 写文件
    os.makedirs(os.path.dirname(output_bin), exist_ok=True)
    os.makedirs(os.path.dirname(output_fvecs), exist_ok=True)
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    write_faiss_bin(output_bin, all_bin)
    write_fvecs(output_fvecs, all_fvecs)

    with open(output_txt, 'w') as f:
        for l in all_labels:
            f.write(l + '\n')
    
    # 保存重复后的算法选择
    if 'all_algos' in locals() and output_algo_csv is not None:
        os.makedirs(os.path.dirname(output_algo_csv), exist_ok=True)
        output_algo_repeat_csv = output_algo_csv.replace('.csv', '_repeat.csv')
        print(f"保存重复后的算法选择到: {output_algo_repeat_csv}")
        algo_repeat_df = pd.DataFrame({
            'Index': range(len(all_algos)),
            'Algo_Choice': all_algos
        })
        algo_repeat_df.to_csv(output_algo_repeat_csv, index=False)

    print(f"\n✅ 完成！总样本数: {len(all_bin)}")

# =========================
# 整体重复（新功能）
# =========================
def process_full_repeat(
    bin_path,
    fvecs_path,
    label_path,
    output_bin,
    output_fvecs,
    output_txt,
    repeat_times=100,
    csv_path=None,
    output_algo_csv=None,
    algo_choices=[0, 2, 5],
    algo_weights=None
):
    print("开始整体重复模式...")

    # 1. 读取原始数据
    bin_vectors = read_faiss_bin(bin_path)
    fvecs_vectors = read_fvecs(fvecs_path)

    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f]

    # 校验
    assert len(bin_vectors) == len(labels) == len(fvecs_vectors), "数据长度不一致"

    print(f"原始数据量: {len(bin_vectors)}")

    if csv_path is not None:
        # 从CSV读取算法选择
        print(f"从CSV读取算法选择: {csv_path}")
        df = pd.read_csv(csv_path)[['QueryID', 'Algo_Choice']]
        
        # 整体重复所有查询
        all_bin = np.tile(bin_vectors, (repeat_times, 1))
        all_fvecs = np.tile(fvecs_vectors, (repeat_times, 1))
        all_labels = labels * repeat_times
        all_algos = df['Algo_Choice'].tolist() * repeat_times

        print(f"重复后数据量: {len(all_bin)}")
        
        # 保存重复后的算法选择
        if output_algo_csv is not None:
            os.makedirs(os.path.dirname(output_algo_csv), exist_ok=True)
            output_algo_repeat_csv = output_algo_csv.replace('.csv', '_repeat.csv')
            print(f"保存重复后的算法选择到: {output_algo_repeat_csv}")
            algo_repeat_df = pd.DataFrame({
                'Index': range(len(all_algos)),
                'Algo_Choice': all_algos
            })
            algo_repeat_df.to_csv(output_algo_repeat_csv, index=False)
    else:
        # 整体重复
        all_bin = np.tile(bin_vectors, (repeat_times, 1))
        all_fvecs = np.tile(fvecs_vectors, (repeat_times, 1))
        all_labels = labels * repeat_times

        print(f"重复后数据量: {len(all_bin)}")

    # （可选）shuffle
    # indices = np.random.permutation(len(all_bin))
    # all_bin = all_bin[indices]
    # all_fvecs = all_fvecs[indices]
    # all_labels = [all_labels[i] for i in indices]

    # 3. 写入
    write_faiss_bin(output_bin, all_bin)
    write_fvecs(output_fvecs, all_fvecs)

    with open(output_txt, 'w') as f:
        for l in all_labels:
            f.write(l + '\n')

    print("✅ 整体重复完成！")

# =========================
# 使用示例
# =========================
if __name__ == "__main__":
    repeat_times= 100
    algo_weights = [0.5, 0.33, 0.17]  # 示例权重，调整为你需要的比例
    # 方式1：随机生成算法选择（csv_path=None）
    # process(
    #     bin_path="/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-123456789/Reviews_query.bin",
    #     fvecs_path="/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-123456789/Reviews_query.fvecs",
    #     label_path="/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-123456789/Reviews_query_labels.txt",
    #     output_bin=f"/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-repeat_{repeat_times}_{algo_weights[0]}/Reviews_query.bin",
    #     output_fvecs=f"/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-repeat_{repeat_times}_{algo_weights[0]}/Reviews_query.fvecs",
    #     output_txt=f"/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-repeat_{repeat_times}_{algo_weights[0]}/Reviews_query_labels.txt",
    #     repeat_times=repeat_times,
    #     csv_path=None,  # 不读取CSV，随机生成
    #     output_algo_csv=f"/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-repeat_{repeat_times}_{algo_weights[0]}/algo_choice.csv",  # 保存算法选择
    #     algo_choices=[0, 2, 5],
    #     algo_weights=algo_weights  # 使用自定义比例
    # )
    
    # 方式2：从CSV读取算法选择（取消注释以使用）
    # process(
    #     csv_path="/mnt/disk1/syh/ljk/FilterVector/FilterVectorResults/Reviews/Results/FastSmartRoute+/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls10-Le10-Lp10_efsS100-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
    #     bin_path="/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-123456789/Reviews_query.bin",
    #     fvecs_path="/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-123456789/Reviews_query.fvecs",
    #     label_path="/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-123456789/Reviews_query_labels.txt",
    #     output_bin=f"/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-repeat_{repeat_times}/Reviews_query.bin",
    #     output_fvecs=f"/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-repeat_{repeat_times}/Reviews_query.fvecs",
    #     output_txt=f"/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-repeat_{repeat_times}/Reviews_query_labels.txt",
    #     repeat_times=repeat_times
    # )
    process_full_repeat(
        bin_path="/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-123456789/Reviews_query.bin",
        fvecs_path="/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-123456789/Reviews_query.fvecs",
        label_path="/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-123456789/Reviews_query_labels.txt",
        output_bin=f"/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-repeat_{repeat_times}_2/Reviews_query.bin",
        output_fvecs=f"/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-repeat_{repeat_times}_2/Reviews_query.fvecs",
        output_txt=f"/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-repeat_{repeat_times}_2/Reviews_query_labels.txt",
        repeat_times=repeat_times,
        csv_path=f"/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-repeat_{repeat_times}_{algo_weights[0]}/algo_choice.csv",  # 读取CSV，直接整体重复
        output_algo_csv=f"/mnt/disk1/syh/ljk/FilterVector/FilterVectorData/Reviews/query_select_imp_A_B_C-sub-base-repeat_{repeat_times}_2/algo_choice.csv"  # 保存算法选择
    )