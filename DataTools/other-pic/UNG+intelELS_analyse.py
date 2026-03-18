# 20260312:计算 (UNG - ELS + IntelELS) / UNG 耗时比值的平均数
import pandas as pd
import numpy as np
import os

def calculate_intel_els_ratio(ung_csv_path: str, sr_csv_path: str) -> dict:
    """
    1.一句话概括函数核心作用：计算 (UNG - ELS + IntelELS) / UNG 耗时比值的平均数并输出中间处理日志。
    
    2.思路说明：
      - 加载 UNG 和 SmartRoute(SR) 的结果 CSV，增加 header=0 或强制类型转换跳过英文字符串表头。
      - 提取核心字段，并通过 query_id (结合 repeat 和 Lsearch) 将两个表格的数据逐行对齐拼接。
      - 从 UNG 提取总耗时和 ELS 耗时；从 SR 提取 IntelELS 的三部分耗时相加。
      - 计算每条查询的耗时比值，过滤掉分母为0或无效的数据后，求平均值。
      
    3.输入参数：
      - ung_csv_path: str，含义（必填）：UNG 基线算法跑出的 query_details_repeatX.csv 文件路径。
      - sr_csv_path: str，含义（必填）：SmartRoute 算法跑出的 query_details_repeatX.csv 文件路径。
      
    4.返回值类型和具体含义：
      - dict: 包含计算结果的字典，包括 'average_ratio' (平均比值), 'valid_queries' (有效查询数量), 'avg_ung_time' (平均UNG总耗时), 'avg_intel_els_time' (平均IntelELS耗时)。
    """
    
    # 依据 search_UNG_index.cpp 中的输出顺序定义列名
    columns = [
        "repeat", "Lsearch", "acorn_efs_used", "query_id", 
        "time_ms", "search_time_ms", "core_search_time_ms", "recall", 
        "is_idea1_used", "is_idea2_used", "num_distance_calcs", 
        "num_nodes_visited", "get_min_super_sets_time_ms", 
        "idea1_selector_pred_time_ms", "idea2_selector_pred_time_ms", 
        "idea1_flag_time_ms", "idea2_flag_time_ms", "bitmap_time_ms", 
        "query_length", "candidate_set_size", "num_entry_points"
    ]
    
    print(f"\n[INFO] 开始处理数据...")
    print(f"  -> UNG CSV路径: {ung_csv_path}")
    print(f"  -> SR  CSV路径: {sr_csv_path}")

    # 1. 读取 CSV 文件 (增加 header=0 跳过原文件表头，低内存模式设为False)
    try:
        df_ung = pd.read_csv(ung_csv_path, names=columns, header=0, low_memory=False)
        df_sr = pd.read_csv(sr_csv_path, names=columns, header=0, low_memory=False)
        
        # [核心修复] 强制将时间列和主键转换为数值类型（如果遇到字符串表头残留则转为NaN）
        time_cols = ["time_ms", "get_min_super_sets_time_ms", "idea1_flag_time_ms", "idea2_flag_time_ms"]
        key_cols = ["repeat", "Lsearch", "query_id"]
        for col in time_cols + key_cols:
            df_ung[col] = pd.to_numeric(df_ung[col], errors='coerce')
            df_sr[col] = pd.to_numeric(df_sr[col], errors='coerce')
            
        # 丢弃因转换表头而产生的 NaN 无效行
        df_ung = df_ung.dropna(subset=["query_id"])
        df_sr = df_sr.dropna(subset=["query_id"])
            
        print(f"[INFO] 成功读取: UNG包含 {len(df_ung)} 条记录, SmartRoute包含 {len(df_sr)} 条记录。")
    except FileNotFoundError as e:
        print(f"[ERROR] 找不到文件: {e}")
        return {}

    # 2. 根据查询的主键合并两个数据集
    df_merged = pd.merge(
        df_ung, 
        df_sr, 
        on=["repeat", "Lsearch", "query_id"], 
        suffixes=("_ung", "_sr")
    )
    
    if df_merged.empty:
        print("[WARN] 未能通过 (repeat, Lsearch, query_id) 对齐数据，请检查两个 CSV 是否匹配。")
        return {}
    else:
        print(f"[INFO] 数据对齐完成: 成功匹配到 {len(df_merged)} 条相同的查询。")

    # 3. 提取我们需要的时间列
    T_ung = df_merged["time_ms_ung"]
    T_ung_els = df_merged["get_min_super_sets_time_ms_ung"]
    T_intel_els = (
        df_merged["get_min_super_sets_time_ms_sr"] + 
        df_merged["idea1_flag_time_ms_sr"] + 
        df_merged["idea2_flag_time_ms_sr"]
    )
    
    # 打印前三条耗时数据供肉眼抽查
    print(f"[INFO] 耗时数据抽查 (前3条):")
    print(f"  -> 原 UNG 总耗时      : {[round(x, 4) for x in T_ung.head(3).tolist()]} ms")
    print(f"  -> 原 UNG ELS 耗时    : {[round(x, 4) for x in T_ung_els.head(3).tolist()]} ms")
    print(f"  -> SR 提取的 IntelELS : {[round(x, 4) for x in T_intel_els.head(3).tolist()]} ms")

    # 4. 计算理论上的新总耗时 和 比值
    T_new_total = T_ung - T_ung_els + T_intel_els
    
    epsilon = 1e-9
    ratios = T_new_total / (T_ung + epsilon)
    
    valid_ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
    dropped_count = len(ratios) - len(valid_ratios)
    if dropped_count > 0:
        print(f"[WARN] 清理了 {dropped_count} 条无效比值数据。")

    # 计算 Global Ratio (总耗时之比)
    global_ratio = T_new_total.sum() / T_ung.sum()

    # 5. 组装结果
    result = {
        "average_ratio": valid_ratios.mean(),
        "global_ratio": global_ratio,
        "valid_queries": len(valid_ratios),
        "avg_ung_time": T_ung.mean(),
        "avg_ung_els_time": T_ung_els.mean(),      # 新增：原UNG中的ELS平均耗时
        "avg_intel_els_time": T_intel_els.mean()
    }
    
    
    print(f"[INFO] 计算结束！计算使用的有效记录数: {result['valid_queries']}\n")
    return result

# ==========================================
# 批量处理多个数据集
# ==========================================
if __name__ == "__main__":
    datasets = {
        "Amazon": {
            "ung": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Amazon/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
            "sr":  "/home/fengxiaoyao/FilterVector/FilterVectorResults/Amazon/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN602453_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv"
        },
        "BookReviews": {
            "ung": "/home/fengxiaoyao/FilterVector/FilterVectorResults/BookReviews/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_select_imp_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls10-Le20000-Lp500_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
            "sr":  "/home/fengxiaoyao/FilterVector/FilterVectorResults/BookReviews/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN2065775_AM32_AMB64_AG80]_GT[GT_query_select_imp_B_C_D-weighted-sub-base-123456789_K10]_Search[Ls10-Le20000-Lp500_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv"
        },
        "Genome": {
            "ung": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Genome/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls10-Le40000-Lp1000_efsS20-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
            "sr": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Genome/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN108077_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls10-Le40000-Lp1000_efsS20-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv"
        },
        "Music": {
            "ung": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
            "sr": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv"
        },
        "Reviews": {
            "ung": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Reviews/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv",
            "sr": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Reviews/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN288065_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss100-efsf100-lt5000_K10_th100]/results/query_details_repeat1.csv"
        },
        "Tiktok": {
            "ung": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Tiktok/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv",
            "sr": "/home/fengxiaoyao/FilterVector/FilterVectorResults/Tiktok/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN2201307_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS500-efss500-efsf500-lt5000_K10_th100]/results/query_details_repeat1.csv"
        },
        "VariousImg": {
            "ung": "/home/fengxiaoyao/FilterVector/FilterVectorResults/VariousImg/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv",
            "sr": "/home/fengxiaoyao/FilterVector/FilterVectorResults/VariousImg/Results/method3/Index[M32_LB100_alpha1.2_C6_EP16_AN758935_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls20-Le40000-Lp1000_efsS50-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv"
        }
    }
    
    # 建立一个汇总列表最后统一打印
    summary_results = []

    for ds_name, paths in datasets.items():
        if os.path.exists(paths["ung"]) and os.path.exists(paths["sr"]):
            res = calculate_intel_els_ratio(paths["ung"], paths["sr"])
            if res:
                summary_results.append((
                    ds_name, 
                    res['global_ratio'],           # 新增
                    res['average_ratio'], 
                    res['valid_queries'], 
                    res['avg_ung_time'], 
                    res['avg_ung_els_time'],       # 新增
                    res['avg_intel_els_time']
                ))
        else:
            print(f"[ERROR] 请确认 {ds_name} 的文件路径有效。无法找到指定文件。\n")
            
    # 打印最终的汇总表格
    print("============================================================ 最终汇总报告 ============================================================")
    print(f"{'Dataset':<12} | {'Global Ratio (Sums)':<20} | {'Avg Ratio (Mean)':<17} | {'Valid Queries':<14} | {'Avg UNG Time':<15} | {'Avg ELS in UNG':<15} | {'Avg IntelELS Time'}")
    print("-" * 134)
    for row in summary_results:
        # row: [0]ds_name, [1]global_ratio, [2]avg_ratio, [3]queries, [4]ung_time, [5]ung_els, [6]intel_els
        print(f"{row[0]:<12} | {row[1]:<20.4f} | {row[2]:<17.4f} | {row[3]:<14} | {row[4]:>12.4f} ms | {row[5]:>12.4f} ms | {row[6]:>13.4f} ms")
    print("======================================================================================================================================")