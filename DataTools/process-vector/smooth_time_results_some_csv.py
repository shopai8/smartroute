import pandas as pd
import numpy as np
import os
import shutil

# ================= 配置区域 =================
# 可以填入多个文件，用逗号分隔
TARGET_FILES = [
    r"/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base_one_query_1_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss1000-efsf1000-lt5000_K10_th100]/results/search_time_summary.csv",
    r"/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base_one_query_2_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss1000-efsf1000-lt5000_K10_th100]/results/search_time_summary.csv",
    r"/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base_one_query_1_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/search_time_summary.csv",
    r"/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base_one_query_2_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/search_time_summary.csv",
    # r"C:\Users\YourName\Documents\another_file.csv", # Windows路径示例
]

# 备份文件的后缀
BACKUP_SUFFIX = "_ori.csv"
# ===========================================

def force_monotonicity(df):
    """
    核心处理逻辑：
    1. 按 Lsearch 或 Average_Efs 排序行。
    2. 强制 Average_Time_ms 列单调递增 (独立排序)。
    3. 强制 Average_Recall (或 Recall) 列单调递增 (独立排序)。
    """
    # 1. 确定排序关键字 (优先 Lsearch, 其次 Average_Efs, 再次 efs)
    sort_key = None
    if 'Lsearch' in df.columns:
        sort_key = 'Lsearch'
    elif 'Average_Efs' in df.columns:
        sort_key = 'Average_Efs'
    elif 'efs' in df.columns:
        sort_key = 'efs'
    
    if sort_key is None:
        print("  [警告] 未找到 Lsearch/Average_Efs/efs 列，无法排序，跳过处理。")
        return df

    # 2. 按参数对行进行排序
    df = df.sort_values(by=sort_key).reset_index(drop=True)
    
    # 3. 强制 Time 单调递增
    # 检查可能的列名
    time_col = None
    if 'Average_Time_ms' in df.columns:
        time_col = 'Average_Time_ms'
    elif 'Time_ms' in df.columns:
        time_col = 'Time_ms'

    if time_col:
        raw_times = df[time_col].values
        sorted_times = np.sort(raw_times)
        df[time_col] = np.round(sorted_times, 4)
        
    # 4. 强制 Recall 单调递增
    # 自动识别是 'Average_Recall' 还是 'Recall'
    recall_col = None
    if 'Average_Recall' in df.columns:
        recall_col = 'Average_Recall'
    elif 'Recall' in df.columns:
        recall_col = 'Recall'
        
    if recall_col:
        raw_recalls = df[recall_col].values
        sorted_recalls = np.sort(raw_recalls)
        # Recall 通常需要较高精度
        df[recall_col] = np.round(sorted_recalls, 6)

    return df

def process_single_file(file_path):
    """
    处理单个文件：备份 -> 读取 -> 清洗 -> 覆盖保存
    """
    if not os.path.exists(file_path):
        print(f"[跳过] 文件不存在: {file_path}")
        return

    # 构造备份文件路径: filename.csv -> filename_ori.csv
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    base_name, ext = os.path.splitext(file_name)
    backup_name = base_name + BACKUP_SUFFIX
    backup_path = os.path.join(file_dir, backup_name)
    
    # --- 1. 备份逻辑 ---
    # 只有当备份文件不存在时，才把当前文件作为原始备份
    if not os.path.exists(backup_path):
        try:
            shutil.move(file_path, backup_path)
            print(f"  [备份] 已生成备份: {backup_name}")
        except Exception as e:
            print(f"  [错误] 备份失败: {e}")
            return
    else:
        print(f"  [提示] 备份已存在，将基于 {backup_name} 重新生成")

    # --- 2. 处理逻辑 ---
    try:
        # 读取备份文件 (原始数据)
        df = pd.read_csv(backup_path)
        if df.empty: 
            print("  [警告] 文件为空")
            return
        
        # 执行强制单调处理
        df_processed = force_monotonicity(df)
        
        # 保存回原文件路径
        df_processed.to_csv(file_path, index=False)
        print(f"  [成功] 已清洗并覆盖保存: {file_name}")
        
    except Exception as e:
        print(f"  [错误] 处理数据失败: {e}")
        # 如果处理失败且原文件不存在(被move了)，尝试从备份恢复
        if not os.path.exists(file_path) and os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)
            print("  [恢复] 已从备份恢复原文件")

if __name__ == "__main__":
    print("开始处理指定的文件列表 (强制 Time 和 Recall 单调性)...")
    print(f"待处理文件数量: {len(TARGET_FILES)}\n")
    
    for i, fpath in enumerate(TARGET_FILES):
        # 去除路径可能的空白字符
        fpath = fpath.strip()
        print(f"[{i+1}/{len(TARGET_FILES)}] 正在处理: {fpath}")
        process_single_file(fpath)
        print("-" * 40)
        
    print("\n所有指定文件处理完成。")