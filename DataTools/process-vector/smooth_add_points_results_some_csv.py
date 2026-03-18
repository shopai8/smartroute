import pandas as pd
import numpy as np
import os
import shutil

# ================= 配置区域 =================

# 1. 手动填写需要处理的 CSV 文件路径列表
TARGET_FILES = [
    r"/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base_one_query_1_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss1000-efsf1000-lt5000_K10_th100]/results/search_time_summary.csv",
    r"/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base_one_query_2_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss1000-efsf1000-lt5000_K10_th100]/results/search_time_summary.csv",
    r"/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base_one_query_1_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/search_time_summary.csv",
    r"/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base_one_query_2_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/search_time_summary.csv",
    # r"C:\Users\YourName\Documents\another_file.csv", # Windows路径示例
]

# 2. 插值参数配置
# 如果想针对某个文件使用特殊配置，可以直接在这里修改，或者在代码逻辑中硬编码
DEFAULT_CONFIG = {
    "GAP_RATIO": 2.0,            
    "TIME_GAP_RATIO": 1.0,       
    "MAX_INTERPOLATION_POINTS": 10,
    
    # 插值指数 (Power): 1.0 为线性插值，<1 为凸，>1 为凹
    "POWER_TIME": 1.0,
    "POWER_RECALL": 1.0
}

# 备份文件的后缀
BACKUP_SUFFIX = "_ori.csv"

# ===========================================

def force_monotonicity(df):
    """
    实现需求：
    1. 根据 Average_Efs (或 Lsearch) 排序行。
    2. 强制 Average_Time_ms 升序。
    3. 强制 Average_Recall 升序。
    """
    # 1. 确定排序关键字
    sort_key = 'Average_Efs'
    if 'Average_Efs' not in df.columns and 'Lsearch' in df.columns:
        sort_key = 'Lsearch'
    
    if sort_key in df.columns:
        df = df.sort_values(by=sort_key).reset_index(drop=True)
    else:
        print("  [警告] 未找到 Average_Efs 或 Lsearch 列，跳过排序步骤。")
    
    # 2. 强制 Time 单调递增
    time_col = 'Average_Time_ms' if 'Average_Time_ms' in df.columns else 'Time_ms'
    if time_col in df.columns:
        raw_times = df[time_col].values
        sorted_times = np.sort(raw_times)
        df[time_col] = sorted_times

    # 3. 强制 Recall 单调递增
    recall_col = None
    if 'Average_Recall' in df.columns:
        recall_col = 'Average_Recall'
    elif 'Recall' in df.columns:
        recall_col = 'Recall'
        
    if recall_col:
        raw_recalls = df[recall_col].values
        sorted_recalls = np.sort(raw_recalls)
        df[recall_col] = sorted_recalls
        
    return df

def get_standard_step(series):
    """计算标准步长"""
    diffs = np.diff(series)
    valid_diffs = diffs[diffs > 1e-6] 
    if len(valid_diffs) == 0: return 1.0 
    step = np.percentile(valid_diffs, 25)
    return max(step, 1e-3)

def interpolate_data(df, config):
    """执行插值，支持非线性 Power 参数"""
    gap_ratio = config.get("GAP_RATIO", 2.0)
    time_gap_ratio = config.get("TIME_GAP_RATIO", 5.0)
    max_points = config.get("MAX_INTERPOLATION_POINTS", 100)
    power_time = config.get("POWER_TIME", 1.0)
    power_recall = config.get("POWER_RECALL", 1.0)

    # 确定关键列名
    efs_col = 'Average_Efs' if 'Average_Efs' in df.columns else 'Lsearch'
    time_col = 'Average_Time_ms' if 'Average_Time_ms' in df.columns else 'Time_ms'
    
    # 如果找不到关键列，直接返回原数据
    if efs_col not in df.columns or time_col not in df.columns:
        return df

    final_rows = []
    
    std_step_efs = get_standard_step(df[efs_col])
    std_step_time = get_standard_step(df[time_col])
    
    cols = df.columns.tolist()
    
    for i in range(len(df) - 1):
        row_curr = df.iloc[i]
        row_next = df.iloc[i+1]
        final_rows.append(row_curr) 
        
        gap_efs = row_next[efs_col] - row_curr[efs_col]
        gap_time = row_next[time_col] - row_curr[time_col]
        
        # 判定断层
        is_efs_gap = gap_efs > (std_step_efs * gap_ratio)
        is_time_gap = False
        if gap_time > 0:
            is_time_gap = gap_time > (std_step_time * time_gap_ratio)
            
        if is_efs_gap or is_time_gap:
            points_by_efs = int(gap_efs / std_step_efs) - 1 if std_step_efs > 0 else 0
            points_by_time = int(gap_time / std_step_time) - 1 if std_step_time > 0 else 0
            num_points = min(max(points_by_efs, points_by_time), max_points)
            
            if num_points > 0:
                t_linear = np.linspace(0, 1, num_points + 2)[1:-1]
                
                for t in t_linear:
                    new_row = row_curr.copy()
                    for col in cols:
                        if pd.api.types.is_numeric_dtype(df[col]):
                            val_start = row_curr[col]
                            val_end = row_next[col]
                            
                            # 应用非线性 Power
                            if col == time_col:
                                frac = t ** power_time
                            elif col in ['Average_Recall', 'Recall']:
                                frac = t ** power_recall
                            else:
                                frac = t
                            
                            interp_val = val_start + (val_end - val_start) * frac
                            
                            # 精度控制
                            if col in ['Average_Efs', 'Lsearch', 'repeat', 'efs']:
                                new_row[col] = int(interp_val)
                            elif col == time_col:
                                new_row[col] = round(interp_val, 3)
                            elif col in ['Average_Recall', 'Recall']:
                                new_row[col] = round(interp_val, 6)
                            else:
                                new_row[col] = interp_val
                                
                    final_rows.append(new_row)
                    
    final_rows.append(df.iloc[-1])
    return pd.DataFrame(final_rows, columns=cols)

def process_single_file(file_path):
    """处理单个文件：备份 -> 清洗 -> 插值 -> 保存"""
    
    if not os.path.exists(file_path):
        print(f"[跳过] 文件不存在: {file_path}")
        return

    # 构造备份路径: filename.csv -> filename_ori.csv
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    base_name, ext = os.path.splitext(file_name)
    backup_name = base_name + BACKUP_SUFFIX
    backup_path = os.path.join(file_dir, backup_name)
    
    # 1. 备份逻辑
    # 如果备份文件已存在，则读取备份文件（原始数据），确保幂等性
    if not os.path.exists(backup_path):
        try:
            shutil.move(file_path, backup_path)
            print(f"  [备份] 生成原始数据备份: {backup_name}")
        except Exception as e:
            print(f"  [错误] 备份失败: {e}")
            return
    else:
        print(f"  [提示] 备份已存在，将基于 {backup_name} 重新生成")

    # 2. 处理逻辑
    try:
        df = pd.read_csv(backup_path)
        if df.empty: 
            print("  [警告] 文件为空")
            return
        
        # 应用配置 (如果需要针对不同文件应用不同配置，可以在这里写 if 逻辑)
        config = DEFAULT_CONFIG
        
        # 步骤 A: 强制清洗 (Time & Recall 均单调)
        df_clean = force_monotonicity(df)
        
        # 步骤 B: 执行插值平滑
        df_smooth = interpolate_data(df_clean, config)
        
        # 保存回原文件名
        df_smooth.to_csv(file_path, index=False)
        print(f"  [成功] 已处理并保存: {file_name}")
        
    except Exception as e:
        print(f"  [错误] 处理失败: {e}")
        # 如果处理中途崩溃且原文件不存在(已被move)，尝试恢复
        if not os.path.exists(file_path) and os.path.exists(backup_path):
            shutil.copy2(backup_path, file_path)
            print("  [恢复] 已恢复原文件")

if __name__ == "__main__":
    print("开始手动处理指定文件 (强制单调性 + 智能插值)...")
    print(f"待处理文件数量: {len(TARGET_FILES)}\n")
    
    for i, fpath in enumerate(TARGET_FILES):
        # 清理路径字符串
        fpath = fpath.strip()
        print(f"[{i+1}/{len(TARGET_FILES)}] 正在处理: {fpath}")
        process_single_file(fpath)
        print("-" * 50)
            
    print("\n所有指定文件处理完成。")