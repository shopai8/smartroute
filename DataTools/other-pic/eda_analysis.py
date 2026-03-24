import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 配置区域 (Configuration)
# ==========================================
BASE_DIR = "/home/fengxiaoyao/FilterVector/FilterVectorResults"

# "Amazon","BookReviews","Genome","Music","Reviews", "Tiktok","VariousImg","Laion"
DATASETS = ["Amazon","BookReviews","Genome","Music","Reviews", "Tiktok","VariousImg","Laion"]

# 算法名称到文件夹名称的映射
ALGO_FOLDERS = {
    'UNG-nTfalse': 'UNG-nTfalse',
    'UNG-nTtrue': 'UNG-nTtrue',
    'ACORN-gamma': 'ACORN-gamma',
    'ACORN-improved': 'ACORN-gamma-improved',
    'NaviX': 'NaviX-ACORN',    
    'pre-filter': 'pre-filter'
}

# 目标召回率
MIN_RECALL = 0.90

# 统一的输出图片/CSV根目录
GLOBAL_OUTPUT_DIR = os.path.join(BASE_DIR, "EDA_Plots_UNGnTtrue")
os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)

# ==========================================
# Step 1: 数据加载 (主表与特征表)
# ==========================================
def load_data(dataset_name):
    print(f"\n=============================================")
    print(f"[*] 开始处理数据集: {dataset_name}")
    print(f"=============================================")
    df_list = []
    
    for algo_name, folder_name in ALGO_FOLDERS.items():
        search_pattern = os.path.join(BASE_DIR, dataset_name, "Results", folder_name, "Index*", "results", "query_details_repeat1.csv")
        all_matched_files = glob.glob(search_pattern)
        # --- 过滤掉路径中包含 "select_imp" 的文件 ---
        matched_files = [f for f in all_matched_files if "select_imp" not in f]
        
        if not matched_files:
            print(f"  [Warning] 未找到 {algo_name} 的结果文件! 匹配路径: {search_pattern}")
            continue
            
        file_path = matched_files[0] 
        print(f"  [√] 加载 {algo_name}: {os.path.basename(os.path.dirname(os.path.dirname(file_path)))}")
        
        df = pd.read_csv(file_path)
        df['Algorithm'] = algo_name
        df_list.append(df)
        
    if not df_list:
        return pd.DataFrame()
        
    df_long = pd.concat(df_list, ignore_index=True)
    
    print("[*] 正在统一时间度量标准 (L1特征时间 / L2特征时间 / 端到端绝对时间)...")
    
    # 预处理：确保涉及计算的列存在且无空值
    if 'MinSupersetT_ms' not in df_long.columns:
        df_long['MinSupersetT_ms'] = 0.0
    else:
        df_long['MinSupersetT_ms'] = df_long['MinSupersetT_ms'].fillna(0.0)
        
    def calculate_l1_time(row):
        """L1 视角所需时间特征：只有选 UNG 时需要外加计算 ELS 的耗时"""
        if 'UNG' in row['Algorithm']:
            return row['search_time_ms'] + row['MinSupersetT_ms']
        return row['search_time_ms']
        
    def calculate_l2_time(row):
        """L2 视角所需时间特征：所有前置开销均视为已发生"""
        return row['search_time_ms']

    # L1 和 L2 时间维持纯加法计算，供下游模型脚本进行动态打标
    df_long['L1_Time_ms'] = df_long.apply(calculate_l1_time, axis=1)
    df_long['L2_Time_ms'] = df_long.apply(calculate_l2_time, axis=1)
    
    # 真正的端到端绝对时间，直接使用 C++ 输出的总 time_ms (最准确，供纯 EDA 数据分析用)
    if 'Time_ms' in df_long.columns:
        df_long['True_EndToEnd_Time_ms'] = df_long['Time_ms']
    else:
        print("  [Warning] 未检测到 time_ms 列，请检查 C++ 输出！暂用 search_time_ms 替代。")
        df_long['True_EndToEnd_Time_ms'] = df_long['search_time_ms']
    
    return df_long

def load_features(dataset_name):
    """自动寻找并加载 query_features.csv"""
    search_pattern = os.path.join(BASE_DIR, dataset_name, "Results", "*", "Index*", "results", "query_features.csv")
    matched_files = glob.glob(search_pattern)
    if matched_files:
        print(f"  [√] 成功找到附加特征文件: query_features.csv")
        return pd.read_csv(matched_files[0])
    else:
        print(f"  [Warning] 未找到附加特征文件: query_features.csv (将只使用 details 表中的基础特征)")
        return pd.DataFrame()

# ==========================================
# Step 2: 提炼最优表现与多源特征对齐
# ==========================================
def preprocess_and_align(df_long, dataset_name):
    print(f"[*] 正在拼装特征宽表 (只输出客观数据，不做任何打标逻辑)...")
    
    valid_mask = df_long['Recall'] >= MIN_RECALL
    df_valid = df_long[valid_mask]
    
    if not df_valid.empty:
        idx_valid = df_valid.groupby(['Algorithm', 'QueryID'])['True_EndToEnd_Time_ms'].idxmin()
        best_valid = df_valid.loc[idx_valid]
    else:
        best_valid = pd.DataFrame()
        
    processed_keys = best_valid.set_index(['Algorithm', 'QueryID']).index if not best_valid.empty else []
    df_invalid = df_long[~df_long.set_index(['Algorithm', 'QueryID']).index.isin(processed_keys)]
    
    if not df_invalid.empty:
        idx_invalid = df_invalid.groupby(['Algorithm', 'QueryID'])['Recall'].idxmax()
        best_invalid = df_invalid.loc[idx_invalid]
    else:
        best_invalid = pd.DataFrame()
        
    df_best = pd.concat([best_valid, best_invalid])
    
    # 提取 details 表中的全局共有基础特征
    feature_source = df_best[df_best['Algorithm'].isin(['NaviX', 'pre-filter'])]
    if feature_source.empty:
        feature_source = df_best
        
    base_features_cols = ['QueryID', 'QuerySize', 'CandSize', 'ExactCandSize', 'GlobalPpass', 'FeatureT_ms']
    existing_base_features = [col for col in base_features_cols if col in feature_source.columns]
    features_df = feature_source[existing_base_features].drop_duplicates(subset=['QueryID']).set_index('QueryID')
    
    # 宽表化算法耗时与 Recall (客观记录 3 套时间)
    df_wide = df_best.pivot_table(
        index='QueryID', 
        columns='Algorithm', 
        values=['Recall', 'L1_Time_ms', 'L2_Time_ms', 'True_EndToEnd_Time_ms', 'MinSupersetT_ms'],
        aggfunc='first'
    )
    df_wide.columns = [f"{col[0]}_{col[1]}" for col in df_wide.columns]
    
    # 合并基础特征
    df_final = df_wide.join(features_df).reset_index()
    
    # === 加载并合并额外的 query_features.csv ===
    df_extra_features = load_features(dataset_name)
    if not df_extra_features.empty:
        overlap_cols = [col for col in df_extra_features.columns if col in df_final.columns and col != 'QueryID']
        df_extra_clean = df_extra_features.drop(columns=overlap_cols)
        df_final = pd.merge(df_final, df_extra_clean, on='QueryID', how='left')
        print("  [√] 已将附加图拓扑特征成功合并至宽表！")
        
    return df_final

# ==========================================
# Step 3: 数据分析、绘图及数据导出
# ==========================================
def perform_eda(df_final, output_dir):
    """
    注意：此处画图及寻找 Fastest_Algo 纯粹是为了 EDA 数据可视化。
    不生成任何用于 ML 训练的标签 (不会把计算的最优写入宽表 CSV 中)。
    画图统一使用用户的直观端到端感受时间 (True_EndToEnd_Time_ms)。
    """
    algorithms = list(ALGO_FOLDERS.keys())
    
    # -----------------------------------------
    # 【图 1】：召回达标率 (Recall Success Rate)
    # -----------------------------------------
    success_rates = {}
    for algo in algorithms:
        if f'Recall_{algo}' in df_final.columns:
            rate = (df_final[f'Recall_{algo}'] >= MIN_RECALL).mean() * 100
            success_rates[algo] = rate
            
    df_p1 = pd.DataFrame(list(success_rates.items()), columns=['Algorithm', f'Success_Rate_Pct_Recall_{MIN_RECALL}'])
    df_p1.to_csv(os.path.join(output_dir, "plot1_data_recall_success_rate.csv"), index=False)
            
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(x=list(success_rates.keys()), y=list(success_rates.values()), 
                      hue=list(success_rates.keys()), palette='viridis', legend=False)
    plt.title(f'Recall Success Rate (>= {MIN_RECALL}) by Algorithm')
    plt.ylabel('Success Rate (%)')
    plt.xticks(rotation=45)
    plt.ylim(0, 110)
    for i, v in enumerate(success_rates.values()):
        ax1.text(i, v + 1.5, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_recall_success_rate.png"), dpi=300)
    plt.close()

    # -----------------------------------------
    # 【图 2】：达标中位数绝对耗时 (Median Search Time)
    # -----------------------------------------
    median_times = {}
    for algo in algorithms:
        recall_col = f'Recall_{algo}'
        time_col = f'True_EndToEnd_Time_ms_{algo}'
        if time_col in df_final.columns:
            mask = df_final[recall_col] >= MIN_RECALL
            if mask.sum() > 0:
                median_times[algo] = df_final.loc[mask, time_col].median()
            else:
                median_times[algo] = 0
                
    df_p2 = pd.DataFrame(list(median_times.items()), columns=['Algorithm', 'Median_True_EndToEnd_Time_ms'])
    df_p2.to_csv(os.path.join(output_dir, "plot2_data_median_search_time.csv"), index=False)
                
    plt.figure(figsize=(10, 6))
    ax2 = sns.barplot(x=list(median_times.keys()), y=list(median_times.values()), 
                      hue=list(median_times.keys()), palette='magma', legend=False)
    plt.title(f'Median True End-to-End Time (ms) for Successful Queries')
    plt.ylabel('Time (ms)')
    plt.xticks(rotation=45)
    for i, v in enumerate(median_times.values()):
        if v > 0:
            ax2.text(i, v + (v*0.02), f"{v:.2f}ms", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_median_search_time.png"), dpi=300)
    plt.close()

    # -----------------------------------------
    # 辅助计算：找出全局基于端到端时间的最快算法 (仅用于画后续分析图)
    # -----------------------------------------
    time_cols = []
    temp_df = df_final.copy()
    for algo in algorithms:
        time_col = f'True_EndToEnd_Time_ms_{algo}'
        recall_col = f'Recall_{algo}'
        valid_time_col = f'Valid_Time_{algo}'
        if time_col in temp_df.columns:
            temp_df[valid_time_col] = np.where(temp_df[recall_col] >= MIN_RECALL, temp_df[time_col], np.inf)
            time_cols.append(valid_time_col)
            
    temp_df['Best_Time'] = temp_df[time_cols].min(axis=1)
    temp_df['Fastest_Algo'] = temp_df[time_cols].idxmin(axis=1).str.replace('Valid_Time_', '')
    temp_df.loc[temp_df['Best_Time'] == np.inf, 'Fastest_Algo'] = 'None_Qualified'
    valid_df = temp_df[temp_df['Fastest_Algo'] != 'None_Qualified'].copy()
    
    # -----------------------------------------
    # 【图 3】：端到端全局最快算法占比饼图 (Fastest Algorithm Pie)
    # -----------------------------------------
    algo_counts = valid_df['Fastest_Algo'].value_counts()
    
    df_p3 = algo_counts.reset_index()
    df_p3.columns = ['Algorithm', 'Dominance_Count']
    df_p3['Dominance_Pct'] = (df_p3['Dominance_Count'] / df_p3['Dominance_Count'].sum()) * 100
    df_p3.to_csv(os.path.join(output_dir, "plot3_data_algorithm_dominance.csv"), index=False)
    
    plt.figure(figsize=(10, 8))
    colors = sns.color_palette('Set3')[0:len(algo_counts)]
    plt.pie(algo_counts, labels=algo_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title(f"Fastest End-to-End Algorithm (Recall >= {MIN_RECALL})")
    plt.savefig(os.path.join(output_dir, "03_algorithm_dominance_pie.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # -----------------------------------------
    # 【图 4】：Ppass 区间堆叠图 (Dominance by Ppass Bin)
    # -----------------------------------------
    if not valid_df.empty and 'GlobalPpass' in valid_df.columns:
        try:
            valid_df['Ppass_Bin'] = pd.qcut(valid_df['GlobalPpass'], q=8, duplicates='drop')
        except:
            valid_df['Ppass_Bin'] = pd.cut(valid_df['GlobalPpass'], bins=8)
            
        bin_algo_counts = valid_df.groupby(['Ppass_Bin', 'Fastest_Algo'], observed=False).size().unstack(fill_value=0)
        bin_algo_ratio = bin_algo_counts.div(bin_algo_counts.sum(axis=1), axis=0)
        
        df_p4 = bin_algo_ratio.copy()
        df_p4.to_csv(os.path.join(output_dir, "plot4_data_dominance_by_ppass.csv"))
        
        ax = bin_algo_ratio.plot(kind='bar', stacked=True, figsize=(12, 6), cmap='tab10')
        plt.title('Fastest Algorithm Distribution across GlobalPpass Bins (End-to-End Time)')
        plt.xlabel('GlobalPpass Bins')
        plt.ylabel('Ratio of Best Performance')
        plt.xticks(rotation=45)
        plt.legend(title='Best Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "04_dominance_by_ppass.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # -----------------------------------------
    # 【图 5】：特征提取耗时 (Feature Extraction Cost)
    # -----------------------------------------
    if 'FeatureT_ms' in valid_df.columns:
        df_p5 = valid_df['FeatureT_ms'].describe().reset_index()
        df_p5.columns = ['Statistic', 'FeatureT_ms']
        df_p5.to_csv(os.path.join(output_dir, "plot5_data_feature_time_cost.csv"), index=False)
        
        plt.figure(figsize=(10, 2))
        sns.boxplot(x=valid_df['FeatureT_ms'], color='lightblue')
        plt.title('Distribution of FeatureT_ms (Bitmap Intersection Time)')
        plt.xlabel('Time (ms)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "05_feature_time_cost.png"), dpi=300, bbox_inches='tight')
        plt.close()

    # -----------------------------------------
    # 【图 6】：散点图 (GlobalPpass vs End-to-End Search Time)
    # -----------------------------------------
    cols_p6 = ['QueryID', 'GlobalPpass'] + [c for c in valid_df.columns if 'True_EndToEnd_Time_ms' in c]
    df_p6 = valid_df[cols_p6]
    df_p6.to_csv(os.path.join(output_dir, "plot6_data_scatter.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    if 'True_EndToEnd_Time_ms_pre-filter' in valid_df.columns:
        sns.scatterplot(data=valid_df, x='GlobalPpass', y='True_EndToEnd_Time_ms_pre-filter', 
                        color='red', label='pre-filter', alpha=0.5, s=20)
    if 'True_EndToEnd_Time_ms_ACORN-gamma' in valid_df.columns:
        sns.scatterplot(data=valid_df, x='GlobalPpass', y='True_EndToEnd_Time_ms_ACORN-gamma', 
                        color='blue', label='ACORN-gamma', alpha=0.5, s=20)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.title('GlobalPpass vs True End-to-End Search Time (Log-Log Scale)')
    plt.xlabel('GlobalPpass (Log Scale)')
    plt.ylabel('True End-to-End Time (ms) (Log Scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "06_ppass_vs_time_scatter.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[√] {os.path.basename(output_dir)} 数据集 EDA 分析及绘图数据导出完成！")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    for dataset in DATASETS:
        df_long = load_data(dataset)
        if not df_long.empty:
            df_final = preprocess_and_align(df_long, dataset)
            
            dataset_output_dir = os.path.join(GLOBAL_OUTPUT_DIR, dataset)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            # 宽表导出：此处导出的 CSV 是“干净”的，只有数据，不含“Best_Algo”等路由决策标签
            csv_output_path = os.path.join(dataset_output_dir, f"{dataset}_aligned_results.csv")
            df_final.to_csv(csv_output_path, index=False)
            print(f"[*] 【最全宽表】已导出至: {csv_output_path}")
            
            # 画图函数会在内部创建一个拷贝(temp_df)去临时计算最快算法，不会污染 df_final
            perform_eda(df_final, dataset_output_dir)
        else:
            print(f"[!] 未加载到 {dataset} 的数据，跳过分析。")