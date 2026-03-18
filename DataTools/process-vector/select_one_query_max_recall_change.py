import pandas as pd
import numpy as np

def process_search_data(input_file, output_file, param_name='efs', time_limit=None, recall_limit=None):
    """
    param_name: 用于分析趋势的变量名，通常为 'efs' 或 'Lsearch'
    time_limit: 最小参数下的 Time_ms 上限 (若超标则剔除)
    recall_limit: 最小参数下的 Recall 下限 (若低于此值则剔除)
    """
    # 1. 读取数据
    try:
        df = pd.read_csv(input_file)
        print(f"数据读取成功，正在处理 (分析变量: {param_name})...")
        print(f"原始包含 QueryID 数量: {df['QueryID'].nunique()}")
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return
    except Exception as e:
        print(f"读取文件出错: {e}")
        return

    # 检查指定的列是否存在
    if param_name not in df.columns:
        print(f"错误：CSV文件中找不到指定的列名 '{param_name}'")
        return

    # ================= 新增：基于最小参数的【时间】和【Recall】剔除逻辑 =================
    if time_limit is not None or recall_limit is not None:
        print("正在应用起始条件过滤 (检查最小参数下的表现)...")
        
        # 步骤 A: 计算每个 (QueryID, param_name) 组合的平均值
        stats = df.groupby(['QueryID', param_name]).agg({
            'Time_ms': 'mean',
            'Recall': 'mean'
        }).reset_index()
        
        # 步骤 B: 找出每个 QueryID 对应的 最小 param_name 的那一行数据
        min_param_indices = stats.groupby('QueryID')[param_name].idxmin()
        min_param_data = stats.loc[min_param_indices]
        
        bad_qids = set()
        
        # 步骤 C1: 检查时间 (Time > Limit 则剔除)
        if time_limit is not None:
            time_bad = min_param_data[min_param_data['Time_ms'] > time_limit]['QueryID'].unique()
            bad_qids.update(time_bad)
            print(f"  - 设定时间上限: {time_limit} ms (剔除 {len(time_bad)} 个起始耗时过高的 QueryID)")

        # 步骤 C2: 检查 Recall (Recall < Limit 则剔除)
        if recall_limit is not None:
            recall_bad = min_param_data[min_param_data['Recall'] < recall_limit]['QueryID'].unique()
            bad_qids.update(recall_bad)
            print(f"  - 设定 Recall 下限: {recall_limit} (剔除 {len(recall_bad)} 个起始 Recall 过低的 QueryID)")

        # 步骤 D: 从原始数据中剔除这些 QueryID
        if len(bad_qids) > 0:
            original_rows = len(df)
            df = df[~df['QueryID'].isin(bad_qids)]
            print(f"过滤后剩余 QueryID 数量: {df['QueryID'].nunique()} (共删除了 {original_rows - len(df)} 行数据)")
        else:
            print("没有 QueryID 被剔除（所有 QueryID 在最小参数下均符合要求）。")
            
        if df.empty:
            print("错误：所有数据都被剔除了，请检查限制条件是否过严。")
            return
    # ======================================================================

    # 2. 预处理：计算每个 QueryID 在不同 param_name 下的平均 Recall
    try:
        agg_df = df.groupby(['QueryID', param_name])['Recall'].mean().reset_index()
    except KeyError as e:
        print(f"数据处理错误，请检查列名: {e}")
        return

    best_qid = None
    max_recall_span = -1.0
    
    unique_qids = agg_df['QueryID'].unique()
    print(f"正在筛选最佳候选者...")

    # 3. 筛选逻辑
    for qid in unique_qids:
        sub_data = agg_df[agg_df['QueryID'] == qid]
        
        # 条件: 最大 Recall 必须 >= 0.9
        max_recall = sub_data['Recall'].max()
        if max_recall < 0.9:
            continue
            
        # 计算 Recall 的跨度 (Max - Min)
        min_recall = sub_data['Recall'].min()
        span = max_recall - min_recall
        
        if span > max_recall_span:
            max_recall_span = span
            best_qid = qid

    if best_qid is None:
        print("未找到符合条件（最大 Recall >= 0.9）的 QueryID。")
        return

    print(f"--------------------------------------------------")
    print(f"选中的 QueryID: {best_qid}")
    print(f"分析变量: {param_name}")
    print(f"限制条件: Start_Time <= {time_limit} ms, Start_Recall >= {recall_limit}")
    print(f"该 QueryID 的 Recall 跨度: {max_recall_span:.4f} (Max: {agg_df[agg_df['QueryID']==best_qid]['Recall'].max():.4f})")
    print(f"--------------------------------------------------")

    # 4. 整理输出数据
    target_df = df[df['QueryID'] == best_qid].copy()

    result_df = target_df.groupby(['Lsearch', 'efs']).agg({
        'Time_ms': 'mean',
        'Recall': 'mean'
    }).reset_index()

    # 5. 重命名列
    result_df.rename(columns={
        'efs': 'Average_Efs',
        'Time_ms': 'Average_Time_ms',
        'Recall': 'Average_Recall'
    }, inplace=True)

    # 6. 输出整理
    output_cols = ['Lsearch', 'Average_Efs', 'Average_Time_ms', 'Average_Recall']
    final_output = result_df[output_cols]
    
    sort_col = 'Average_Efs' if param_name == 'efs' else param_name
    final_output = final_output.sort_values(by=sort_col)

    final_output.to_csv(output_file, index=False)
    print(f"结果已保存至: {output_file}")
    print("\n输出文件预览:")
    print(final_output.head())

if __name__ == "__main__":
    # ================= 配置区域 =================
    
    # 1. 变量名 ('efs' 或 'Lsearch')
    search_param = 'efs'  
    #search_param = 'Lsearch' 
    
    # 2. 【Time Limit】: 最小参数下的耗时不能超过此值 (设为 None 则不限制)
    max_time_limit = 100
    
    # 3. 【Recall Limit】: 最小参数下的 Recall 不能低于此值 (设为 None 则不限制)
    # 例如：设置 0.5 表示在 efs/Lsearch 最小时，Recall 至少要有 0.5
    min_recall_limit = 0.5
    
    # 4. 文件路径
    input_filename = '/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/UNG-nTfalse/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss200-efsf200-lt5000_K10_th100]/results/query_details_repeat1.csv'
    output_filename = 'ung_query_result.csv' 
    
    # ===========================================
    
    process_search_data(input_filename, output_filename, search_param, max_time_limit, min_recall_limit)