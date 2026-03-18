# 手动指定 QueryID 和原始数据文件路径，然后提取特定 QueryID 的数据并计算平均值，最后输出search_time_summary.csv
import pandas as pd
import os

def process_ung_query(input_file, target_query_id, output_file):
    """
    读取UNG算法原始数据，提取指定QueryID，计算平均值并输出。
    """
    # 1. 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：找不到文件 {input_file}")
        return

    try:
        # 2. 读取数据
        print(f"正在读取文件: {input_file} ...")
        df = pd.read_csv(input_file)
        
        # 检查必要的列是否存在
        required_cols = ['QueryID', 'Lsearch', 'efs', 'Time_ms', 'Recall']
        if not all(col in df.columns for col in required_cols):
            print(f"错误：输入文件缺少必要的列。请确保包含: {required_cols}")
            return

        # 3. 筛选指定的 QueryID
        # 注意：QueryID 在 csv 中可能是整数也可能是字符串，这里做个统一转换对比
        # 先判断数据中的类型
        if df['QueryID'].dtype == object:
             target_query_id = str(target_query_id)
        
        target_df = df[df['QueryID'] == target_query_id].copy()

        if target_df.empty:
            print(f"警告：在文件中未找到 QueryID 为 {target_query_id} 的数据。")
            return

        print(f"找到 QueryID {target_query_id} 的数据行数: {len(target_df)}")

        # 4. 数据聚合
        # 按照 Lsearch 和 efs 进行分组
        # 计算 Time_ms 和 Recall 的平均值 (跨多个 repeat)
        result_df = target_df.groupby(['Lsearch', 'efs']).agg({
            'Time_ms': 'mean',
            'Recall': 'mean'
        }).reset_index()

        # 5. 重命名列以符合输出要求
        result_df.rename(columns={
            'efs': 'Average_Efs',
            'Time_ms': 'Average_Time_ms',
            'Recall': 'Average_Recall'
        }, inplace=True)

        # 6. 整理列顺序并排序
        output_cols = ['Lsearch', 'Average_Efs', 'Average_Time_ms', 'Average_Recall']
        final_output = result_df[output_cols]
        
        # 按 Average_Efs 排序
        final_output = final_output.sort_values(by='Average_Efs')

        # 7. 输出到 CSV
        final_output.to_csv(output_file, index=False)
        print(f"处理完成！结果已保存至: {output_file}")
        print("\n结果预览:")
        print(final_output.head())

    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    # ================= 配置区域 =================
    
    # 1. 设置新的UNG算法原始csv文件路径
    ung_file = '/home/fengxiaoyao/FilterVector/FilterVectorResults/Music/Results/ACORN-gamma/Index[M32_LB100_alpha1.2_C6_EP16_AN1511563_AM32_AMB64_AG80]_GT[GT_query_select_imp_A_B_C-weighted-sub-base-123456789_K10]_Search[Ls1000-Le40000-Lp1000_efsS200-efss1000-efsf1000-lt5000_K10_th100]/results/query_details_repeat1.csv' 
    
    # 2. 手动输入上一步得到的 QueryID 
    target_qid = 9
    
    # 3. 设置输出文件名
    output_file = 'acorn_result_query.csv'
    
    # ===========================================
    
    process_ung_query(ung_file, target_qid, output_file)