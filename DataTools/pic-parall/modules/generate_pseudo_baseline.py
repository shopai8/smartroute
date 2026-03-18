# 根据指定的 Global Ratio，将原 UNG 算法的耗时缩减，自动生成 ImprovedUNG 伪基线结果文件
import os
import glob
import json
import pandas as pd

def generate_pseudo_baseline_csv(dataset_name: str, ung_csv_path: str, target_csv_path: str) -> bool:
    """
    1.一句话概括函数核心作用：根据指定的 Global Ratio，将原 UNG 算法的耗时缩减，自动生成 ImprovedUNG 伪基线结果文件。
    
    2.思路说明：
      - 预置各数据集的 Global Ratio 字典（硬编码）。
      - 检查并读取原始 UNG-nTfalse 的 search_time_summary.csv 文件。
      - 将 Average_Time_ms 列的数值乘以对应数据集的 Global Ratio，模拟 IntelELS 优化后的耗时。
      - 自动创建目标文件夹目录，并将修改后的 DataFrame 另存为目标 CSV 文件。
      
    3.输入参数：
      - dataset_name: str，含义（必填）：当前处理的数据集名称，用于匹配对应的 Global Ratio。
      - ung_csv_path: str，含义（必填）：真实的 UNG 算法 summary csv 文件的绝对路径。
      - target_csv_path: str，含义（必填）：要生成的伪基线 csv 文件的保存绝对路径。
      
    4.返回值类型和具体含义：
      - bool: 若成功生成并保存文件则返回 True，若原文件不存在、读取为空或遇到异常则返回 False。
    """
    # 根据日志提取的 Global Ratio 字典
    GLOBAL_RATIOS = {
        "Amazon": 0.6515,
        "BookReviews": 0.2381,
        "Genome": 0.2812,
        "Music": 0.3487,
        "Reviews": 0.3112,
        "Tiktok": 0.8880,
        "VariousImg": 0.9767,
        "Laion": 1.0  # 默认兜底
    }
    
    if not os.path.exists(ung_csv_path):
        return False
        
    try:
        df = pd.read_csv(ung_csv_path)
        if df.empty:
            return False
            
        ratio = GLOBAL_RATIOS.get(dataset_name, 1.0)
        
        # 仅缩减时间，Recall 保持原 UNG 的能力不变
        df['Average_Time_ms'] = pd.to_numeric(df['Average_Time_ms'], errors='coerce') * ratio
        
        # 保存到目标路径
        os.makedirs(os.path.dirname(target_csv_path), exist_ok=True)
        df.to_csv(target_csv_path, index=False)
        return True
    except Exception as e:
        print(f"[WARN] 处理 {ung_csv_path} 时失败: {e}")
        return False

def main():
    # 1. 加载配置文件获取基础输出目录
    config_file = "/home/fengxiaoyao/FilterVector/FilterVectorCode/DataTools/pic-parall/config_overall_qps.json"
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        base_results_dir = config['global_settings']['base_results_dir']
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        return
        
    datasets = ["Amazon", "BookReviews", "Genome", "Music", "Reviews", "Tiktok", "VariousImg", "Laion"]
    source_alg = "UNG-nTfalse"
    target_alg = "ImprovedUNG"
    
    success_count = 0
    
    print(f"开始扫描目录: {base_results_dir} ...\n")
    
    # 2. 遍历数据集并使用 glob 搜索所有的 UNG 结果文件
    for dataset in datasets:
        # 匹配模式：寻找所有属于 UNG-nTfalse 算法的 search_time_summary.csv
        search_pattern = os.path.join(base_results_dir, dataset, "Results", source_alg, "*", "results", "search_time_summary.csv")
        matched_files = glob.glob(search_pattern)
        
        print(f"在 {dataset} 中找到 {len(matched_files)} 个 {source_alg} 结果文件。")
        
        # 3. 逐个生成新的伪基线文件
        for ung_csv in matched_files:
            # 将路径中的算法名称替换为伪基线名称
            target_csv = ung_csv.replace(f"/Results/{source_alg}/", f"/Results/{target_alg}/")
            
            if generate_pseudo_baseline_csv(dataset, ung_csv, target_csv):
                success_count += 1
                
    print(f"\n[完成] 共成功生成 {success_count} 个 {target_alg} 伪基线文件！")

if __name__ == "__main__":
    main()