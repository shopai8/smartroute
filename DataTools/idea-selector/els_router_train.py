import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False

# ==========================================
# 1. 全局配置区域
# ==========================================
# "Amazon", "BookReviews", "Genome", "Music", "Reviews", "Tiktok", "VariousImg", "Laion"
DATASET_LIST = ["BookReviews", "Genome", "Music", "Reviews", "Tiktok", "VariousImg", "Laion"]
BASE_DIR = "/home/fengxiaoyao/FilterVector/FilterVectorResults"
PERCENTAGE_THRESHOLD = 0.4    # 性能差异阈值，低于此差异的样本将被视为模糊样本剔除

# ==========================================
# 2. 核心功能与特征工厂
# ==========================================

def load_and_label_els_data(csv_path, threshold=0.2):
    df = pd.read_csv(csv_path)
    col_f = 'MinSupersetT_ms_UNG-nTfalse'
    col_t = 'MinSupersetT_ms_UNG-nTtrue'
    
    # 包含了极简特征以及 Trie 树特征
    required_features = [
        'QuerySize', 'CandSize', 'GlobalPpass', 
        'TrieTotalNodes', 'TrieLabelCardinality', 'TrieAvgBranchingFactor'
    ]
    
    for feat in [col_f, col_t] + required_features:
        if feat not in df.columns:
            raise ValueError(f"宽表中缺失必需列: {feat}")
            
    # === 修复数据骤降: 对 Trie 静态全局特征进行智能填充 ===
    trie_static_cols = ['TrieTotalNodes', 'TrieLabelCardinality', 'TrieAvgBranchingFactor']
    for col in trie_static_cols:
        # 取该列第一个非空的有效值，填充给所有的 NaN
        valid_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else 0
        df[col].fillna(valid_val, inplace=True)
    # ========================================================
            
    # 此时 dropna 就不会因为 Trie 特征缺失而误杀大量行了
    df = df.dropna(subset=[col_f, col_t] + required_features).copy()
    
    df['time_diff_abs'] = np.abs(df[col_t] - df[col_f])
    df['min_time'] = np.minimum(df[col_t], df[col_f])
    df['time_diff_percent'] = df['time_diff_abs'] / (df['min_time'] + 1e-9)
    
    df_clean = df[df['time_diff_percent'] > threshold].copy()
    y = (df_clean[col_t] < df_clean[col_f]).astype(int)
    
    return df_clean, y

def generate_els_features(df):
    """
    1.一句话概括函数核心作用：提取极简的 3 个基础维度特征。
    2.思路说明：
      - 摒弃复杂的数学组合，直接切片提取 QuerySize, CandSize, GlobalPpass。
      - 清洗生成的特征（替换 Inf 为 NaN 并填充 0），确保输出纯净。
    3.输入参数：
      - df: pd.DataFrame，含义（必填，包含基础特征列的宽表数据框）
    4.返回值类型和具体含义：
      - pd.DataFrame：返回仅包含这 3 个基础特征的数据集。
    """
    features = df[['QuerySize', 'CandSize', 'GlobalPpass']].copy()
    
    # 清理异常值（兜底）
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)
    
    return features

from sklearn.metrics import confusion_matrix

def train_and_evaluate_rf(X, y, description):
    print(f"\n[执行阶段] {description}")
    
    class_counts = y.value_counts()
    if len(class_counts) > 1 and class_counts.min() >= 2:
        stratify_param = y
    else:
        stratify_param = None
        min_count = class_counts.min() if len(class_counts) > 0 else 0
        print(f"  ⚠️ 警告: 某类别样本极度不平衡或单一 (最少类别仅 {min_count} 条)，已禁用分层拆分。")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify_param)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced')
    
    t0 = time.perf_counter()
    rf.fit(X_train.values, y_train.values)
    duration = time.perf_counter() - t0
    
    y_pred = rf.predict(X_test.values)
    acc = accuracy_score(y_test, y_pred)
    
    report = classification_report(y_test, y_pred, labels=[0, 1], target_names=['nTfalse (0)', 'nTtrue (1)'], zero_division=0)
    
    # === 新增混淆矩阵计算与格式化 ===
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=['True_nTfalse(0)', 'True_nTtrue(1)'], columns=['Pred_nTfalse(0)', 'Pred_nTtrue(1)'])
    
    imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
    imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    print(f"  ▶ 拟合耗时 : {duration:.4f} 秒")
    print(f"  ▶ 测试集 Acc: {acc:.4%}")
    print("  [混淆矩阵]")
    print("  " + cm_df.to_string().replace('\n', '\n  '))
    
    # 将混淆矩阵拼接到 report 字符串里，这样后续写 txt 也能顺便写进去
    full_report = report + "\n\n[Confusion Matrix]\n" + cm_df.to_string()
    
    return rf, acc, duration, full_report, imp_df

# ==========================================
# 3. 主干业务流
# ==========================================

def process_dataset(dataset_name):
    print(f"\n{'='*70}")
    print(f"🚀 开始重构极简 ELS Router | 数据集: {dataset_name}")
    print(f"{'='*70}")
    
    csv_path = os.path.join(BASE_DIR, "EDA_Plots_UNGnTtrue", dataset_name, f"{dataset_name}_aligned_results.csv")
    output_dir = os.path.join(BASE_DIR, dataset_name, "SelectModels", "intelElS")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到宽表文件: {csv_path}，跳过。")
        return
        
    # 1. 数据加载与打标
    try:
        df_clean, y = load_and_label_els_data(csv_path, threshold=PERCENTAGE_THRESHOLD)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
        
    print(f"[*] 数据过滤 (Threshold > {PERCENTAGE_THRESHOLD*100}% 差异):")
    print(f"  - 过滤后剩余有效样本数: {len(df_clean)}")
    counts = y.value_counts()
    print(f"  - Label 0 (nTfalse 更快): {counts.get(0, 0)} 条")
    print(f"  - Label 1 (nTtrue 更快) : {counts.get(1, 0)} 条")
    
    if len(df_clean) < 10:
        print("❌ 有效样本过少，停止训练。")
        return

    # 2. 提取 3 个基础特征
    X_features = generate_els_features(df_clean)
    
    # 3. 极简训练一波流
    final_model, acc, duration, report, imp_df = train_and_evaluate_rf(
        X_features, y, "基于 3 个极简特征训练最终 ELS 模型"
    )
    
    print("\n" + "*"*60)
    print(f"🔥 C++ 重点对齐: C++ 端 calculate_idea1_features 必须严格按此顺序返回:")
    for idx, feat in enumerate(X_features.columns):
        print(f"   [{idx+1}] {feat}")
    print("*"*60 + "\n")
    
    # 4. 模型持久化导出
    joblib_path = os.path.join(output_dir, "idea1_selector_model_final.joblib")
    onnx_path = os.path.join(output_dir, "idea1_selector_model_final.onnx")
    
    joblib.dump(final_model, joblib_path)
    if SKL2ONNX_AVAILABLE:
        # 注意: 这里 input tensor 的维度固定为了 len(X_features.columns) 即 3
        initial_type = [('float_input', FloatTensorType([None, len(X_features.columns)]))]
        onnx_model = convert_sklearn(final_model, initial_types=initial_type, target_opset=15)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"✅ ONNX 模型成功导出: {onnx_path}")
    else:
        print("⚠️ 未安装 skl2onnx，跳过 ONNX 导出。")
        
    # 5. 生成最终战报
    report_path = os.path.join(output_dir, f"ELS_Router_Report_{dataset_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n")
        f.write("┃                ELS Router (Idea 1 - Minimal Features)              ┃\n")
        f.write("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n")
        f.write(f"┃ 数据集  : {dataset_name:<15} 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<20} ┃\n")
        f.write("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n")
        
        f.write("【壹 | 数据集与标签分布】\n")
        f.write(f"  ▶ 过滤阈值 (MinSupersetT_ms 相对差异) : > {PERCENTAGE_THRESHOLD*100}%\n")
        f.write(f"  ▶ 保留有效样本数 : {len(df_clean)}\n")
        f.write(f"  ▶ Label 0 (nTfalse 胜出) : {counts.get(0, 0)}\n")
        f.write(f"  ▶ Label 1 (nTtrue 胜出)  : {counts.get(1, 0)}\n\n")
        
        f.write("【贰 | 特征列表 (严格对齐顺序)】\n")
        f.write("  " + " | ".join(X_features.columns) + "\n\n")
        
        f.write("【叁 | 模型表现】\n")
        f.write(f"  ▶ 测试集准确率 : {acc:.4%}\n")
        f.write(f"  ▶ 拟合耗时     : {duration:.4f} 秒\n\n")
        f.write("  [分类报告]\n")
        f.write("  " + report.replace('\n', '\n  ') + "\n\n")
        f.write("  [特征重要性]\n")
        f.write("  " + imp_df.to_string().replace('\n', '\n  ') + "\n")
        
if __name__ == "__main__":
    for dataset in DATASET_LIST:
        process_dataset(dataset)