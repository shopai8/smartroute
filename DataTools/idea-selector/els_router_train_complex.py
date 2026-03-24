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
DATASET_LIST = ["Amazon", "BookReviews", "Genome", "Music", "Reviews", "Tiktok", "VariousImg", "Laion"]
BASE_DIR = "/home/fengxiaoyao/FilterVector/FilterVectorResults"
TOP_N_FEATURES_TO_SELECT = 7  # 最终模型保留的特征维度
PERCENTAGE_THRESHOLD = 0.2    # 性能差异阈值 (20%)

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

def generate_complex_features(df):
    """
    1.一句话概括函数核心作用：利用基础特征与 Trie 静态信息生成高阶组合特征池。
    2.思路说明：
      - 将 QuerySize, CandSize, GlobalPpass 作为基础底座。
      - 引入树结构指标，计算覆盖率、选择率、密度等非线性交叉项。
      - 替换异常值并补零，确保树模型不报错。
    3.输入参数：
      - df: pd.DataFrame，含义（必填，包含原始列的数据集）
    4.返回值类型和具体含义：
      - pd.DataFrame：包含 10+ 个备选特征的 DataFrame。
    """
    raw = df.copy()
    eps = 1e-9
    
    features = pd.DataFrame(index=raw.index)
    
    # --- 基础特征 ---
    features['QuerySize'] = raw['QuerySize']
    features['CandSize'] = raw['CandSize']
    features['GlobalPpass'] = raw['GlobalPpass']
    
    # --- 对数特征 ---
    log_cand = np.log1p(raw['CandSize'])
    log_query = np.log1p(raw['QuerySize'])
    
    # --- 比例与覆盖率特征 ---
    features['Cand_Coverage_Ratio'] = raw['CandSize'] / (raw['TrieTotalNodes'] + eps)
    features['Query_Cardinality_Ratio'] = raw['QuerySize'] / (raw['TrieLabelCardinality'] + eps)
    
    avg_nodes_per_label = raw['TrieTotalNodes'] / (raw['TrieLabelCardinality'] + eps)
    features['Cand_Selectivity'] = avg_nodes_per_label / (raw['CandSize'] + eps)
    features['Query_Cand_Ratio'] = raw['QuerySize'] / (raw['CandSize'] + eps)
    
    # --- 乘积与交互特征 ---
    features['Query_Path_Density'] = raw['QuerySize'] * raw['TrieAvgBranchingFactor']
    features['Cand_x_Query_Interaction'] = raw['CandSize'] * raw['QuerySize']
    features['Branching_x_CandSize'] = raw['TrieAvgBranchingFactor'] * raw['CandSize']
    features['Branching_x_QuerySize'] = raw['TrieAvgBranchingFactor'] * raw['QuerySize']
    features['LogCand_x_LogQuery'] = log_cand * log_query
    
    # --- 二次方特征 ---
    features['CandSize_sq'] = raw['CandSize'] ** 2
    features['QuerySize_sq'] = raw['QuerySize'] ** 2
    
    # 清理异常值
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
    print(f"🚀 开始重构多特征 ELS Router | 数据集: {dataset_name}")
    print(f"{'='*70}")
    
    csv_path = os.path.join(BASE_DIR, "EDA_Plots_UNGnTtrue", dataset_name, f"{dataset_name}_aligned_results.csv")
    output_dir = os.path.join(BASE_DIR, dataset_name, "SelectModels", "intelElS_complex")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到宽表文件: {csv_path}，跳过。")
        return
        
    try:
        df_clean, y = load_and_label_els_data(csv_path, threshold=PERCENTAGE_THRESHOLD)
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
        
    print(f"[*] 数据过滤 (Threshold > {PERCENTAGE_THRESHOLD*100}% 差异):")
    counts = y.value_counts()
    print(f"  - 过滤后剩余样本数: {len(df_clean)} | Label 0: {counts.get(0, 0)} | Label 1: {counts.get(1, 0)}")
    
    if len(df_clean) < 10:
        return

    # 1. 生成大特征池
    X_full = generate_complex_features(df_clean)
    
    # 2. 阶段一：全量探索
    _, _, _, _, imp_df_full = train_and_evaluate_rf(X_full, y, "阶段一：全量组合特征探索性训练")
    
    # 3. 获取 Top 7
    top_n_features = imp_df_full['Feature'].head(TOP_N_FEATURES_TO_SELECT).tolist()
    
    print("\n" + "*"*60)
    print(f"🔥 C++ 重点对齐: 数据集 [{dataset_name}] 的 Top 7 特征如下:")
    for idx, feat in enumerate(top_n_features):
        print(f"   [{idx+1}] {feat}")
    print("*"*60 + "\n")
    
    # 4. 阶段二：精简训练
    X_top = X_full[top_n_features]
    final_model, acc_p2, duration_p2, report_p2, imp_df_top = train_and_evaluate_rf(
        X_top, y, f"阶段二：最终 Top {TOP_N_FEATURES_TO_SELECT} 特征模型训练"
    )
    
    # 5. 导出
    joblib_path = os.path.join(output_dir, "idea1_selector_model_complex.joblib")
    onnx_path = os.path.join(output_dir, "idea1_selector_model_complex.onnx")
    
    joblib.dump(final_model, joblib_path)
    if SKL2ONNX_AVAILABLE:
        initial_type = [('float_input', FloatTensorType([None, TOP_N_FEATURES_TO_SELECT]))]
        onnx_model = convert_sklearn(final_model, initial_types=initial_type, target_opset=15)
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"✅ ONNX 模型成功导出: {onnx_path}")
        
    # 6. 报告记录
    report_path = os.path.join(output_dir, f"ELS_Router_Report_Complex_{dataset_name}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"数据集: {dataset_name} | 测试集准确率: {acc_p2:.4%}\n")
        f.write("-" * 40 + "\n")
        f.write("Top 7 特征列表:\n")
        f.write("\n".join(top_n_features) + "\n")
        
if __name__ == "__main__":
    for dataset in DATASET_LIST:
        process_dataset(dataset)