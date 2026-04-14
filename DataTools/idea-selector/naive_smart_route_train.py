import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ONNX 导出库支持
try:
    from skl2onnx import convert_sklearn
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False

try:
    import onnxmltools
    ONNXMLTOOLS_AVAILABLE = True
except ImportError:
    ONNXMLTOOLS_AVAILABLE = False

# ==========================================
# 1. 全局配置区域
# ==========================================
DATASET_LIST = ["Amazon", "BookReviews", "Genome", "Music", "Reviews", "Tiktok", "VariousImg", "Laion"] 
BASE_DIR = "/mnt/disk1/syh/ljk/FilterVector/FilterVectorResults"

# 记得修改output_dir！！！！！！

ALGO_LIST = ['ACORN-gamma', 'NaviX', 'UNG-nTfalse', 'pre-filter']
MODELS_TO_TRY = ["RandomForest", "XGBoost", "LightGBM", "DecisionTree"]

NAIVE_STRATEGY = {
    "default": "auto"
}

SUMMARY_OUT_DIR = os.path.join(BASE_DIR, "SelectModels_summary", "naive_smart_route")

# ==========================================
# 2. 核心功能函数
# ==========================================

def create_classifier(model_type="RandomForest", **kwargs):
    if model_type == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        params = {'n_estimators': 150, 'max_depth': 12, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'}
        params.update(kwargs)
        return RandomForestClassifier(**params)
        
    elif model_type == "LightGBM":
        import lightgbm as lgb
        params = {'max_depth': 8, 'learning_rate': 0.05, 'n_estimators': 200, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced', 'verbose': -1}
        params.update(kwargs)
        return lgb.LGBMClassifier(**params)
        
    elif model_type == "XGBoost":
        import xgboost as xgb
        params = {'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 200, 'random_state': 42, 'n_jobs': -1}
        params.update(kwargs)
        return xgb.XGBClassifier(**params)
        
    elif model_type == "DecisionTree":
        from sklearn.tree import DecisionTreeClassifier
        params = {'max_depth': 10, 'random_state': 42, 'class_weight': 'balanced'}
        params.update(kwargs)
        return DecisionTreeClassifier(**params)
        
    else:
        raise ValueError(f"不支持的模型引擎: {model_type}")

def label_best_algorithm(df, min_recall=0.90, threshold=0.15):
    best_algos = []
    fuzzy_count = 0 
    
    for idx, row in df.iterrows():
        candidates = []
        for algo in ALGO_LIST:
            recall_col = f'Recall_{algo}'
            time_col = f'L2_Time_ms_{algo}' 
            
            if recall_col in row and time_col in row and pd.notna(row[recall_col]) and pd.notna(row[time_col]):
                candidates.append({'algo': algo, 'recall': row[recall_col], 'time': row[time_col]})
                
        if not candidates:
            best_algos.append('Unknown')
            continue
            
        qualified = [c for c in candidates if c['recall'] >= min_recall]
        
        if not qualified:
            best = max(candidates, key=lambda x: x['recall'])
            best_algos.append(best['algo'])
        else:
            qualified.sort(key=lambda x: x['time'])
            best = qualified[0]
            
            if len(qualified) > 1:
                second_best = qualified[1]
                time_diff_percent = (second_best['time'] - best['time']) / (best['time'] + 1e-9)
                
                if time_diff_percent < threshold:
                    best_algos.append('Unknown')
                    fuzzy_count += 1
                    continue
                    
            best_algos.append(best['algo'])
            
    print(f"  [Info] 阈值过滤 (Threshold={threshold*100}%): 共剔除 {fuzzy_count} 个模糊样本。")
    return pd.Series(best_algos, index=df.index)

def generate_naive_features(df):
    X = pd.DataFrame(index=df.index)
    # 严格只保留 3 个 feature
    X['GlobalPpass'] = df['GlobalPpass']
    X['NumEntries'] = df['NumEntries']
    X['NumDescendants'] = df['NumDescendants']
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    return X

def train_and_evaluate(X, y, target_map, model_type):
    y_mapped = y.map(target_map)
    valid_mask = y_mapped.notna()
    X_clean, y_clean = X[valid_mask].copy(), y_mapped[valid_mask].astype(int).copy()
    
    class_counts = y_clean.value_counts()
    if class_counts.min() < 2:
        valid_classes = class_counts[class_counts >= 2].index
        safe_mask = y_clean.isin(valid_classes)
        X_clean, y_clean = X_clean[safe_mask], y_clean[safe_mask]

    if model_type == "XGBoost":
        unique_labels = sorted(y_clean.unique())
        remapping = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(unique_labels)}
        y_clean = y_clean.map(remapping)
        internal_to_original = {new_lbl: old_lbl for old_lbl, new_lbl in remapping.items()}
        real_classes = [internal_to_original[i] for i in range(len(internal_to_original))]
    else:
        internal_to_original = {lbl: lbl for lbl in y_clean.unique()}
        real_classes = sorted(y_clean.unique())

    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean)
    
    classifier = create_classifier(model_type=model_type)
    
    t_train_start = time.perf_counter()
    classifier.fit(X_train.values, y_train)
    train_time_ms = (time.perf_counter() - t_train_start) * 1000.0

    t_pred_start = time.perf_counter()
    y_pred = classifier.predict(X_test.values)
    pred_latency_us = ((time.perf_counter() - t_pred_start) * 1e6) / len(X_test)
    
    y_test_orig = y_test.map(internal_to_original)
    y_pred_orig = pd.Series(y_pred, index=X_test.index).map(internal_to_original)
        
    acc = accuracy_score(y_test_orig, y_pred_orig)
    
    inv_map = {v: k for k, v in target_map.items()}
    present_labels = sorted(y_test_orig.unique())
    target_names = [inv_map[l] for l in present_labels]
    
    cls_report = classification_report(y_test_orig, y_pred_orig, labels=present_labels, target_names=target_names, zero_division=0)
    conf_mat = confusion_matrix(y_test_orig, y_pred_orig, labels=present_labels)
    cm_df = pd.DataFrame(conf_mat, index=[f"True_{name}" for name in target_names], columns=[f"Pred_{name}" for name in target_names])
    
    importances_dict = {}
    if hasattr(classifier, 'feature_importances_'):
        imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': classifier.feature_importances_})
        imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        importance_str = imp_df.to_string(formatters={'Importance': '{:.4f}'.format})
        # 提取数值字典用于外部汇总
        importances_dict = dict(zip(X.columns, classifier.feature_importances_))
    else:
        importance_str = "当前模型不支持直接提取特征重要性。"
        
    return {
        "acc": acc,
        "train_time_ms": train_time_ms,
        "pred_latency_us": pred_latency_us,
        "test_size": len(X_test),
        "cls_report": cls_report,
        "cm_df": cm_df,
        "importance_str": importance_str,
        "importances_dict": importances_dict,
        "classifier": classifier,
        "params": classifier.get_params(),
        "pred_counts": y_pred_orig.value_counts(),
        "real_classes": real_classes,
        "test_indices": X_test.index,
        "y_test_orig": y_test_orig,
        "y_pred_orig": y_pred_orig
    }
    
def run_naive_ablation_study(X, y, target_map, best_model_type):
    print(f"  > [正在运行消融实验] 评估基座: {best_model_type}")
    valid_mask = y.notna() & (y != 'Unknown')
    X_clean, y_clean = X[valid_mask], y[valid_mask]
    if len(X_clean) == 0: return {}
    
    all_features = list(X_clean.columns)
    def get_ablated_features(feature_to_remove):
        return [f for f in all_features if f != feature_to_remove]

    configs = {
        "完全体 (All 3 Features)": all_features,
        "消融: 移除 GlobalPpass": get_ablated_features("GlobalPpass"),
        "消融: 移除 NumEntries": get_ablated_features("NumEntries"),
        "消融: 移除 NumDescendants": get_ablated_features("NumDescendants")
    }
    
    results = {}
    for config_name, feat_cols in configs.items():
        res = train_and_evaluate(X_clean[feat_cols], y_clean, target_map, best_model_type)
        results[config_name] = res["acc"]
    return results

def save_onnx_model(classifier, model_type, num_features, output_dir, real_classes=None, filename="naive_smart_route.onnx"):
    onnx_filename = os.path.join(output_dir, filename)
    try:
        if model_type in ["RandomForest", "DecisionTree"] and SKL2ONNX_AVAILABLE:
            from skl2onnx.common.data_types import FloatTensorType as SklFloatTensorType
            from skl2onnx import convert_sklearn
            initial_type = [('float_input', SklFloatTensorType([None, num_features]))]
            onnx_model = convert_sklearn(classifier, initial_types=initial_type, target_opset=15)
            
        elif model_type in ["LightGBM", "XGBoost"] and ONNXMLTOOLS_AVAILABLE:
            from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType
            initial_type = [('float_input', OnnxFloatTensorType([None, num_features]))]
            if model_type == "LightGBM":
                import onnxmltools
                onnx_model = onnxmltools.convert_lightgbm(classifier, initial_types=initial_type, target_opset=15)
            else:  
                import onnxmltools
                onnx_model = onnxmltools.convert_xgboost(classifier, initial_types=initial_type, target_opset=15)
        else:
            print(f"⚠️ 无法导出 {model_type} 的 ONNX 模型。")
            return
            
        with open(onnx_filename, "wb") as f:
            f.write(onnx_model.SerializeToString())
            
        if real_classes is not None:
            import onnx
            model = onnx.load(onnx_filename)
            patched = False
            for node in model.graph.node:
                for attr in node.attribute:
                    if attr.name == 'classlabels_int64s':
                        del attr.ints[:]
                        attr.ints.extend(real_classes)
                        patched = True
            if patched:
                onnx.save(model, onnx_filename)
                print(f"🔧 [ONNX Hack] 成功将底层输出节点硬编码映射为绝对 ID: {real_classes}")
    except Exception as e:
        print(f"❌ 导出 ONNX 模型时发生错误: {e}")

# ==========================================
# 3. 主干执行流
# ==========================================
def process_single_dataset(dataset_name):
    print(f"\n{'='*60}")
    print(f"🚀 开始处理数据集: {dataset_name}")
    print(f"{'='*60}")
    
    csv_path = os.path.join(BASE_DIR, "EDA_Plots", dataset_name, f"{dataset_name}_aligned_results.csv")
    output_dir = os.path.join(BASE_DIR, dataset_name, "SelectModels", "naive_smart_route")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}，跳过。")
        return [], []
        
    df = pd.read_csv(csv_path)
    total_queries = len(df)
    
    # 1. 先将原始的最优算法打标存入一个独立列
    df['Original_Best_Algo'] = label_best_algorithm(df, min_recall=0.90, threshold=0.20)
    
    # 2. 统计当前数据集上，ACORN 家族中占比最大的具体算法 (用于 C++ 端的 fallback 短路机制)
    acorn_mask = df['Original_Best_Algo'].isin(['ACORN-gamma', 'NaviX'])
    if acorn_mask.sum() > 0:
        majority_acorn_algo = df[acorn_mask]['Original_Best_Algo'].mode()[0]
    else:
        majority_acorn_algo = 'ACORN-gamma' # 默认兜底
        
    # 3. 映射为 C++ ID 并保存到 TXT 文件
    cpp_algo_map = {'ACORN-gamma': 2, 'ACORN-improved': 3, 'NaviX': 4}
    majority_id = cpp_algo_map.get(majority_acorn_algo, 2)
    with open(os.path.join(output_dir, "naive_majority_acorn_id.txt"), "w") as f:
        f.write(str(majority_id))
    print(f"  [Info] ACORN 家族多数派为: {majority_acorn_algo} (ID: {majority_id})，已保存至 naive_majority_acorn_id.txt")
    
    # 4. 然后再强制将 ACORN-gamma 和 NaviX 归为一类：ACORN-family，供模型训练使用
    df['Best_Algo'] = df['Original_Best_Algo'].replace({
        'ACORN-gamma': 'ACORN-family',
        'NaviX': 'ACORN-family'
    })
    
    X_All = generate_naive_features(df)
    
    # 映射表变为 3 分类
    TARGET_MAP = {
        'UNG-nTfalse': 0, 
        'pre-filter': 1, 
        'ACORN-family': 2
    }

    model_metrics = []
    feature_importances_list = []
    model_results_cache = {}

    print("\n[模型全量评测中...]")
    for model_type in MODELS_TO_TRY:
        try:
            res = train_and_evaluate(X_All, df['Best_Algo'], TARGET_MAP, model_type)
            model_results_cache[model_type] = res
            
            model_metrics.append({
                "Dataset": dataset_name,
                "Model": model_type,
                "Accuracy": res['acc'],
                "Train_Time_ms": res['train_time_ms'],
                "Pred_Latency_us": res['pred_latency_us'],
                "Test_Size": res['test_size']
            })
            
            # 收集各个模型跑出来的特征重要性
            if res['importances_dict']:
                feat_dict = res['importances_dict'].copy()
                feat_dict['Dataset'] = dataset_name
                feat_dict['Model'] = model_type
                feature_importances_list.append(feat_dict)
                
            print(f"  > 跑通: {model_type:<15}")
        except ImportError:
            print(f"  > ⚠️ 缺少 {model_type} 库。")
        except Exception as e:
            print(f"  > ❌ 模型 {model_type} 评估失败: {e}")

    if not model_metrics:
        print("❌ 所有模型训练失败。")
        return [], []

    table_lines = []
    table_lines.append(f"  {'算法模型(Model)':<16} | {'准确率(Accuracy)':<16} | {'训练耗时(Train ms)':<18} | {'单次推理(Pred μs)':<18}")
    table_lines.append("  " + "-" * 76)
    
    sorted_metrics = sorted(model_metrics, key=lambda x: x["Accuracy"], reverse=True)
    for m in sorted_metrics:
        table_lines.append(f"  {m['Model']:<16} | {m['Accuracy']:<18.4%} | {m['Train_Time_ms']:<18.2f} | {m['Pred_Latency_us']:<18.2f}")
    
    comparison_table_str = "\n".join(table_lines)
    print("\n[所有模型性能大比武 (Model Comparison Summary)]")
    print(comparison_table_str)

    current_strategy = NAIVE_STRATEGY
    if isinstance(NAIVE_STRATEGY, dict):
        selected_model_type = NAIVE_STRATEGY.get(dataset_name, NAIVE_STRATEGY.get("default", "auto"))
    else:
        selected_model_type = NAIVE_STRATEGY

    if selected_model_type == "auto":
        selected_model_type = sorted_metrics[0]['Model']
        strategy_desc = "Auto (全自动取优)"
    else:
        strategy_desc = f"Manual (手动指定 {selected_model_type})"

    print(f"\n[策略选定] {dataset_name} 最终胜出/指定引擎: {selected_model_type} | 策略: {strategy_desc}")

    print("\n[开始为所有模型生成分析报告与消融实验]")
    for m_type, res in model_results_cache.items():
        ablation_results = run_naive_ablation_study(X_All, df['Best_Algo'], TARGET_MAP, m_type)
        
        suffix = f"_{m_type}"
        report_path = os.path.join(output_dir, f"NaiveRoute_Report_{dataset_name}{suffix}.txt")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n")
            f.write("┃                           NaiveRoute                               ┃\n")
            f.write("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n")
            f.write(f"┃ 基座模型: {m_type:<16}  数据集: {dataset_name:<15}            ┃\n")
            f.write(f"┃ 策略状态: {strategy_desc:<45} ┃\n")
            # 在报告头部记录兜底的 ACORN 算法
            f.write(f"┃ 多数派 ACORN: {majority_acorn_algo:<42} ┃\n")
            f.write(f"┃ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<40} ┃\n")
            f.write("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n")
            
            f.write("【零 | 全局模型性能对比 (Model Comparison Summary)】\n")
            f.write("-" * 80 + "\n")
            f.write(comparison_table_str + "\n\n")
            
            f.write("【壹 | 数据集标签分布 (Ground Truth vs Prediction)】\n")
            f.write("-" * 80 + "\n")
            f.write(f"  总查询量: {total_queries} 条 (基于 80/20 划分测试集)\n\n")
            f.write(f"  {'算法分类':<20} | {'真实最优频次 (占比)':<25} | {'模型预测频次':<15}\n")
            f.write("  " + "-" * 65 + "\n")
            
            gt_counts = df['Best_Algo'].value_counts()
            pred_counts = res['pred_counts']
            for algo in list(TARGET_MAP.keys()) + ['Unknown']:
                if algo in gt_counts:
                    gt_c = gt_counts[algo]
                    gt_pct = gt_c / total_queries
                    pred_c = pred_counts.get(algo, 0)
                    f.write(f"  {algo:<20} | {gt_c:>6} 次 ({gt_pct:>6.2%})          | {pred_c:>6} 次\n")
            f.write("\n")
            
            f.write("【贰 | 模型超参数与效率 (Hyperparameters & Efficiency)】\n")
            f.write("-" * 80 + "\n")
            f.write("  [关键超参数]\n")
            params = res['params']
            key_params = ['max_depth', 'n_estimators', 'learning_rate', 'n_jobs', 'random_state']
            for k in key_params:
                if k in params:
                    f.write(f"  - {k:<15} : {params[k]}\n")
            f.write("\n  [时间开销]\n")
            f.write(f"  - 训练集拟合总耗时 : {res['train_time_ms']:.2f} ms\n")
            f.write(f"  - 测试集单条推理延迟 : ~{res['pred_latency_us']:.2f} μs / query\n\n")
            
            if ablation_results:
                base_acc_ab = ablation_results.get("完全体 (All 3 Features)", 0)
                f.write("【叁 | 特征消融实验评估 (Ablation Study on 3 Core Indicators)】\n")
                f.write("-" * 80 + "\n")
                f.write("  (注: 性能衰减越大，说明被移除的特征越重要。出现负数说明移除反而提分)\n\n")
                f.write(f"  {'配置方案':<35} | {'预测准确率':<12} | {'性能衰减':<12}\n")
                f.write("  " + "-" * 65 + "\n")
                for config, acc_ab in ablation_results.items():
                    drop = base_acc_ab - acc_ab
                    drop_str = f"↓ {drop:.2%}" if drop > 0 else f"↑ {abs(drop):.2%}"
                    if config == "完全体 (All 3 Features)": drop_str = "Baseline"
                    f.write(f"  {config:<35} | {acc_ab:>10.2%}   | {drop_str:>10}\n")
                f.write("\n")
                
            f.write("【肆 | 细粒度分类表现 (Classification Report & Confusion Matrix)】\n")
            f.write("-" * 80 + "\n")
            f.write(f"  ▶ 测试集全局准确率 : {res['acc']:.4%}\n\n")
            f.write("  [详细报告 (F1-Score / Precision / Recall)]\n")
            f.write("  " + res['cls_report'].replace('\n', '\n  ') + "\n\n")
            
            f.write("  [混淆矩阵 (Confusion Matrix)]\n")
            f.write("  (行: 真实标签 True | 列: 模型预测标签 Pred)\n")
            cm_str = res['cm_df'].to_string()
            f.write("  " + cm_str.replace('\n', '\n  ') + "\n\n")
            
            f.write("【伍 | 树模型内禀特征重要性 (Feature Importance Weight)】\n")
            f.write("-" * 80 + "\n")
            f.write("  " + res['importance_str'].replace('\n', '\n  ') + "\n\n")

        if m_type == selected_model_type:
            onnx_name = "naive_smart_route.onnx"
            save_onnx_model(res['classifier'], m_type, X_All.shape[1], output_dir, res.get('real_classes'), filename=onnx_name)
            
    return model_metrics, feature_importances_list

def main():
    global_metrics = []
    global_importances = []
    
    # 遍历所有数据集并收集统计指标
    for dataset in DATASET_LIST:
        dataset_metrics, dataset_importances = process_single_dataset(dataset)
        if dataset_metrics:
            global_metrics.extend(dataset_metrics)
            global_importances.extend(dataset_importances)
            
    # 如果有成功运行的指标，聚合生成全局 CSV
    if global_metrics:
        os.makedirs(SUMMARY_OUT_DIR, exist_ok=True)
        df_all = pd.DataFrame(global_metrics)
        
        cols_order = ["Dataset", "Model", "Accuracy", "Train_Time_ms", "Pred_Latency_us", "Test_Size"]
        df_all = df_all[cols_order]
        csv1_path = os.path.join(SUMMARY_OUT_DIR, "all_datasets_metrics.csv")
        df_all.to_csv(csv1_path, index=False)
        print(f"\n✅ [全局报表 1] 各数据集算法明细已保存至: {csv1_path}")
        
        df_avg = df_all.groupby("Model")[["Accuracy", "Train_Time_ms", "Pred_Latency_us"]].mean().reset_index()
        csv2_path = os.path.join(SUMMARY_OUT_DIR, "average_metrics.csv")
        df_avg.to_csv(csv2_path, index=False)
        print(f"✅ [全局报表 2] 跨数据集算法均值已保存至: {csv2_path}")

        # 【导出新增的 feature importance 报表】
        if global_importances:
            df_imp = pd.DataFrame(global_importances)
            # 整理列的顺序，让 Dataset 和 Model 排在最前面
            imp_cols = ['Dataset', 'Model', 'GlobalPpass', 'NumEntries', 'NumDescendants']
            df_imp = df_imp[[c for c in imp_cols if c in df_imp.columns]]
            
            csv3_path = os.path.join(SUMMARY_OUT_DIR, "feature_importances.csv")
            df_imp.to_csv(csv3_path, index=False)
            print(f"✅ [全局报表 3] 特征重要性统计已保存至: {csv3_path}\n")

if __name__ == "__main__":
    main()