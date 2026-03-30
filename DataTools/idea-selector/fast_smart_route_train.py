import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib

# ONNX 导出库支持
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType as SklFloatTensorType
    SKL2ONNX_AVAILABLE = True
except ImportError:
    SKL2ONNX_AVAILABLE = False

try:
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType as OnnxFloatTensorType
    ONNXMLTOOLS_AVAILABLE = True
except ImportError:
    ONNXMLTOOLS_AVAILABLE = False

# ==========================================
# 1. 全局配置区域
# ==========================================
## "Amazon", "BookReviews", "Genome", "Music", "Reviews", "Tiktok", "VariousImg", "Laion"
DATASET_LIST = ["Amazon", "BookReviews", "Genome", "Music", "Reviews", "Tiktok", "VariousImg", "Laion"] 
BASE_DIR = "/home/fengxiaoyao/FilterVector/FilterVectorResults"

ALGO_LIST = ['ACORN-gamma', 'ACORN-improved', 'NaviX', 'UNG-nTfalse', 'UNG-nTtrue', 'pre-filter']
ACORN_FAMILY = ['ACORN-gamma', 'ACORN-improved', 'NaviX']

# 模型竞技场候选者扩容
MODELS_TO_TRY = ["RandomForest", "XGBoost", "LightGBM", "DecisionTree"]

# 模糊样本过滤阈值
MARGIN_THRESHOLD = 0.20 

# SMOTE 样本平衡开关 (处理长尾类别的召回)
USE_SMOTE_L1 = False
USE_SMOTE_L2 = False

# 组装策略控制 (Cascade Strategy)
CASCADE_STRATEGY = {
    "default": "auto"
}

# 全局汇总 CSV 存放目录
SUMMARY_OUT_DIR = os.path.join(BASE_DIR, "SelectModels_summary", "fast_smart_route_2")

# ==========================================
# 2. 核心功能与特征工厂
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
        raise ValueError(f"不支持的模型类型: {model_type}")

def label_best_algorithm_by_time(df, time_prefix='L1_Time_ms', min_recall=0.90, threshold=0.15):
    best_algos = []
    fuzzy_count = 0
    for idx, row in df.iterrows():
        candidates = []
        for algo in ALGO_LIST:
            recall_col = f'Recall_{algo}'
            time_col = f'{time_prefix}_{algo}'
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
            
    return pd.Series(best_algos, index=df.index)

def generate_cascade_features(df):
    X_L1 = pd.DataFrame(index=df.index)
    X_L1['QuerySize'] = df['QuerySize']
    X_L1['CandSize'] = df['CandSize']
    X_L1['GlobalPpass'] = df['GlobalPpass']
    
    X_L2 = X_L1.copy() 
    X_L2['NumEntries'] = df['NumEntries']
    X_L2['NumDescendants'] = df['NumDescendants']
    
    X_L1.replace([np.inf, -np.inf], np.nan, inplace=True); X_L1.fillna(0, inplace=True)
    X_L2.replace([np.inf, -np.inf], np.nan, inplace=True); X_L2.fillna(0, inplace=True)
    
    return X_L1, X_L2

def train_and_evaluate_model(X_train, y_train, X_test, y_test, target_map, model_type, use_smote=False):
    y_train_mapped = y_train.map(target_map)
    y_test_mapped = y_test.map(target_map)
    
    train_mask = y_train_mapped.notna()
    X_train_clean = X_train[train_mask].copy()
    y_train_clean = y_train_mapped[train_mask].astype(int)
    
    test_mask = y_test_mapped.notna()
    X_test_clean = X_test[test_mask].copy()
    y_test_clean = y_test_mapped[test_mask].astype(int) 
    
    train_unique = sorted(y_train_clean.unique())
    remapping = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(train_unique)}
    y_train_cont = y_train_clean.map(remapping).astype(int)
    
    internal_to_original = {new_lbl: old_lbl for old_lbl, new_lbl in remapping.items()}
    real_classes = [internal_to_original[i] for i in range(len(internal_to_original))]

    if use_smote:
        try:
            min_samples = y_train_cont.value_counts().min()
            safe_k = min(5, min_samples - 1) if min_samples > 1 else 1
            if min_samples > 1:
                smote = SMOTE(random_state=42, k_neighbors=safe_k)
                X_train_clean, y_train_cont = smote.fit_resample(X_train_clean, y_train_cont)
        except Exception:
            pass

    X_train_np = X_train_clean.values
    y_train_np = y_train_cont.values
    X_test_np = X_test_clean.values

    classifier = create_classifier(model_type=model_type)
    
    t_train_start = time.perf_counter()
    classifier.fit(X_train_np, y_train_np)
    train_time_ms = (time.perf_counter() - t_train_start) * 1000.0
    
    t_pred_start = time.perf_counter()
    y_pred_np = classifier.predict(X_test_np)
    # 统一耗时口径，直接计算单次推理耗时 (微秒)
    pred_latency_us = ((time.perf_counter() - t_pred_start) * 1e6) / len(X_test_np)
    
    y_pred_abs = np.array([real_classes[int(idx)] for idx in y_pred_np])
    y_test_abs = y_test_clean.values 
    
    acc = accuracy_score(y_test_abs, y_pred_abs)
    
    inv_map = {v: k for k, v in target_map.items()}
    present_labels = sorted(np.unique(np.concatenate((y_test_abs, y_pred_abs))))
    target_names = [inv_map[l] for l in present_labels]
    
    cls_report = classification_report(y_test_abs, y_pred_abs, labels=present_labels, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_test_abs, y_pred_abs, labels=present_labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{name}" for name in target_names], columns=[f"Pred_{name}" for name in target_names])
    
    if hasattr(classifier, 'feature_importances_'):
        importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': classifier.feature_importances_})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        importance_str = importance_df.to_string(formatters={'Importance': '{:.4f}'.format})
    else:
        importance_str = "当前模型不支持提取 Feature Importances。"
        
    return {
        "acc": acc,
        "train_time_ms": train_time_ms,
        "pred_latency_us": pred_latency_us,
        "cls_report": cls_report,
        "cm_df": cm_df,
        "importance_str": importance_str,
        "classifier": classifier,
        "real_classes": real_classes,
        "test_size": len(X_test_np)
    }

def run_arena_for_layer(X_train, y_train, X_test, y_test, target_map, layer_name, use_smote=False):
    smote_status = "启用" if use_smote else "未启用"
    print(f"\n[{layer_name} - 模型竞技场启动 | SMOTE: {smote_status}]")
    results_cache = {}
    metrics_list = []
    
    for model_type in MODELS_TO_TRY:
        try:
            res = train_and_evaluate_model(X_train, y_train, X_test, y_test, target_map, model_type, use_smote=use_smote)
            results_cache[model_type] = res
            metrics_list.append({
                "Model": model_type,
                "Accuracy": res['acc'],
                "Train_Time_ms": res['train_time_ms'],
                "Pred_Latency_us": res['pred_latency_us'],
                "Test_Size": res['test_size']
            })
            print(f"  > {model_type:<15} | 准确率: {res['acc']:.4%} | 训练耗时: {res['train_time_ms']:.2f} ms")
        except ImportError:
             print(f"  > ⚠️ 缺少 {model_type} 引擎依赖库。")
        except Exception as e:
            print(f"  > ⚠️ {model_type} 训练失败: {e}")
            
    return results_cache, metrics_list

def save_onnx_model(classifier, model_type, num_features, output_dir, filename, real_classes=None):
    onnx_filename = os.path.join(output_dir, filename)
    try:
        if model_type in ["RandomForest", "DecisionTree"] and SKL2ONNX_AVAILABLE:
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
                print(f"🔧 [ONNX Hack] {filename} 已成功硬编码底层输出节点。")
    except Exception as e:
        print(f"❌ 导出 ONNX 模型时发生错误: {e}")

def calculate_cascade_system_accuracy(X_L1_test, X_L2_test, y_global_best_test, 
                                      l1_clf, l1_real_classes, L1_MAP,
                                      els_clf,
                                      l2_clf, l2_real_classes, L2_MAP,
                                      majority_acorn_algo):
    inv_L1 = {v: k for k, v in L1_MAP.items()}
    inv_L2 = {v: k for k, v in L2_MAP.items()}
    
    preds_l1_raw = l1_clf.predict(X_L1_test.values)
    preds_l1_mapped = [l1_real_classes[int(idx)] for idx in preds_l1_raw]
    
    final_preds = []
    for i, l1_mapped_idx in enumerate(preds_l1_mapped):
        l1_decision = inv_L1[l1_mapped_idx]
        
        if l1_decision == 'pre-filter':
            final_preds.append('pre-filter')
        elif l1_decision == 'ACORN_Family':
            final_preds.append(majority_acorn_algo) 
        else: 
            els_decision = 'UNG-nTfalse' 
            if els_clf is not None:
                els_features = X_L1_test.iloc[[i]].values 
                els_pred = els_clf.predict(els_features)[0]
                els_decision = 'UNG-nTtrue' if els_pred == 1 else 'UNG-nTfalse'
                
            row_features = X_L2_test.iloc[[i]].values
            pred_l2_raw = l2_clf.predict(row_features)[0]
            pred_l2_mapped = l2_real_classes[int(pred_l2_raw)]
            l2_decision = inv_L2[pred_l2_mapped]
            
            if l2_decision == 'UNG_Family':
                final_preds.append(els_decision)
            else:
                final_preds.append(l2_decision)
            
    valid_mask = y_global_best_test != 'Unknown'
    y_true_valid = y_global_best_test[valid_mask]
    final_preds_valid = [final_preds[j] for j, valid in enumerate(valid_mask) if valid]
    
    return accuracy_score(y_true_valid, final_preds_valid)

def generate_comparison_table(metrics_list, layer_name, feature_count):
    metrics_list.sort(key=lambda x: x["Accuracy"], reverse=True)
    lines = [f"[{layer_name} - 性能对比 (特征数: {feature_count})]"]
    lines.append(f"  {'算法模型(Model)':<16} | {'准确率(Accuracy)':<16} | {'训练耗时(Train ms)':<18} | {'单次推理(Pred μs)':<18}")
    lines.append("  " + "-" * 76)
    for m in metrics_list:
        lines.append(f"  {m['Model']:<16} | {m['Accuracy']:<18.4%} | {m['Train_Time_ms']:<18.2f} | {m['Pred_Latency_us']:<18.2f}")
    return "\n".join(lines) + "\n"

# ==========================================
# 3. 单一数据集处理总线
# ==========================================
def process_single_dataset(dataset_name):
    print(f"\n{'='*70}")
    print(f"🚀 开始处理双层路由数据集: {dataset_name}")
    print(f"{'='*70}")
    
    csv_path = os.path.join(BASE_DIR, "EDA_Plots", dataset_name, f"{dataset_name}_aligned_results.csv")
    output_dir = os.path.join(BASE_DIR, dataset_name, "SelectModels", "fast_smart_route_2")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"FastSmartRoute_Cascade_Report_{dataset_name}.txt")
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}，跳过。")
        return []
        
    df = pd.read_csv(csv_path)
    
    df['Global_Best'] = label_best_algorithm_by_time(df, time_prefix='L1_Time_ms', min_recall=0.90, threshold=MARGIN_THRESHOLD)
    df['L2_Best'] = label_best_algorithm_by_time(df, time_prefix='L2_Time_ms', min_recall=0.90, threshold=MARGIN_THRESHOLD)
    
    df['L1_Target'] = df['Global_Best'].replace({a: 'ACORN_Family' for a in ACORN_FAMILY})
    df['L1_Target'] = df['L1_Target'].replace({'UNG-nTfalse': 'NEED_ELS', 'UNG-nTtrue': 'NEED_ELS'})
    df['L2_Target'] = df['L2_Best'].replace({'UNG-nTfalse': 'UNG_Family', 'UNG-nTtrue': 'UNG_Family'})
    
    X_L1, X_L2 = generate_cascade_features(df)
    
    valid_mask = (df['Global_Best'] != 'Unknown') & (df['L2_Best'] != 'Unknown')
    valid_indices = df[valid_mask].index
    
    train_idx, test_idx = train_test_split(valid_indices, test_size=0.2, random_state=42)
    
    X_L1_train, X_L1_test = X_L1.loc[train_idx], X_L1.loc[test_idx]
    X_L2_train, X_L2_test = X_L2.loc[train_idx], X_L2.loc[test_idx]
    
    y_L1_train, y_L1_test = df.loc[train_idx, 'L1_Target'], df.loc[test_idx, 'L1_Target']
    y_L2_train, y_L2_test = df.loc[train_idx, 'L2_Target'], df.loc[test_idx, 'L2_Target']
    y_Global_Best_test = df.loc[test_idx, 'Global_Best']

    els_model_path = os.path.join(BASE_DIR, dataset_name, "SelectModels", "intelElS", "idea1_selector_model_final.joblib")
    els_clf = None
    if os.path.exists(els_model_path):
        print(f"[ELS 集成] 成功加载 ELS Router: {els_model_path}")
        els_clf = joblib.load(els_model_path)
    else:
        print("[ELS 集成] ⚠️ 未找到 ELS 模型，级联评估时 UNG 将默认走 nTfalse。")

    L1_TARGET_MAP = {'NEED_ELS': 0, 'ACORN_Family': 1, 'pre-filter': 2}
    L2_TARGET_MAP = {'UNG_Family': 0, 'ACORN-gamma': 1, 'ACORN-improved': 2, 'NaviX': 3, 'pre-filter': 4}

    l1_cache, l1_metrics = run_arena_for_layer(X_L1_train, y_L1_train, X_L1_test, y_L1_test, L1_TARGET_MAP, "Layer 1 (网关层)", use_smote=USE_SMOTE_L1)
    l2_cache, l2_metrics = run_arena_for_layer(X_L2_train, y_L2_train, X_L2_test, y_L2_test, L2_TARGET_MAP, "Layer 2 (裁判层)", use_smote=USE_SMOTE_L2)

    current_strategy = CASCADE_STRATEGY
    if isinstance(CASCADE_STRATEGY, dict):
        current_strategy = CASCADE_STRATEGY.get(dataset_name, CASCADE_STRATEGY.get("default", "auto"))

    if isinstance(current_strategy, dict):
        best_l1_model = current_strategy.get("L1", "RandomForest")
        best_l2_model = current_strategy.get("L2", "RandomForest")
        strategy_desc = f"Manual (针对 {dataset_name} 指定 L1:{best_l1_model}, L2:{best_l2_model})"
    else: 
        best_l1_model = max(l1_metrics, key=lambda x: x['Accuracy'])['Model']
        best_l2_model = max(l2_metrics, key=lambda x: x['Accuracy'])['Model']
        strategy_desc = "Auto (全自动综合取优)"

    print(f"\n[策略选定] L1 胜出模型: {best_l1_model} | L2 胜出模型: {best_l2_model}")
    
    l1_res = l1_cache[best_l1_model]
    l2_res = l2_cache[best_l2_model]

    acorn_mask_train = df.loc[train_idx, 'L1_Target'] == 'ACORN_Family'
    if acorn_mask_train.sum() > 0:
        majority_acorn_algo = df.loc[train_idx][acorn_mask_train]['Global_Best'].mode()[0]
    else:
        majority_acorn_algo = 'ACORN-gamma'
        
    cpp_algo_map = {'ACORN-gamma': 2, 'ACORN-improved': 3, 'NaviX': 4}
    with open(os.path.join(output_dir, "l1_majority_acorn_id.txt"), "w") as f:
        f.write(str(cpp_algo_map.get(majority_acorn_algo, 2)))

    system_acc = calculate_cascade_system_accuracy(
        X_L1_test, X_L2_test, y_Global_Best_test, 
        l1_res['classifier'], l1_res['real_classes'], L1_TARGET_MAP,
        els_clf,
        l2_res['classifier'], l2_res['real_classes'], L2_TARGET_MAP,
        majority_acorn_algo
    )

    save_onnx_model(l1_res['classifier'], best_l1_model, X_L1_train.shape[1], output_dir, "l1_router.onnx", l1_res['real_classes'])
    save_onnx_model(l2_res['classifier'], best_l2_model, X_L2_train.shape[1], output_dir, "l2_router.onnx", l2_res['real_classes'])

    # 汇总输出到 TXT 战报
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n")
        f.write("┃               Fast Cascade Routing Assessment Report               ┃\n")
        f.write("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n")
        f.write(f"┃ 数据集  : {dataset_name:<15} 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<20} ┃\n")
        f.write(f"┃ 组装策略: {strategy_desc:<52} ┃\n")
        f.write(f"┃ SMOTE   : L1({USE_SMOTE_L1}), L2({USE_SMOTE_L2}){ ' ' * 38}┃\n")
        f.write("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n")
        
        f.write("【零 | 双层路由竞技场横向大比武 (Layer Arena Comparison)】\n")
        f.write("-" * 80 + "\n")
        f.write(generate_comparison_table(l1_metrics, "Layer 1 - 极速拦截层", X_L1_train.shape[1]) + "\n")
        f.write(generate_comparison_table(l2_metrics, "Layer 2 - 图结构裁判层", X_L2_train.shape[1]) + "\n")

        f.write("【壹 | 级联决算与端到端表现 (End-to-End Evaluation)】\n")
        f.write("-" * 80 + "\n")
        f.write(f"  ▶ 选定 L1 引擎 : {best_l1_model}\n")
        f.write(f"  ▶ 选定 L2 引擎 : {best_l2_model}\n")
        f.write(f"  ▶ 多数派短路兜底: {majority_acorn_algo}\n\n")
        f.write(f"  🚀 模拟流水线分发最终准确率 (System End-to-End Accuracy): {system_acc:.4%}\n")
        f.write("  (注: 包含 Margin 阈值过滤后的测试集，请求依次流经 L1 -> ELS(若触发) -> L2)\n\n")

        f.write("【贰 | 胜出模型深度透视 (Selected Models Deep Dive)】\n")
        f.write("-" * 80 + "\n")
        
        f.write(f"[L1 胜出者: {best_l1_model}]\n")
        f.write("  ▶ 详细分类报告 (Classification Report):\n")
        f.write("  " + l1_res['cls_report'].replace('\n', '\n  ') + "\n")
        f.write("  ▶ 混淆矩阵 (Confusion Matrix):\n")
        f.write("  " + l1_res['cm_df'].to_string().replace('\n', '\n  ') + "\n\n")
        f.write("  ▶ 特征重要性 (Feature Importance):\n")
        f.write("  " + l1_res['importance_str'].replace('\n', '\n  ') + "\n\n")

        f.write(f"[L2 胜出者: {best_l2_model}]\n")
        f.write("  ▶ 详细分类报告 (Classification Report):\n")
        f.write("  " + l2_res['cls_report'].replace('\n', '\n  ') + "\n")
        f.write("  ▶ 混淆矩阵 (Confusion Matrix):\n")
        f.write("  " + l2_res['cm_df'].to_string().replace('\n', '\n  ') + "\n\n")
        f.write("  ▶ 特征重要性 (Feature Importance):\n")
        f.write("  " + l2_res['importance_str'].replace('\n', '\n  ') + "\n\n")
        
    print(f"\n✅ 评估完成！详尽级联战报已同步至: {report_path}")
    
    # 收集当前数据集的所有指标，并附加 Layer 标签供全局 CSV 使用
    dataset_metrics = []
    for m in l1_metrics:
        m_copy = m.copy()
        m_copy.update({"Dataset": dataset_name, "Layer": "L1"})
        dataset_metrics.append(m_copy)
        
    for m in l2_metrics:
        m_copy = m.copy()
        m_copy.update({"Dataset": dataset_name, "Layer": "L2"})
        dataset_metrics.append(m_copy)
        
    # 追加端到端表现
    dataset_metrics.append({
        "Dataset": dataset_name,
        "Layer": "System_End_to_End",
        "Model": f"L1({best_l1_model})_L2({best_l2_model})",
        "Accuracy": system_acc,
        "Train_Time_ms": np.nan, 
        "Pred_Latency_us": np.nan,
        "Test_Size": len(y_Global_Best_test[y_Global_Best_test != 'Unknown'])
    })
    
    return dataset_metrics

def main():
    global_metrics = []
    
    # 遍历所有数据集并收集统计指标
    for dataset in DATASET_LIST:
        dataset_metrics = process_single_dataset(dataset)
        if dataset_metrics:
            global_metrics.extend(dataset_metrics)
            
    # 聚合生成全局 CSV
    if global_metrics:
        os.makedirs(SUMMARY_OUT_DIR, exist_ok=True)
        df_all = pd.DataFrame(global_metrics)
        
        # [CSV 1]: 细粒度明细表
        cols_order = ["Dataset", "Layer", "Model", "Accuracy", "Train_Time_ms", "Pred_Latency_us", "Test_Size"]
        df_all = df_all[[c for c in cols_order if c in df_all.columns]]
        csv1_path = os.path.join(SUMMARY_OUT_DIR, "fast_all_datasets_metrics.csv")
        df_all.to_csv(csv1_path, index=False)
        print(f"\n✅ [全局报表 1] 各数据集双层算法明细已保存至: {csv1_path}")
        
        # [CSV 2]: 跨数据集求平均值表 (仅计算 L1 和 L2 各自的模型均值，排除端到端结合行)
        df_layer_only = df_all[df_all["Layer"] != "System_End_to_End"]
        df_avg = df_layer_only.groupby(["Layer", "Model"])[["Accuracy", "Train_Time_ms", "Pred_Latency_us"]].mean().reset_index()
        csv2_path = os.path.join(SUMMARY_OUT_DIR, "fast_average_metrics.csv")
        df_avg.to_csv(csv2_path, index=False)
        print(f"✅ [全局报表 2] 跨数据集双层算法均值已保存至: {csv2_path}\n")

if __name__ == "__main__":
    main()