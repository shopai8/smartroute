import os
import pandas as pd
import numpy as np
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
# 支持批量处理的数据集列表
DATASET_LIST = ["Amazon", "BookReviews", "Genome", "Music", "Reviews", "Tiktok", "VariousImg", "Laion"] 
BASE_DIR = "/home/fengxiaoyao/FilterVector/FilterVectorResults"

ALGO_LIST = ['ACORN-gamma', 'ACORN-improved', 'NaviX', 'UNG-nTfalse', 'UNG-nTtrue', 'pre-filter']
ACORN_FAMILY = ['ACORN-gamma', 'ACORN-improved', 'NaviX']

# 模型竞技场候选者
MODELS_TO_TRY = ["RandomForest", "XGBoost"]

# 模糊样本过滤阈值 (冠亚军耗时差异小于该值则丢弃该样本)
MARGIN_THRESHOLD = 0.20 

# ==========================================
# 2. 核心功能与特征工厂
# ==========================================
def create_classifier(model_type="RandomForest", **kwargs):
    if model_type == "RandomForest":
        params = {'n_estimators': 100, 'max_depth': 12, 'random_state': 42, 'n_jobs': -1, 'class_weight': 'balanced'}
        params.update(kwargs)
        return RandomForestClassifier(**params)
    elif model_type == "XGBoost":
        import xgboost as xgb
        params = {'max_depth': 6, 'learning_rate': 0.1, 'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
        params.update(kwargs)
        return xgb.XGBClassifier(**params)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def label_best_algorithm_by_time(df, time_prefix='L1_Time_ms', min_recall=0.90, threshold=0.15):
    """
    1.一句话概括函数核心作用：基于纯搜索时间为数据集打上最优算法标签，并引入 Margin Threshold 过滤模糊样本。
    2.思路说明：
      - 找出所有召回率达标的候选算法，并按耗时从快到慢排序。
      - 提取第一名(最快)和第二名，计算性能差异百分比：(Time_2nd - Time_1st) / Time_1st。
      - 如果差距小于 threshold，说明冠亚军难分伯仲，强制标记为 'Unknown' 以剔除该噪音样本。
    3.输入参数：
      - df: pd.DataFrame，含义（必填，宽表数据）
      - time_prefix: str，含义（可选，时间度量前缀，如 L1_Time_ms 或 L2_Time_ms）
      - min_recall: float，含义（可选，最低召回率，默认 0.90）
      - threshold: float，含义（可选，第一名必须比第二名快出的百分比，默认 0.15）
    4.返回值类型和具体含义：
      - pd.Series：最优算法标签序列，模糊样本将被标记为 'Unknown'。
    """
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
            
    print(f"  [Info] {time_prefix} 阈值过滤 (Threshold={threshold*100}%): 剔除 {fuzzy_count} 个模糊样本。")
    return pd.Series(best_algos, index=df.index)

def generate_cascade_features(df):
    """
    1.一句话概括函数核心作用：生成级联路由 L1 和 L2 所需的极简特征。
    2.思路说明：
      - L1 特征仅包含前置的 3 个基础特征：QuerySize, CandSize, GlobalPpass。
      - L2 特征在 L1 基础上，追加算完 ELS 才会获得的 NumEntries 和 NumDescendants。
    3.输入参数：
      - df: pd.DataFrame，含义（必填，包含所有特征列的原始宽表）
    4.返回值类型和具体含义：
      - tuple(pd.DataFrame, pd.DataFrame)：返回清洗完毕的 X_L1 和 X_L2 两个特征集。
    """
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

def train_and_evaluate_model(X_train, y_train, X_test, y_test, target_map, model_type, use_smote=True):
    # 1. 映射为目标字典中的绝对 ID
    y_train_mapped = y_train.map(target_map)
    y_test_mapped = y_test.map(target_map)
    
    # 2. 剥离无法映射的脏数据 ('Unknown')
    train_mask = y_train_mapped.notna()
    X_train_clean = X_train[train_mask].copy()
    y_train_clean = y_train_mapped[train_mask].astype(int)
    
    test_mask = y_test_mapped.notna()
    X_test_clean = X_test[test_mask].copy()
    y_test_clean = y_test_mapped[test_mask].astype(int) 
    
    # 3. 建立从 0 开始的连续隐式映射
    train_unique = sorted(y_train_clean.unique())
    remapping = {old_lbl: new_lbl for new_lbl, old_lbl in enumerate(train_unique)}
    y_train_cont = y_train_clean.map(remapping).astype(int)
    
    internal_to_original = {new_lbl: old_lbl for old_lbl, new_lbl in remapping.items()}
    real_classes = [internal_to_original[i] for i in range(len(internal_to_original))]

    # 4. SMOTE 平衡
    train_size_orig = len(X_train_clean)
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

    # 5. 模型训练
    classifier = create_classifier(model_type=model_type)
    start_time = time.perf_counter()
    classifier.fit(X_train_np, y_train_np)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    
    # 6. 预测并还原为绝对 ID 以计算真实准确率
    y_pred_np = classifier.predict(X_test_np)
    y_pred_abs = np.array([real_classes[int(idx)] for idx in y_pred_np])
    y_test_abs = y_test_clean.values 
    
    acc = accuracy_score(y_test_abs, y_pred_abs)
    
    inv_map = {v: k for k, v in target_map.items()}
    present_labels = sorted(np.unique(np.concatenate((y_test_abs, y_pred_abs))))
    target_names = [inv_map[l] for l in present_labels]
    
    report_str  = f"  ▶ 底层引擎     : {model_type}\n"
    smote_status = f" (SMOTE后 {len(X_train_np)} 条)" if use_smote else " (未启用SMOTE)"
    report_str += f"  ▶ 数据分布     : 训练集 {train_size_orig} 条{smote_status} | 测试集 {len(X_test_np)} 条\n"
    report_str += f"  ▶ 拟合耗时     : {duration_ms:.2f} ms\n"
    report_str += f"  ▶ 测试集准确率 : {acc:.4%}\n\n"
    
    report_str += "  [核心指标 (Classification Report)]\n"
    cls_report = classification_report(y_test_abs, y_pred_abs, labels=present_labels, target_names=target_names, zero_division=0)
    report_str += "  " + cls_report.replace('\n', '\n  ') + "\n"
    
    cm = confusion_matrix(y_test_abs, y_pred_abs, labels=present_labels)
    cm_df = pd.DataFrame(cm, index=[f"True_{name}" for name in target_names], columns=[f"Pred_{name}" for name in target_names])
    report_str += "  [混淆矩阵 (Confusion Matrix)]\n"
    report_str += "  " + cm_df.to_string().replace('\n', '\n  ') + "\n\n"

    if hasattr(classifier, 'feature_importances_'):
        importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': classifier.feature_importances_})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        importance_str = importance_df.to_string(formatters={'Importance': '{:.4f}'.format})
    else:
        importance_str = "当前模型不支持提取 Feature Importances。"
        
    return {
        "report_str": report_str,
        "importance_str": importance_str,
        "duration_ms": duration_ms,
        "classifier": classifier,
        "acc": acc,
        "real_classes": real_classes
    }

def save_onnx_model(classifier, model_type, num_features, output_dir, filename, real_classes=None):
    onnx_filename = os.path.join(output_dir, filename)
    try:
        if model_type == "RandomForest" and SKL2ONNX_AVAILABLE:
            initial_type = [('float_input', SklFloatTensorType([None, num_features]))]
            onnx_model = convert_sklearn(classifier, initial_types=initial_type, target_opset=15)
        elif model_type == "XGBoost" and ONNXMLTOOLS_AVAILABLE:
            initial_type = [('float_input', OnnxFloatTensorType([None, num_features]))]
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
                print(f"🔧 [ONNX Hack] 成功将 {filename} 底层节点硬编码为绝对映射 ID: {real_classes}")
    except Exception as e:
        print(f"❌ 导出 ONNX 模型时发生错误: {e}")

def calculate_cascade_system_accuracy(X_L1_test, X_L2_test, y_global_best_test, 
                                      l1_clf, l1_real_classes, L1_MAP,
                                      els_clf,
                                      l2_clf, l2_real_classes, L2_MAP,
                                      majority_acorn_algo):
    """
    1.一句话概括函数核心作用：模拟 C++ 端的真实推演路径，计算级联架构的端到端最终准确率。
    2.思路说明：
      - 第一层：使用 l1_clf 拦截 ACORN 和 pre-filter 分支。
      - ELS层：如果 L1 预测为 'NEED_ELS'，则调用预先训练好的 intelELS 模型 (els_clf) 裁定 nTtrue 或 nTfalse。
      - 第二层：调用 l2_clf 进行复判。若 L2 判定使用 UNG_Family，则最终输出为 ELS 层的裁定结果。
    3.输入参数：
      - X_L1_test / X_L2_test: pd.DataFrame，含义（必填，L1和L2特征测试集）
      - y_global_best_test: pd.Series，含义（必填，测试集的真实全局最优标签）
      - l1_clf / l2_clf: Classifier，含义（必填，训练好的随机森林对象）
      - els_clf: Classifier，含义（必填，加载自 intelElS 的 ELS 路由模型对象）
      - majority_acorn_algo: str，含义（必填，用于 L1 阶段的 ACORN 多数派短路兜底）
    4.返回值类型和具体含义：
      - tuple(float, list)：返回系统整体 Accuracy 和具体的预测列表。
    """
    inv_L1 = {v: k for k, v in L1_MAP.items()}
    inv_L2 = {v: k for k, v in L2_MAP.items()}
    
    preds_l1_raw = l1_clf.predict(X_L1_test.values)
    preds_l1_mapped = [l1_real_classes[int(idx)] for idx in preds_l1_raw]
    
    final_preds = []
    for i, l1_mapped_idx in enumerate(preds_l1_mapped):
        l1_decision = inv_L1[l1_mapped_idx]
        
        # --- 1. L1 极速放行 ---
        if l1_decision == 'pre-filter':
            final_preds.append('pre-filter')
        elif l1_decision == 'ACORN_Family':
            final_preds.append(majority_acorn_algo) 
        else: 
            # --- 2. 走到这里的都需要计算 ELS，我们先模拟 ELS 模型做决定 ---
            els_decision = 'UNG-nTfalse' # 默认保守分支
            if els_clf is not None:
                # 极简版 ELS 模型的 3 个特征正好与 L1 完全重合！
                els_features = X_L1_test.iloc[[i]].values 
                els_pred = els_clf.predict(els_features)[0]
                els_decision = 'UNG-nTtrue' if els_pred == 1 else 'UNG-nTfalse'
                
            # --- 3. 将 L2 特征喂给 L2 裁判长 ---
            row_features = X_L2_test.iloc[[i]].values
            pred_l2_raw = l2_clf.predict(row_features)[0]
            pred_l2_mapped = l2_real_classes[int(pred_l2_raw)]
            l2_decision = inv_L2[pred_l2_mapped]
            
            # 若 L2 确认使用 UNG 家族，则输出 ELS 的裁定；否则听 L2 (如反悔改判 ACORN-gamma)
            if l2_decision == 'UNG_Family':
                final_preds.append(els_decision)
            else:
                final_preds.append(l2_decision)
            
    # 只针对 Global Best 不是 Unknown 的样本计算准确率
    valid_mask = y_global_best_test != 'Unknown'
    y_true_valid = y_global_best_test[valid_mask]
    final_preds_valid = [final_preds[j] for j, valid in enumerate(valid_mask) if valid]
    
    acc = accuracy_score(y_true_valid, final_preds_valid)
    return acc, final_preds

# ==========================================
# 3. 单一数据集处理总线
# ==========================================
def process_single_dataset(dataset_name):
    print(f"\n{'='*70}")
    print(f"🚀 开始处理双层路由数据集: {dataset_name}")
    print(f"{'='*70}")
    
    csv_path = os.path.join(BASE_DIR, "EDA_Plots", dataset_name, f"{dataset_name}_aligned_results.csv")
    output_dir = os.path.join(BASE_DIR, dataset_name, "SelectModels", "fast_smart_route")
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"FastSmartRoute_Cascade_Report_{dataset_name}.txt")
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}，跳过。")
        return
        
    df = pd.read_csv(csv_path)
    
    # 核心：带 15% 容忍度的打标过滤
    df['Global_Best'] = label_best_algorithm_by_time(df, time_prefix='L1_Time_ms', min_recall=0.90, threshold=MARGIN_THRESHOLD)
    df['L2_Best'] = label_best_algorithm_by_time(df, time_prefix='L2_Time_ms', min_recall=0.90, threshold=MARGIN_THRESHOLD)
    
    # 组装 L1 和 L2 的目标标签
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

    # ==================================
    # 阶段零：加载 ELS Router 模型
    # ==================================
    els_model_path = os.path.join(BASE_DIR, dataset_name, "SelectModels", "intelElS", "idea1_selector_model_final.joblib")
    els_clf = None
    if os.path.exists(els_model_path):
        print(f"\n[ELS 模型集成] 成功加载 ELS Router: {els_model_path}")
        els_clf = joblib.load(els_model_path)
    else:
        print("\n[ELS 模型集成] ⚠️ 未找到 ELS 模型，级联评估时 UNG 将默认走 nTfalse。")

    # ==================================
    # 阶段一: Layer 1 自动竞技场
    # ==================================
    print("\n[Layer 1 拦截网关 - 模型竞技场]")
    L1_TARGET_MAP = {'NEED_ELS': 0, 'ACORN_Family': 1, 'pre-filter': 2}
    best_l1_acc, best_l1_model, best_l1_res = -1, "", None
    
    for model_type in MODELS_TO_TRY:
        try:
            res = train_and_evaluate_model(X_L1_train, y_L1_train, X_L1_test, y_L1_test, L1_TARGET_MAP, model_type, use_smote=True)
            print(f"  > {model_type:<15} | 准确率: {res['acc']:.4%} | 耗时: {res['duration_ms']:.2f} ms")
            if res['acc'] > best_l1_acc:
                best_l1_acc = res['acc']
                best_l1_model = model_type
                best_l1_res = res
        except Exception as e:
            print(f"  > ⚠️ {model_type} 训练失败: {e}")
            
    print(f"🏆 L1 胜出模型: {best_l1_model} (Acc: {best_l1_acc:.4%})")
    save_onnx_model(best_l1_res['classifier'], best_l1_model, X_L1_train.shape[1], output_dir, "l1_router.onnx", best_l1_res['real_classes'])

    preds_l1_raw = best_l1_res['classifier'].predict(X_L1.loc[valid_indices].values)
    preds_l1_mapped = [best_l1_res['real_classes'][int(idx)] for idx in preds_l1_raw]
    leak_count = np.sum(np.array(preds_l1_mapped) == 0) # 0 is NEED_ELS
    optimal_l2_inflow_rate = (df.loc[valid_indices, 'L1_Target'] == 'NEED_ELS').sum() / len(valid_indices)

    # --- Layer 1.5 极简策略 (直接选取多数派算法) ---
    acorn_mask_train = df.loc[train_idx, 'L1_Target'] == 'ACORN_Family'
    if acorn_mask_train.sum() > 0:
        majority_acorn_algo = df.loc[train_idx][acorn_mask_train]['Global_Best'].mode()[0]
    else:
        majority_acorn_algo = 'ACORN-gamma' # 默认兜底
    print(f"\n[Layer 1.5 极简策略] 选定 ACORN 家族多数派为: {majority_acorn_algo}")
    
    # ==================================
    # 阶段二: Layer 2 自动竞技场
    # ==================================
    print("\n[Layer 2 裁判层 - 模型竞技场]")
    # L2 模型的目标中，UNG 统称为 UNG_Family，内部决策交给 ELS
    L2_TARGET_MAP = {
        'UNG_Family': 0, 'ACORN-gamma': 1, 
        'ACORN-improved': 2, 'NaviX': 3, 'pre-filter': 4
    }
    best_l2_acc, best_l2_model, best_l2_res = -1, "", None
    
    for model_type in MODELS_TO_TRY:
        try:
            res = train_and_evaluate_model(X_L2_train, y_L2_train, X_L2_test, y_L2_test, L2_TARGET_MAP, model_type, use_smote=False)
            print(f"  > {model_type:<15} | 准确率: {res['acc']:.4%} | 耗时: {res['duration_ms']:.2f} ms")
            if res['acc'] > best_l2_acc:
                best_l2_acc = res['acc']
                best_l2_model = model_type
                best_l2_res = res
        except Exception as e:
            print(f"  > ⚠️ {model_type} 训练失败: {e}")

    print(f"🏆 L2 胜出模型: {best_l2_model} (Acc: {best_l2_acc:.4%})")
    save_onnx_model(best_l2_res['classifier'], best_l2_model, X_L2_train.shape[1], output_dir, "l2_router.onnx", best_l2_res['real_classes'])

    # ==================================
    # 阶段三: 系统级端到端推演
    # ==================================
    system_acc, _ = calculate_cascade_system_accuracy(
        X_L1_test, X_L2_test, y_Global_Best_test, 
        best_l1_res['classifier'], best_l1_res['real_classes'], L1_TARGET_MAP,
        els_clf,
        best_l2_res['classifier'], best_l2_res['real_classes'], L2_TARGET_MAP,
        majority_acorn_algo
    )
    print(f"\n🚀 【最终战报】系统级端到端准确率: {system_acc:.4%}")

    # ==========================================
    # 生成最终战报文本
    # ==========================================
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n")
        f.write("┃                           FastSmartRoute                           ┃\n")
        f.write("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n")
        f.write(f"┃ 混合基座: L1({best_l1_model}) / L2({best_l2_model})\n")
        f.write(f"┃ 数据集  : {dataset_name:<15} 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<20} ┃\n")
        f.write("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n")
        
        f.write("【★核心总结 | 系统端到端架构表现 (End-to-End System Evaluation)】\n")
        f.write("-" * 70 + "\n")
        f.write(f"  ▶ Margin 阈值过滤       : < {MARGIN_THRESHOLD*100}%\n")
        f.write(f"  ▶ 模拟全流水线分发准确率: {system_acc:.4%}\n")
        f.write(f"    (基于独立 Test 集，流经 L1 -> L1.5(多数派) -> ELS -> L2，与全局最优对比)\n\n")
        
        f.write("【壹 | 原始数据算法全局支配域分布 (Global Absolute Best)】\n")
        f.write("-" * 70 + "\n")
        for algo, count in df.loc[valid_indices, 'Global_Best'].value_counts().items():
            f.write(f"  > {algo:<18} : {count:>5} 次 ({count/len(valid_indices):>6.2%})\n")
        f.write("\n")
        
        f.write("【贰 | Layer 1 (极速先验拦截网关) 解析】\n")
        f.write("-" * 70 + "\n")
        f.write(f"  [特征池] 极简基础 3 特征: QuerySize, CandSize, GlobalPpass\n")
        f.write(f"  [输出目标] {L1_TARGET_MAP}\n")
        f.write(f"  ▶▶ L2 拓扑计算触发率 : {leak_count} 次 ({leak_count/len(valid_indices):.2%})\n")
        f.write(f"     (理论极限界值: {optimal_l2_inflow_rate:.2%}，即真实需要计算 ELS 的比例)\n")
        f.write(f"  ▶▶ ACORN 兜底策略 : {majority_acorn_algo} (L1预测为ACORN家族时直接选中)\n\n")
        f.write(best_l1_res['report_str'] + "\n")
        f.write("  [特征重要性权重]\n")
        f.write("  " + best_l1_res['importance_str'].replace('\n', '\n  ') + "\n\n")
        
        f.write("【叁 | Layer 2 (全视野兜底裁判层) 解析】\n")
        f.write("-" * 70 + "\n")
        f.write(f"  [特征池] 基础 3 特征 + 图结构 2 特征: NumEntries, NumDescendants\n")
        f.write(f"  [输出目标] {L2_TARGET_MAP}\n")
        f.write(f"  [内部机制] 若预测为 UNG_Family，则结果交由 ELS Router 接管裁定。\n\n")
        f.write(best_l2_res['report_str'] + "\n")
        f.write("  [特征重要性权重]\n")
        f.write("  " + best_l2_res['importance_str'].replace('\n', '\n  ') + "\n\n")
        
    print(f"✅ 数据集 {dataset_name} 处理完成！log已同步至: {report_path}")

def main():
    for dataset in DATASET_LIST:
        process_single_dataset(dataset)

if __name__ == "__main__":
    main()