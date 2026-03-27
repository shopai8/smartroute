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
# "Amazon", "BookReviews", "Genome", "Music", "Reviews", "Tiktok", "VariousImg", "Laion"
DATASET_LIST = ["Laion"] 
BASE_DIR = "/home/fengxiaoyao/FilterVector/FilterVectorResults"

ALGO_LIST = ['ACORN-gamma', 'ACORN-improved', 'NaviX', 'UNG-nTfalse', 'pre-filter']

MODELS_TO_TRY = ["RandomForest", "XGBoost", "LightGBM", "DecisionTree"]
USE_AUTO_ARENA = False 

# ==========================================
# 2. 核心功能函数
# ==========================================

def create_classifier(model_type="RandomForest", **kwargs):
    """
    1.根据传入的模型名称实例化不同的树模型分类器对象。
    2.思路说明：
      - 根据 model_type 字符串，加载对应的 sklearn、lightgbm 或 xgboost 库。
      - 注入默认的超参数以保证基本训练效率和防过拟合（处理不平衡分类）。
      - 允许通过 kwargs 动态更新超参数。
    3.输入参数：
      - model_type: str，含义（可选，选用的模型算法引擎名称，取值为"RandomForest", "XGBoost", "LightGBM", "DecisionTree"）
      - **kwargs: dict，含义（可选，用于动态覆盖默认的模型超参数）
    4.返回值类型和具体含义：
      - Classifier Object：返回已初始化的树模型或分类器实例。
    """
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
        # XGBoost 原生不支持多分类的 class_weight='balanced'
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
    """
    1.基于L2搜索耗时为数据集打上最优算法标签，并引入阈值过滤模糊样本。
    2.思路说明：
      - 找出所有召回率达标的候选算法，并按纯搜索时间从快到慢排序。
      - 计算第一名和第二名的性能差异百分比。
      - 如果差距小于 threshold，说明冠亚军难分伯仲，强制标记为 'Unknown' 以剔除该噪音样本。
    3.输入参数：
      - df: pd.DataFrame，含义（必填，宽表数据）
      - min_recall: float，含义（可选，最低召回率，默认 0.90）
      - threshold: float，含义（可选，第一名必须比第二名快出的百分比，默认 0.15 即 15%）
    4.返回值类型和具体含义：
      - pd.Series：最优算法标签序列，模糊样本将被标记为 'Unknown'。
    """
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
            
    print(f"  [Info] 阈值过滤 (Threshold={threshold*100}%): 共剔除 {fuzzy_count} 个冠亚军差距过小的模糊样本。")
    return pd.Series(best_algos, index=df.index)

def generate_naive_features(df):
    """
    1.提取用于路由模型训练的5个指定的原始核心特征。
    2.思路说明：
      - 直接从原始数据集中提取 QuerySize, CandSize, GlobalPpass, NumEntries, NumDescendants 5个核心量。
      - 对可能出现的正负无穷或 NaN 等异常数值进行清理填零。
    3.输入参数：
      - df: pd.DataFrame，含义（必填，原始路由统计数据集，需包含上述5个特征列）
    4.返回值类型和具体含义：
      - pd.DataFrame：清洗并提取好的纯特征集，供树模型直接拟合使用。
    """
    X = pd.DataFrame(index=df.index)
    X['QuerySize'] = df['QuerySize']
    X['CandSize'] = df['CandSize']
    X['GlobalPpass'] = df['GlobalPpass']
    X['NumEntries'] = df['NumEntries']
    X['NumDescendants'] = df['NumDescendants']
    
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(0, inplace=True)
    return X

def train_and_evaluate(X, y, target_map, model_type):
    """
    1.在纯净数据集上训练指定的树模型，完成评估并分别统计训练和预测耗时。
    2.思路说明：
      - 过滤异常标签，剔除样本少于 2 的极稀有类别防止分层拆分崩溃。
      - 划分 80/20 训练与测试集。
      - 兼容 XGBoost 的连续 ID 要求。
      - 分别利用 perf_counter 捕捉 fit (训练) 和 predict (推理) 的纯粹耗时。
    3.输入参数：
      - X: pd.DataFrame，含义（必填，模型训练的特征集）
      - y: pd.Series，含义（必填，目标路由算法标签）
      - target_map: dict，含义（必填，算法文本名称到整形绝对索引的字典映射）
      - model_type: str，含义（必填，选用的模型算法引擎名称）
    4.返回值类型和具体含义：
      - dict: 返回涵盖准确率、训练耗时、预测耗时、分类报告、绝对标签集以及预测结果明细等综合信息的字典。
    """
    y_mapped = y.map(target_map)
    valid_mask = y_mapped.notna()
    X_clean, y_clean = X[valid_mask].copy(), y_mapped[valid_mask].astype(int).copy()
    
    class_counts = y_clean.value_counts()
    if class_counts.min() < 2:
        valid_classes = class_counts[class_counts >= 2].index
        dropped_classes = class_counts[class_counts < 2].to_dict()
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
    
    # 统计训练耗时
    t_train_start = time.perf_counter()
    classifier.fit(X_train.values, y_train)
    train_time_ms = (time.perf_counter() - t_train_start) * 1000.0

    # 统计预测耗时 (针对整个测试集)
    t_pred_start = time.perf_counter()
    y_pred = classifier.predict(X_test.values)
    pred_time_ms = (time.perf_counter() - t_pred_start) * 1000.0
    
    y_test_orig = y_test.map(internal_to_original)
    y_pred_orig = pd.Series(y_pred, index=X_test.index).map(internal_to_original)
        
    acc = accuracy_score(y_test_orig, y_pred_orig)
    
    inv_map = {v: k for k, v in target_map.items()}
    present_labels = sorted(y_test_orig.unique())
    target_names = [inv_map[l] for l in present_labels]
    
    cls_report = classification_report(y_test_orig, y_pred_orig, labels=present_labels, target_names=target_names, zero_division=0)
    conf_mat = confusion_matrix(y_test_orig, y_pred_orig, labels=present_labels)
    cm_df = pd.DataFrame(conf_mat, index=[f"True_{name}" for name in target_names], columns=[f"Pred_{name}" for name in target_names])
    
    if hasattr(classifier, 'feature_importances_'):
        imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': classifier.feature_importances_})
        imp_df = imp_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)
        importance_str = imp_df.to_string(formatters={'Importance': '{:.4f}'.format})
    else:
        importance_str = "当前模型不支持直接提取特征重要性。"
        
    return {
        "acc": acc,
        "train_time_ms": train_time_ms,
        "pred_time_ms": pred_time_ms,
        "test_size": len(X_test),
        "cls_report": cls_report,
        "cm_df": cm_df,
        "importance_str": importance_str,
        "classifier": classifier,
        "params": classifier.get_params(),
        "pred_counts": y_pred_orig.value_counts(),
        "real_classes": real_classes,
        "test_indices": X_test.index,
        "y_test_orig": y_test_orig,
        "y_pred_orig": y_pred_orig
    }
    
def run_naive_ablation_study(X, y, target_map, best_model_type):
    """
    1.对保留的5个核心特征进行逐一消融实验，评估特征单体重要性。
    2.思路说明：
      - 每次遍历移除单个特定特征，触发重新训练与评估。
      - 记录每次消融后的准确率结果并返回。
    3.输入参数：
      - X: pd.DataFrame，含义（必填，原始5个特征的特征集）
      - y: pd.Series，含义（必填，目标标签）
      - target_map: dict，含义（必填，标签映射字典）
      - best_model_type: str，含义（必填，选定的最优基座模型类型）
    4.返回值类型和具体含义：
      - dict: 返回消融实验结果字典，键为实验配置名，值为模型测试集准确率。
    """
    print(f"--- 正在运行特征消融实验 (基座: {best_model_type}) ---")
    valid_mask = y.notna() & (y != 'Unknown')
    X_clean, y_clean = X[valid_mask], y[valid_mask]
    if len(X_clean) == 0: return {}
    
    all_features = list(X_clean.columns)
    def get_ablated_features(feature_to_remove):
        return [f for f in all_features if f != feature_to_remove]

    configs = {
        "完全体 (All 5 Features)": all_features,
        "消融: 移除 QuerySize": get_ablated_features("QuerySize"),
        "消融: 移除 CandSize": get_ablated_features("CandSize"),
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
    """
    1.导出训练好的模型至ONNX格式，并硬编码覆盖底层节点解决标签映射脱节。
    2.思路说明：
      - 根据模型类型调用相应的 ONNX 转换工具。
      - 如果提供了真实标签映射，通过修改 ONNX 模型的 graph 节点属性强制烧录分类 ID。
    3.输入参数：
      - classifier: Object，含义（必填，已被拟合好的模型对象）
      - model_type: str，含义（必填，选用的模型算法引擎名称）
      - num_features: int，含义（必填，特征维度数）
      - output_dir: str，含义（必填，导出目录）
      - real_classes: list，含义（可选，绝对类别序列）
      - filename: str，含义（可选，导出的 ONNX 文件名称）
    4.返回值类型和具体含义：
      - None: 本地文件IO操作。
    """
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
            print(f"⚠️ 无法导出 {model_type} 的 ONNX 模型，可能缺少依赖库。")
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
                
        print(f"✅ 成功导出修改完成的 ONNX 模型: {onnx_filename}")
    except Exception as e:
        print(f"❌ 导出 ONNX 模型时发生错误: {e}")

# ==========================================
# 3. 主干执行流
# ==========================================
def process_single_dataset(dataset_name, use_auto_arena=True):
    """
    1.调度单个数据集的完整生命周期，并在终端和文件中生成多算法对比指标表。
    2.思路说明：
      - 预处理特征和标签后，遍历跑完所有参与竞技的模型，同时收集耗时与精度指标。
      - 构建性能对比表格单例化存储。
      - 根据 use_auto_arena 标志位，决定是仅将最优模型落盘，还是批量输出所有模型的产物。
    3.输入参数：
      - dataset_name: str，含义（必填，数据集名称）
      - use_auto_arena: bool，含义（可选，是否启用自动竞技并仅保留最优模型）
    4.返回值类型和具体含义：
      - None: 核心逻辑调度控制。
    """
    print(f"\n{'='*60}")
    print(f"🚀 开始处理数据集: {dataset_name} | 模式: {'自动选优' if use_auto_arena else '全量输出'}")
    print(f"{'='*60}")
    
    csv_path = os.path.join(BASE_DIR, "EDA_Plots", dataset_name, f"{dataset_name}_aligned_results.csv")
    output_dir = os.path.join(BASE_DIR, dataset_name, "SelectModels", "naive_smart_route_2")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"❌ 找不到数据文件: {csv_path}，跳过。")
        return
        
    df = pd.read_csv(csv_path)
    total_queries = len(df)
    
    df['Best_Algo'] = label_best_algorithm(df, min_recall=0.90, threshold=0.20)
    X_All = generate_naive_features(df)
    
    TARGET_MAP = {
        'UNG-nTfalse': 0, 'UNG-nTtrue': 1, 'ACORN-gamma': 2, 
        'ACORN-improved': 3, 'NaviX': 4, 'pre-filter': 5
    }

    # 用于收集各模型表现，生成表格
    model_metrics = []
    model_results_cache = {}

    print("\n[模型全量评测中...]")
    for model_type in MODELS_TO_TRY:
        try:
            res = train_and_evaluate(X_All, df['Best_Algo'], TARGET_MAP, model_type)
            model_results_cache[model_type] = res
            
            model_metrics.append({
                "Model": model_type,
                "Accuracy": res['acc'],
                "Train_Time_ms": res['train_time_ms'],
                "Pred_Time_ms": res['pred_time_ms']
            })
            print(f"  > 跑通: {model_type:<15}")
        except ImportError:
            print(f"  > ⚠️ 缺少 {model_type} 库。")
        except Exception as e:
            print(f"  > ❌ 模型 {model_type} 评估失败: {e}")

    if not model_metrics:
        print("❌ 所有模型训练失败。")
        return

    # 动态生成 ASCII 比较表格
    table_lines = []
    table_lines.append(f"  {'算法模型(Model)':<16} | {'准确率(Accuracy)':<16} | {'训练耗时(Train ms)':<18} | {'预测耗时(Pred ms)':<18}")
    table_lines.append("  " + "-" * 76)
    
    # 按照准确率从高到低排序一下表格展示
    model_metrics.sort(key=lambda x: x["Accuracy"], reverse=True)
    
    for m in model_metrics:
        table_lines.append(f"  {m['Model']:<16} | {m['Accuracy']:<18.4%} | {m['Train_Time_ms']:<18.2f} | {m['Pred_Time_ms']:<18.2f}")
    
    comparison_table_str = "\n".join(table_lines)
    print("\n[所有模型性能大比武 (Model Comparison Summary)]")
    print(comparison_table_str)

    # 找出最佳模型
    best_metric = model_metrics[0] 
    best_model_type = best_metric['Model']

    # ----- 内部复用函数：集中处理产物的保存逻辑 -----
    def generate_and_save_artifacts(model_type, res, ablation_results, suffix=""):
        onnx_name = f"naive_smart_route{suffix}.onnx"
        save_onnx_model(res['classifier'], model_type, X_All.shape[1], output_dir, res.get('real_classes'), filename=onnx_name)
        
        report_path = os.path.join(output_dir, f"NaiveRoute_Report_{dataset_name}{suffix}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓\n")
            f.write("┃               NaiveRoute                    ┃\n")
            f.write("┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫\n")
            f.write(f"┃ 基座模型: {model_type:<16}  数据集: {dataset_name:<15}            ┃\n")
            f.write(f"┃ 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<40} ┃\n")
            f.write("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛\n\n")
            
            # --- 新增的全局对比表格注入到报告开头 ---
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
            for algo in ALGO_LIST + ['Unknown']:
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
            f.write(f"  - 测试集预测总耗时 : {res['pred_time_ms']:.2f} ms\n")
            latency_us = (res['pred_time_ms'] * 1000) / res['test_size']
            f.write(f"  - 单条平均推理延迟 : ~{latency_us:.2f} μs / query\n\n")
            
            if ablation_results:
                base_acc_ab = ablation_results.get("完全体 (All 5 Features)", 0)
                f.write("【叁 | 特征消融实验评估 (Ablation Study on 5 Core Indicators)】\n")
                f.write("-" * 80 + "\n")
                f.write("  (注: 性能衰减越大，说明被移除的特征越重要。出现负数说明移除该特征反而提分)\n\n")
                f.write(f"  {'配置方案':<35} | {'预测准确率':<12} | {'性能衰减':<12}\n")
                f.write("  " + "-" * 65 + "\n")
                for config, acc_ab in ablation_results.items():
                    drop = base_acc_ab - acc_ab
                    drop_str = f"↓ {drop:.2%}" if drop > 0 else f"↑ {abs(drop):.2%}"
                    if config == "完全体 (All 5 Features)": drop_str = "Baseline"
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
        print(f"✅ 详尽日志已生成: {report_path}")

        if not use_auto_arena:
            pred_df = pd.DataFrame({
                'True_Algo': res['y_test_orig'],
                'Pred_Algo': res['y_pred_orig']
            }, index=res['test_indices'])
            pred_csv_path = os.path.join(output_dir, f"Predictions_{dataset_name}{suffix}.csv")
            pred_df.to_csv(pred_csv_path)
            print(f"✅ 预测明细已生成: {pred_csv_path}")

    # --- 开始落盘 ---
    if use_auto_arena:
        print(f"\n🏆 自动选优: 胜出模型 {best_model_type}")
        best_res = model_results_cache[best_model_type]
        ablation_results = run_naive_ablation_study(X_All, df['Best_Algo'], TARGET_MAP, best_model_type)
        generate_and_save_artifacts(best_model_type, best_res, ablation_results, suffix="")
    else:
        print("\n[导出全部模型的预测及分析结果]")
        for m_type, m_res in model_results_cache.items():
            print(f"\n--- 正在落盘模型: {m_type} ---")
            ablation_results = run_naive_ablation_study(X_All, df['Best_Algo'], TARGET_MAP, m_type)
            generate_and_save_artifacts(m_type, m_res, ablation_results, suffix=f"_{m_type}")

def main():
    """
    1.脚本执行的主入口，遍历目标数据集列表串行处理。
    2.思路说明：
      - 依序遍历全局设定的 DATASET_LIST。
      - 通过 USE_AUTO_ARENA 控制产出物格式（取优或全量）。
    3.输入参数：
      - 无。
    4.返回值类型和具体含义：
      - None
    """  
    for dataset in DATASET_LIST:
        process_single_dataset(dataset, use_auto_arena=USE_AUTO_ARENA)

if __name__ == "__main__":
    main()