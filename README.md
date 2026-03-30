# FilterVectorCode

本仓库是一个针对 **过滤式近似最近邻搜索 (Filtered ANNS)** 的增强实现。它集成了 **UNG (Unified Navigating Graph)** 与 **ACORN** 的核心思想，并引入了全新的 **AI 智能路由 (SmartRoute & FastSmartRoute)** 机制，通过机器学习模型实现算法的自适应调度。

## 1. 核心算法介绍

### 1.1 UNG & ACORN 基础
* **UNG (Unified Navigating Graph)**: 基于向量分组和标签导航图（LNG）的“先过滤后搜索”架构。
* **ACORN**: 通过构建稠密图，使索引在无需预知过滤条件的情况下具备良好的谓词子图连通性。

### 1.2 智能路由算法
为了解决固定算法在不同查询选择性（Selectivity）下性能不稳的问题，本框架引入了自动路由层：

#### **SmartRoute (高召回决策模式)**
* **特点**: **Recall 极高，但 QPS 相对较低**。
* **逻辑**: 该模式会首先调用 `get_min_super_sets` 计算查询的 **ELS (Entry Label Sets)**。然后使用三个指标GlobalPpass，NumEntries，NumDescendants进行预测。
* **开销原因**: 决策特征中包含 ELS 及其在 LNG 中的后代信息（NumDescendants），这需要完整的 Trie 树遍历，虽然能提供最精准的入口点引导，但会增加查询初期的计算开销。
* **代码**：FilterVectorCode/DataTools/idea-selector/naive_smart_route_train.py

#### **FastSmartRoute (极速级联模式)**
* **特点**: 对 SmartRoute 的重大改进，旨在提升 QPS。
* **逻辑**: 采用 **级联决策 (Cascade Strategy)**：
    1.  **L1 拦截层**: 仅使用 `GlobalPpass` 轻量特征。若判定为 `pre-filter` 或 `ACORN` 路径，则**直接跳过**昂贵的 ELS 计算。
    2.  **按需计算**: 只有当 L1 判定查询必须走 UNG 复杂路径时，才会触发 ELS 计算并进入 L2 裁判层进行精细调度。
* **代码**：FilterVectorCode/DataTools/idea-selector/fast_smart_route_train.py

---

## 2. 运行方式详细说明

本项目通过 `exp.sh` 脚本驱动自动化的实验流水线。

### 2.1 依赖准备
* **环境**: C++17 编译器、Faiss、ONNX Runtime (用于模型推理)。
* **工具**: 必须安装 `jq` 用于解析 JSON 配置文件。

### 2.2 启动实验
```bash
# 格式: ./exp.sh [配置文件.json]
./exp.sh experiments.json

# 批量实验的时候可以使用final_exp.sh
# 记得修改final_exp.sh中指定运行的json，从而控制运行什么实验
```

### 2.3 实验流程
1.  **解析配置**: `exp.sh` 读取 JSON 中的 `experiments` 数组。
2.  **索引构建**: 调用 `build_hybrid.sh`。支持 `parallel` 模式，可同时构建 UNG 组内图和 ACORN 索引。实验的时候设置成serial即可。
3.  **GT 生成**: 调用 `generate_gt.sh` 计算真实最近邻（用于评估召回率）。
4.  **执行搜索**: 调用 `search.sh`，加载预训练的 ONNX 路由器模型进行在线调度。

---

## 3. 核心参数 glossary

在配置文件或脚本中，以下参数决定了算法的行为：

| 参数名称 | 核心作用 | 可选值/说明 |
| :--- | :--- | :--- |
| `ROUTING_MODE` | **决定路由逻辑** | `0`: Baseline 模式；`1`: **SmartRoute**；`2`: **FastSmartRoute**。 |
| `BASELINE_ALG` | 当 `MODE=0` 时的算法 | `0`: UNG-nTfalse；`1`: UNG-nTtrue；`5`: pre-filter；`7`: NaviX。 |
| `BUILD_MODE` | 索引构建模式 | `parallel`: 并行构建全量索引；`acorn_only`: 仅构建 ACORN。 |
| `Lsearch` | UNG 搜索参数 | 类似于 HNSW 的 `efSearch`，控制搜索深度。 |
| `efs_start/step` | ACORN 搜索参数 | 用于在搜索过程中动态调整 ACORN 的过滤强度。 |

---

## 4. 模型训练与部署

智能路由的决策模型通过 Python 脚本进行训练：
* **训练**: 使用 `naive_smart_route_train.py` 或 `fast_smart_route_train.py`。
* **部署**: 模型需导出为 `.onnx` 格式并放置于 `SelectModels` 目录下，供 C++ 端的 `MethodSelector` 调用。

## 5. 实验结果分析
实验完成后，结果存储在 `FilterVectorResults` 中：
* `query_details.csv`: 记录了每个查询的 `Algo_Choice`（路由选择结果）以及 `Routing_TotalT_ms`（决策开销）。
* 通过分析 `Algo_Choice` 的分布，可以直观观察到路由器在面对不同 `GlobalPpass`（选择性）时是如何在 UNG 和 ACORN 之间进行切换的。
