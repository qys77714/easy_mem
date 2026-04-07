# `exp_delete_mem0_del_compare.sh` 运行指导

## 脚本作用

脚本**依次**运行两种记忆方法，用于对比 **Mem0（含删除）** 与 **Mem0 不删除**：

| 方法 | 含义 |
|------|------|
| `mem0` | Mem0 默认行为（含删除相关逻辑） |
| `mem0_nodel` | 关闭删除的 Mem0 变体 |

每种方法的 **JSONL 输出、MemDB、日志/trace 路径**彼此独立，避免两次运行互相覆盖。

可选：通过环境变量 `RUN_ID` 为非空值时，路径后缀变为 `_${RUN_ID}`，便于重复实验不覆盖旧结果。未设置 `RUN_ID` 时，后缀固定为脚本内的 `_0406`。

---

## 运行前准备

1. **环境与依赖**（在项目根目录 `easy_mem`）
   - Python ≥ 3.12；推荐使用 `uv sync` 安装依赖（见主 `README.md`）。
   - 在仓库根目录执行脚本，保证 `python src/pipeline_generate.py` 可用。

2. **数据**
   - 默认 `--benchmark lme_s`，需要存在 `data/preprocessed/longmemeval_s_cleaned.json`。

3. **模型服务**
   - **对话 / Manager 模型**：脚本默认 `Qwen3-32B`（答题与管理记忆），需在 `src/utils/llm_api.py` 中对应 provider 可用，并在 `.env` 中配置好 `VLLM_BASE_URL`、`VLLM_API_KEY`（或云端密钥等），且服务已启动。
   - **Embedding**：脚本内写死了 `embedding_base_url` 与 `embedding_api_key`（默认指向本机 `http://localhost:7110/v1/`）。请保证 embedding 服务地址与密钥与脚本一致，或**直接修改脚本**中对应变量。仅改 `.env` 不会覆盖脚本里传入 `pipeline_generate.py` 的参数。

4. **可选**：参考 `script/0_run_embedding.sh`、`script/0_run_model.sh` 启动 vLLM。

---

## 如何执行

在项目根目录：

```bash
bash script/exp_delete_mem0_del_compare.sh
```

避免覆盖旧结果（路径后缀变为 `_v2` 等，不再使用默认的 `_0406`）：

```bash
RUN_ID=v2 bash script/exp_delete_mem0_del_compare.sh
```

---

## 脚本内可调参数

在 `script/exp_delete_mem0_del_compare.sh` 中直接编辑变量即可，常见项：

| 变量 | 作用 |
|------|------|
| `benchmark` | 基准名（默认 `lme_s`） |
| `answer_model` / `manager_model` | 答题与管理记忆的模型 |
| `embedding_model` / `embedding_base_url` / `embedding_api_key` | 向量模型与 API |
| `retrieve_topk`、`memory_token_limit`、`memory_granularity` | 检索与记忆预算、粒度 |
| `parallel_episodes`、`answer_concurrency`、`mem0_extract_concurrency` | 并行度 |
| `mem0_related_memory_aggregate_cap` | Mem0 合并后保留的旧记忆条数上限（score 截断相关，详见 `pipeline_generate.py --help`） |
| `store_memory_only` | 为 `true` 时只建记忆、不跑完整生成答题（更快） |
| `enable_qwen_thinking` | 为 `true` 时对 pipeline 传入 `--enable-qwen-thinking` |

---

## 输出路径说明（默认 `lme_s`、未设置 `RUN_ID`）

后缀为 `_0406` 时，示例如下（`{method}` 为 `mem0` 或 `mem0_nodel`）：

| 类型 | 路径 |
|------|------|
| 预测 JSONL | `experiment/lme_s_gran4_{method}_Qwen3-32B_top20_0406.jsonl` |
| 向量库根目录 | `MemDB/lme_s_gran4_{method}_Qwen3-32B_top20_0406/` |
| Agent trace | `logs/answer_agent_trace/lme_s_{method}_0406/` |
| Memory trace | `logs/memory_trace/lme_s_gran4_{method}_Qwen3-32B_top20_0406/` |

设置 `RUN_ID=v2` 时，将上述路径中的 `_0406` 换为 `_v2`。

---

## 跑完后评测

对两条 JSONL 可分别用 `pipeline_evaluate.py` / `script/2_evaluate.sh` 做 LLM Judge；Token 级指标可用 `pipeline_evaluate_f1.py`。详见主 `README.md` 中「评测」一节。
