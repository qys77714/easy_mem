#!/usr/bin/env bash
set -euo pipefail

# Compare mem0 (with delete) vs mem0_nodel (no delete) in one script.
# Outputs, MemDB, and traces use distinct paths per variant so runs do not mix.
#
# Optional run_id: non-empty appends _${run_id} to jsonl / MemDB / log paths so re-runs
# do not overwrite. Set in-script below, or: RUN_ID=v2 bash script/exp_delete_mem0_del_compare.sh

run_id="${RUN_ID:-}"

benchmark=lme_s
answer_model=Qwen3-32B
manager_model=Qwen3-32B
manager_max_new_tokens=8192
embedding_model=qwen3-embedding-8b
embedding_base_url=http://localhost:7110/v1/
embedding_api_key=zjj
retrieve_topk=20
memory_token_limit=32768
memory_granularity=4

parallel_episodes=100
answer_concurrency=1
mem0_extract_concurrency=2
# mem0 update: max distinct old memories after merging per-fact hits (score-truncated; see pipeline --help)
mem0_related_memory_aggregate_cap=10
# Set true to only build memory (faster); false for full generate + answer like script/1_generate.sh
store_memory_only=false
# Qwen3 Thinking: pass --enable-qwen-thinking to pipeline (chat_template_kwargs.enable_thinking)
enable_qwen_thinking=false

_run_suffix="_0406"
if [[ -n "$run_id" ]]; then
  _run_suffix="_${run_id}"
fi

for method in mem0 mem0_nodel; do
  output="experiment/${benchmark}_gran${memory_granularity}_${method}_${manager_model}_top${retrieve_topk}${_run_suffix}.jsonl"
  database_root="MemDB/${benchmark}_gran${memory_granularity}_${method}_${manager_model}_top${retrieve_topk}${_run_suffix}"
  agent_trace_dir="logs/answer_agent_trace/${benchmark}_${method}${_run_suffix}"
  memory_trace_dir="logs/memory_trace/${benchmark}_gran${memory_granularity}_${method}_${manager_model}_top${retrieve_topk}${_run_suffix}"

  echo "=== Running ${method} (run_id=${run_id:-<empty>}) ==="
  echo "output=${output}"
  echo "database_root=${database_root}"
  echo "memory_trace_dir=${memory_trace_dir}"

  python src/pipeline_generate.py \
    --benchmark "$benchmark" \
    --output "$output" \
    --answer_model "$answer_model" \
    --manager_model "$manager_model" \
    --manager_max_new_tokens "$manager_max_new_tokens" \
    --embedding_model "$embedding_model" \
    --embedding_base_url "$embedding_base_url" \
    --embedding_api_key "$embedding_api_key" \
    --method "$method" \
    --retrieve_topk "$retrieve_topk" \
    --memory_granularity "$memory_granularity" \
    --memory_token_limit "$memory_token_limit" \
    --database_root "$database_root" \
    --memory_trace_dir "$memory_trace_dir" \
    --agent_trace_dir "$agent_trace_dir" \
    --parallel_episodes "$parallel_episodes" \
    --answer-concurrency "$answer_concurrency" \
    --mem0-extract-concurrency "$mem0_extract_concurrency" \
    --mem0-related-memory-aggregate-cap "$mem0_related_memory_aggregate_cap" \
    $( [[ "$store_memory_only" == "true" ]] && echo "--store-memory-only" ) \
    $( [[ "$enable_qwen_thinking" == "true" ]] && echo "--enable-qwen-thinking" )
done
