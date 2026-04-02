#!/usr/bin/env bash
set -euo pipefail

# mem_alpha del vs nodel on hotpotqa_dev (see BENCHMARK_TO_DATASET key hotpotqa_dev in src/pipeline_generate.py).
# Outputs / MemDB / traces are namespaced by benchmark so they do not mix with locomo runs.

benchmark=hotpotqa_dev
answer_model=Memalpha-4B
manager_model=Memalpha-4B
manager_max_new_tokens=8192
embedding_model=qwen3-embedding-8b
embedding_base_url=http://localhost:7110/v1/
embedding_api_key=zjj
method=mem_alpha
retrieve_topk=20
memory_token_limit=32768
memory_granularity=4

parallel_episodes=10
answer_concurrency=1
mem0_extract_concurrency=2
# Set to true to only build memory (no answers written).
store_memory_only=false

for mem_alpha_allow_delete in true false; do
  if [[ "$mem_alpha_allow_delete" == "true" ]]; then
    exp_tag="del"
    allow_delete_flag="--mem-alpha-allow-delete"
  else
    exp_tag="nodel"
    allow_delete_flag="--no-mem-alpha-allow-delete"
  fi

  output="experiment/${benchmark}_gran${memory_granularity}_${method}_${manager_model}_top${retrieve_topk}_${exp_tag}.jsonl"
  database_root="MemDB/${benchmark}_gran${memory_granularity}_${method}_${manager_model}_top${retrieve_topk}_${exp_tag}"
  agent_trace_dir="logs/answer_agent_trace/${benchmark}_${method}_${exp_tag}"
  memory_trace_dir="logs/memory_trace/${benchmark}_gran${memory_granularity}_${method}_${manager_model}_top${retrieve_topk}_${exp_tag}"

  echo "=== Running mem_alpha (${exp_tag}) ==="
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
    --mem-alpha-including-core \
    $allow_delete_flag \
    $( [[ "$store_memory_only" == "true" ]] && echo "--store-memory-only" )
done
