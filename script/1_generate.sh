#!/usr/bin/env bash
set -euo pipefail

benchmark=locomo
answer_model=Memalpha-4B
manager_model=Memalpha-4B
manager_max_new_tokens=8192
embedding_model=qwen3-embedding-8b
embedding_base_url=http://localhost:7110/v1/
embedding_api_key=zjj
# Use mem_alpha for Mem-alpha-style memory (core/semantic/episodic + tools); or mem0 / mem0_nodel / amem
method=mem_alpha
retrieve_topk=20
memory_token_limit=32768
memory_granularity=4
output="experiment/${benchmark}_gran${memory_granularity}_${method}_${manager_model}_top${retrieve_topk}.jsonl"
agent_trace_dir="logs/answer_agent_trace"

parallel_episodes=10
answer_concurrency=1
mem0_extract_concurrency=2
mem_alpha_allow_delete=false

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
  --agent_trace_dir "$agent_trace_dir" \
  --parallel_episodes "$parallel_episodes" \
  --answer-concurrency "$answer_concurrency" \
  --mem0-extract-concurrency "$mem0_extract_concurrency" \
  --mem-alpha-including-core \
  --mem-alpha-allow-delete "$mem_alpha_allow_delete"
