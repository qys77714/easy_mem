from __future__ import annotations

import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union

import numpy as np

from benchmark.base import ChatSession
from memory.base import BaseMemorySystem, RetrievedMemory, _session_progress_tick
from prompts import render_prompt
from .memory_core import AlphaMemory
from .tools import execute_tool, get_memory_tool_schemas
from memory.tracing import MemoryTraceLogger

if TYPE_CHECKING:
    from openai import OpenAI

logger = logging.getLogger(__name__)

_STATE_NAME = "mem_alpha_state.json"
_EMB_NAME = "mem_alpha_emb.npz"


class MemAlphaMemorySystem(BaseMemorySystem):
    """Mem-alpha style core/semantic/episodic memory with tool-based writes and BM25/embedding retrieval."""

    def __init__(
        self,
        embed_model_name: str,
        llm_client=None,
        embed_client: Optional["OpenAI"] = None,
        database_root: Optional[str] = None,
        related_memory_top_k: int = 20,
        language: str = "en",
        granularity: Union[str, int] = "all",
        trace_log_dir: Optional[str] = None,
        including_core: bool = False,
        search_method: str = "bm25",
        dialogue_format: str = "user_assistant",
        manager_max_new_tokens: int = 2048,
        allow_memory_delete: bool = True,
    ) -> None:
        super().__init__(
            embed_client=embed_client,
            embed_model_name=embed_model_name,
            llm_client=llm_client,
            database_root=database_root,
        )
        if llm_client is None:
            raise ValueError("mem_alpha requires llm_client")
        if embed_client is None:
            raise ValueError("mem_alpha requires embed_client")

        self.related_memory_top_k = max(1, int(related_memory_top_k))
        self.language = language
        self.granularity = self._parse_granularity(granularity)
        self._including_core = bool(including_core)
        sm = (search_method or "bm25").strip().lower()
        if sm in ("embedding", "text-embedding"):
            self._search_method = "text-embedding"
        else:
            self._search_method = "bm25"
        df = (dialogue_format or "user_assistant").strip().lower()
        if df not in ("user_assistant", "named_speakers"):
            raise ValueError("dialogue_format must be 'user_assistant' or 'named_speakers'")
        self.dialogue_format = df
        self._manager_max_new_tokens = max(1, int(manager_max_new_tokens))
        self._allow_memory_delete = bool(allow_memory_delete)

        self.trace = MemoryTraceLogger(
            method="mem_alpha",
            log_dir=trace_log_dir or "logs/memory_trace",
            use_experiment_naming=trace_log_dir is not None,
        )
        self._memory_by_ns: Dict[str, AlphaMemory] = {}
        # pipeline_generate may run multiple episodes in parallel threads sharing this instance;
        # the sync OpenAI client / httpx stack is not safe for concurrent requests — vLLM can return
        # HTTP 400 "Already borrowed" (Rust RefCell). Serialize manager chat completions.
        self._manager_http_lock = threading.Lock()

    @staticmethod
    def _parse_granularity(granularity: Union[str, int]) -> Union[str, int]:
        if isinstance(granularity, str):
            g = granularity.strip().lower()
            if g == "all":
                return "all"
            if g.isdigit():
                granularity = int(g)
            else:
                raise ValueError("mem_alpha granularity must be 'all' or a positive integer.")
        if isinstance(granularity, int) and granularity > 0:
            return granularity
        raise ValueError("mem_alpha granularity must be 'all' or a positive integer.")

    def _embed_one(self, text: str) -> np.ndarray:
        from utils.embed_utils import embed_texts

        return embed_texts(self.embed_client, [text], self.embed_model_name)[0]

    def _state_path(self, history_name: str) -> Path:
        return self.episode_storage_path(history_name) / _STATE_NAME

    def _emb_path(self, history_name: str) -> Path:
        return self.episode_storage_path(history_name) / _EMB_NAME

    def _save_memory(self, history_name: str, memory: AlphaMemory) -> None:
        root = self.episode_storage_path(history_name)
        root.mkdir(parents=True, exist_ok=True)
        payload = memory.to_serializable()
        payload["embed_model_name"] = self.embed_model_name
        self._state_path(history_name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.savez_compressed(
            self._emb_path(history_name),
            semantic_matrix=memory.semantic_embedding_matrix,
            episodic_matrix=memory.episodic_embedding_matrix,
        )

    def _load_memory(self, history_name: str) -> AlphaMemory:
        path = self._state_path(history_name)
        if not path.is_file():
            m = AlphaMemory(
                including_core=self._including_core,
                embed_one=self._embed_one,
            )
            self._memory_by_ns[history_name] = m
            return m

        data = json.loads(path.read_text(encoding="utf-8"))
        m = AlphaMemory.from_serializable(data, embed_one=self._embed_one)
        emb_p = self._emb_path(history_name)
        if emb_p.is_file():
            z = np.load(emb_p)
            sm = z["semantic_matrix"]
            em = z["episodic_matrix"]
            m.semantic_embedding_matrix = np.asarray(sm, dtype=np.float32)
            m.episodic_embedding_matrix = np.asarray(em, dtype=np.float32)
        saved_model = data.get("embed_model_name")
        if saved_model and saved_model != self.embed_model_name:
            m.rebuild_embeddings_from_content()
        elif (
            m.semantic_embedding_matrix.shape[0] != len(m.semantic_embedding_ids)
            or m.episodic_embedding_matrix.shape[0] != len(m.episodic_embedding_ids)
        ):
            m.rebuild_embeddings_from_content()

        self._memory_by_ns[history_name] = m
        return m

    def _get_memory(self, history_name: str) -> AlphaMemory:
        if history_name not in self._memory_by_ns:
            self._load_memory(history_name)
        return self._memory_by_ns[history_name]

    def episode_storage_path(self, history_name: str) -> Optional[Path]:
        return self.persisted_data_root() / history_name

    def clear(self, history_name: str) -> None:
        if history_name in self._memory_by_ns:
            del self._memory_by_ns[history_name]
        root = self.episode_storage_path(history_name)
        if root and root.exists():
            shutil.rmtree(root)

    def _iter_turn_chunks(self, turns: List[Any]) -> List[Tuple[Optional[int], Optional[int], List[Any]]]:
        if self.granularity == "all":
            return [(None, None, turns)]
        chunk_size = int(self.granularity)
        chunks: List[Tuple[Optional[int], Optional[int], List[Any]]] = []
        for i in range(0, len(turns), chunk_size):
            chunk_turns = turns[i : i + chunk_size]
            if chunk_turns:
                chunks.append((i, i + len(chunk_turns) - 1, chunk_turns))
        return chunks

    @staticmethod
    def _turn_time_for_prompt(turn: Any, session: ChatSession) -> str:
        t = getattr(turn, "time", None)
        if isinstance(t, str) and t.strip():
            return t.strip()
        return str(session.session_date).strip() if session.session_date else ""

    def _build_chunk_transcript(self, session: ChatSession, chunk_turns: List[Any]) -> str:
        lines: List[str] = []
        for turn in chunk_turns:
            content = turn.content.strip()
            if not content:
                continue
            time_str = self._turn_time_for_prompt(turn, session)
            time_prefix = f"[{time_str}] " if time_str else ""
            if self.dialogue_format == "named_speakers":
                speaker = (turn.speaker or "Unknown").strip()
                lines.append(f"{time_prefix}{speaker}: {content}")
                continue
            role = turn.speaker.lower()
            if role not in ("user", "assistant"):
                role = "user" if role in ("human", "人") else "assistant"
            lines.append(f"{time_prefix}{role}: {content}")
        return "\n".join(lines)

    def _run_memorie_chunk(
        self,
        history_name: str,
        memory: AlphaMemory,
        transcript: str,
        trace_scope_id: Optional[str],
        trace: MemoryTraceLogger,
    ) -> None:
        if not transcript.strip():
            return

        # User message: same wrapping as Mem-alpha RL loop (unified_prompt in
        # Mem-alpha/config/prompts_wrt_datasource.yaml), not raw transcript only.
        max_reply = max(1, int(self._manager_max_new_tokens * 0.8))
        user_content = render_prompt(
            "mem_alpha_memorie_user.jinja",
            context=transcript,
            max_new_tokens=max_reply,
        )

        messages = memory.render_system_prompt_memorie()
        messages.append({"role": "user", "content": user_content})

        tools = get_memory_tool_schemas(
            memory,
            allow_memory_delete=self._allow_memory_delete,
        )
        if not tools:
            logger.warning("mem_alpha: no tools available for memory configuration")
            return

        client = getattr(self.llm_client, "client", None)
        if client is None:
            raise RuntimeError("llm_client must expose .client for tool calling (OpenAI compatible).")
        model_name = getattr(self.llm_client, "model_name", None)
        if not model_name:
            raise RuntimeError("llm_client must expose .model_name")

        from utils.openai_client import _extra_body_disable_qwen_thinking

        # Do not set tool_choice="auto": vLLM rejects it unless the server is started with
        # --enable-auto-tool-choice and --tool-call-parser (see vLLM OpenAI server docs).
        # Omitting tool_choice lets compatible servers accept the request; behavior matches
        # OpenAI default (model may emit tool calls when tools are present).
        create_kw: Dict[str, Any] = dict(
            model=model_name,
            messages=messages,
            tools=tools,
            max_tokens=self._manager_max_new_tokens,
            temperature=0.0,
        )
        if model_name not in ("gpt-4o-mini",):
            create_kw["extra_body"] = _extra_body_disable_qwen_thinking(None)
        with self._manager_http_lock:
            completion = client.chat.completions.create(**create_kw)

        msg = completion.choices[0].message
        raw_text = msg.content or ""
        trace.log_llm_interaction(
            purpose="mem_alpha_memorie",
            messages=messages,
            response=raw_text,
            scope_id=trace_scope_id,
            metadata={"tool_calls": bool(msg.tool_calls)},
        )

        if not msg.tool_calls:
            return

        for tc in msg.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(name, memory, args)
            trace.log_llm_interaction(
                purpose="mem_alpha_tool",
                messages=[{"role": "tool", "name": name, "content": json.dumps(args)}],
                response=json.dumps(result, ensure_ascii=False),
                scope_id=trace_scope_id,
                metadata={"tool": name},
            )
            self._save_memory(history_name, memory)

    def store_session(self, history_name: str, session_idx: int, session: ChatSession) -> None:
        self.store_episode(history_name, [session], session_progress=None)

    def store_episode(
        self,
        history_name: str,
        sessions: List[ChatSession],
        *,
        session_progress: Optional[Any] = None,
    ) -> None:
        if not sessions:
            return
        trace = self.trace.get_logger_for(history_name)
        memory = self._get_memory(history_name)

        for session_idx, session in enumerate(sessions, start=1):
            session_scope = trace.create_scope(
                "mem_alpha_store_session",
                metadata={
                    "history_name": history_name,
                    "session_idx": session_idx,
                    "session_date": str(session.session_date),
                },
            )
            chunks = self._iter_turn_chunks(session.turns)
            for turn_start, turn_end, chunk_turns in chunks:
                transcript = self._build_chunk_transcript(session, chunk_turns)
                chunk_scope = trace.create_scope(
                    "mem_alpha_chunk",
                    parent_scope_id=session_scope,
                    metadata={
                        "turn_start": turn_start,
                        "turn_end": turn_end,
                    },
                )
                self._run_memorie_chunk(
                    history_name,
                    memory,
                    transcript,
                    chunk_scope,
                    trace,
                )
                trace.close_scope(chunk_scope, status="ok")
            trace.close_scope(session_scope, status="ok")
            _session_progress_tick(session_progress, 1)

        self._save_memory(history_name, memory)

    def retrieve(
        self,
        history_name: str,
        query: str,
        current_time: str,
        top_k: int = 5,
    ) -> List[RetrievedMemory]:
        _ = current_time
        memory = self._get_memory(history_name)
        k = max(1, int(top_k))
        sm = self._search_method

        out: List[RetrievedMemory] = []
        slots_for_search = k
        if memory.including_core and memory.core:
            out.append(
                RetrievedMemory(
                    memory_id="core:0",
                    text=memory.core,
                    source_index="mem_alpha:core",
                    time="",
                    score=1.0,
                    metadata={"memory_type": "core"},
                )
            )
            slots_for_search = max(0, k - 1)

        merged: List[Tuple[str, str, float, str]] = []
        if slots_for_search > 0:
            sem = memory.memory_search(
                "semantic", query, top_k=slots_for_search, search_method=sm
            )
            epi = memory.memory_search(
                "episodic", query, top_k=slots_for_search, search_method=sm
            )
            for d, score in sem:
                for mid, text in d.items():
                    merged.append(
                        (f"semantic:{mid}", text, float(score), f"mem_alpha:semantic:{mid}")
                    )
            for d, score in epi:
                for mid, text in d.items():
                    merged.append(
                        (f"episodic:{mid}", text, float(score), f"mem_alpha:episodic:{mid}")
                    )
            merged.sort(key=lambda x: x[2], reverse=True)
            merged = merged[:slots_for_search]

        for mem_id, text, score, src in merged:
            out.append(
                RetrievedMemory(
                    memory_id=mem_id,
                    text=text,
                    source_index=src,
                    time="",
                    score=score,
                    metadata={"memory_type": mem_id.split(":")[0]},
                )
            )

        return out[:k]

    def format_retrieved_for_context(
        self, retrieved: List[RetrievedMemory], language: str = "zh"
    ) -> str:
        if not retrieved:
            template = "agent_context_empty_zh.jinja" if language == "zh" else "agent_context_empty_en.jinja"
            return render_prompt(template)

        unit_template = "mem0_context_unit_zh.jinja" if language == "zh" else "mem0_context_unit_en.jinja"
        context_lines = [
            render_prompt(
                unit_template,
                index=idx + 1,
                text=item.text,
                time=item.time,
                metadata=item.metadata or {},
            )
            for idx, item in enumerate(retrieved)
        ]
        return "\n\n".join(context_lines)
