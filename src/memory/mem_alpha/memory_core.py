"""Mem-alpha style memory store: core / semantic / episodic with BM25 + embedding search."""

from __future__ import annotations

import re
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from prompts import render_prompt

try:
    import tiktoken

    def count_tokens_memalpha(text: str, model: str = "gpt-4o-mini") -> int:
        if not isinstance(text, str):
            text = str(text)
        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))

except Exception:  # pragma: no cover

    def count_tokens_memalpha(text: str, model: str = "gpt-4o-mini") -> int:
        if not isinstance(text, str):
            text = str(text)
        return max(1, len(text) // 4)


def _cosine_similarity_query_matrix(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """query: (d,), matrix: (n, d) -> (n,) cosine similarities."""
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    q = query.astype(np.float64)
    qn = np.linalg.norm(q)
    if qn < 1e-12:
        return np.zeros((matrix.shape[0],), dtype=np.float32)
    q = q / qn
    norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    norms = np.maximum(norms, 1e-12)
    rows = matrix.astype(np.float64) / norms[:, np.newaxis]
    return (rows @ q).astype(np.float32)


class AlphaMemory:
    """Holds core, semantic, episodic memories (Mem-alpha compatible)."""

    MAX_MEMORY_ITEMS = 10

    def __init__(
        self,
        including_core: bool = False,
        disabled_memory_types: Optional[List[str]] = None,
        embed_one: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        disabled_memory_types = disabled_memory_types or []
        normalized_disabled = {mem_type.lower() for mem_type in disabled_memory_types}
        invalid = normalized_disabled - {"core", "semantic", "episodic"}
        if invalid:
            raise ValueError(f"Invalid memory types to disable: {', '.join(sorted(invalid))}")

        self.disabled_memory_types = normalized_disabled
        including_core = including_core and "core" not in self.disabled_memory_types

        if including_core:
            self.core: str = ""
        else:
            self.core = None
        self.instructions = None
        self.semantic: List[Dict[str, str]] = []
        self.episodic: List[Dict[str, str]] = []
        self.semantic_embedding_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.episodic_embedding_matrix: np.ndarray = np.empty((0, 0), dtype=np.float32)
        self.semantic_embedding_ids: List[str] = []
        self.episodic_embedding_ids: List[str] = []
        self.including_core = including_core
        self._embed_one = embed_one

    def is_memory_type_enabled(self, memory_type: str) -> bool:
        memory_type = memory_type.lower()
        if memory_type == "core":
            return self.including_core
        if memory_type in {"semantic", "episodic"}:
            return memory_type not in self.disabled_memory_types
        raise ValueError(f"Unknown memory type: {memory_type}")

    def _ensure_memory_type_enabled(self, memory_type: str) -> None:
        if not self.is_memory_type_enabled(memory_type):
            raise ValueError(f"{memory_type.capitalize()} memory is disabled for this run.")

    def _get_embedding(self, text: str) -> np.ndarray:
        if self._embed_one is None:
            raise RuntimeError("embed_one must be set for embedding operations.")
        vec = np.asarray(self._embed_one(text), dtype=np.float32).reshape(-1)
        return vec

    def _generate_memory_id(self) -> str:
        return str(uuid.uuid4())[:4]

    def _content_exists(self, memory_type: str, content: str) -> bool:
        if memory_type == "core":
            return self.core == content if self.core is not None else False
        mem_list = getattr(self, memory_type)
        for mem in mem_list:
            for _, existing_content in mem.items():
                if existing_content == content:
                    return True
        return False

    def _block(
        self,
        title: str = "",
        lines: Optional[List[Dict[str, str]]] = None,
        content: Optional[str] = None,
    ) -> str:
        lines = lines or []
        if content is not None:
            if title:
                return f"<{title}>\n{content}\n</{title}>"
            return content

        if not lines:
            if title:
                return f"<{title}>\nEmpty.\n</{title}>"
            return "Empty."

        formatted_lines = []
        for mem in lines:
            for mem_id, c in mem.items():
                formatted_lines.append(f"[{mem_id}] {c}")

        body = "\n".join(formatted_lines)
        if title:
            return f"<{title}>\n{body}\n</{title}>"
        return body

    def render_system_prompt_memorie(
        self,
        max_num_of_recent_chunks: Optional[int] = None,
    ) -> List[Dict[str, str]]:
        """Mem-alpha memorizing mode: system message only."""
        semantic_enabled = self.is_memory_type_enabled("semantic")
        episodic_enabled = self.is_memory_type_enabled("episodic")

        max_num = max_num_of_recent_chunks if max_num_of_recent_chunks is not None else self.MAX_MEMORY_ITEMS
        if max_num > 0:
            if semantic_enabled:
                semantic_items = (
                    self.semantic if max_num >= len(self.semantic) else self.semantic[-max_num:]
                )
            else:
                semantic_items = []

            if episodic_enabled:
                episodic_items = (
                    self.episodic if max_num >= len(self.episodic) else self.episodic[-max_num:]
                )
            else:
                episodic_items = []
        else:
            episodic_items = []
            semantic_items = []

        core_memory_section = ""
        if self.including_core and self.core is not None:
            core_memory_section = f"<core_memory>\n{self.core}\n</core_memory>"

        memory_type_instructions = []
        if self.is_memory_type_enabled("core"):
            memory_type_instructions.append(
                "* core_memory: Information stored so far (stored as a compact paragraph)"
            )
        if semantic_enabled:
            memory_type_instructions.append(
                "* semantic_memory: General knowledge, factual or conceptual information"
            )
        if episodic_enabled:
            memory_type_instructions.append(
                "* episodic_memory: Specific personal experiences or events with timestamp (mandatory), place, or context"
            )
        if not memory_type_instructions:
            memory_type_instructions.append("* No memory modules are enabled for this run.")

        memory_state_sections = []
        if core_memory_section:
            memory_state_sections.append(core_memory_section)
        if semantic_enabled:
            total_semantic = len(self.semantic)
            visible_semantic = min(len(semantic_items), total_semantic)
            memory_state_sections.append(
                f"<semantic_memory> (Only show the most recent {visible_semantic} out of {total_semantic} memories)\n"
                f"{self._block(lines=semantic_items)}\n"
                f"</semantic_memory>"
            )
        if episodic_enabled:
            total_episodic = len(self.episodic)
            visible_episodic = min(len(episodic_items), total_episodic)
            memory_state_sections.append(
                f"<episodic_memory> (Only show the most recent {visible_episodic} out of {total_episodic} memories)\n"
                f"{self._block(lines=episodic_items)}\n"
                f"</episodic_memory>"
            )
        if not memory_state_sections:
            memory_state_sections.append("No memory modules are enabled for this run.")

        memory_state_text = "\n\n".join(memory_state_sections)
        instructions_text = "\n".join(memory_type_instructions)

        system_prompt = render_prompt(
            "mem_alpha_memorie_system.jinja",
            instructions_text=instructions_text,
            memory_state_text=memory_state_text,
        )

        return [{"role": "system", "content": system_prompt}]

    def new_memory_insert(self, memory_type: str, content: str) -> Optional[Dict[str, str]]:
        if memory_type in ["semantic", "episodic"]:
            self._ensure_memory_type_enabled(memory_type)

        if memory_type == "core" and not self.including_core:
            raise ValueError("Core memory is not available. Set including_core=True to use core memory.")
        if memory_type == "core" and self.core is None:
            raise ValueError("Core memory is not initialized. Set including_core=True to use core memory.")
        if memory_type == "core":
            raise ValueError("Core memory cannot be inserted. Use memory_update to modify core memory content.")

        if self._content_exists(memory_type, content):
            return None

        memory_id = self._generate_memory_id()
        getattr(self, memory_type).append({memory_id: content})

        if memory_type in ["semantic", "episodic"]:
            embedding = self._get_embedding(content)
            self._append_embedding_row(memory_type, memory_id, embedding)

        return {memory_id: content}

    def _append_embedding_row(self, memory_type: str, memory_id: str, embedding: np.ndarray) -> None:
        emb = embedding.reshape(1, -1).astype(np.float32)
        d = emb.shape[1]
        name = f"{memory_type}_embedding_matrix"
        ids_name = f"{memory_type}_embedding_ids"
        matrix: np.ndarray = getattr(self, name)
        if matrix.size == 0:
            new_matrix = emb
        else:
            if matrix.shape[1] != d:
                raise ValueError(
                    f"Embedding dim mismatch: matrix {matrix.shape[1]} vs new {d}"
                )
            new_matrix = np.vstack([matrix, emb])
        setattr(self, name, new_matrix)
        getattr(self, ids_name).append(memory_id)

    def memory_update(self, memory_type: str, new_content: str, memory_id: Optional[str] = None) -> Any:
        if memory_type in ["semantic", "episodic"]:
            self._ensure_memory_type_enabled(memory_type)

        if memory_type == "core" and not self.including_core:
            raise ValueError("Core memory is not available. Set including_core=True to use core memory.")
        if memory_type == "core" and self.core is None:
            raise ValueError("Core memory is not initialized. Set including_core=True to use core memory.")

        if memory_type == "core":
            token_count = count_tokens_memalpha(new_content)
            if token_count > 512:
                truncation_msg = " [content exceeds 512 tokens, truncated]"
                target_tokens = 512 - count_tokens_memalpha(truncation_msg)
                words = new_content.split()
                truncated_content = new_content
                while count_tokens_memalpha(truncated_content) > target_tokens and words:
                    words.pop()
                    truncated_content = " ".join(words)
                self.core = truncated_content + truncation_msg
            else:
                self.core = new_content
            return self.core

        mem_list = getattr(self, memory_type)
        if not memory_id:
            raise ValueError("memory_id is required for semantic/episodic update.")
        for i, mem in enumerate(mem_list):
            if memory_id in mem:
                mem_list[i] = {memory_id: new_content}
                break

        if memory_type in ["semantic", "episodic"]:
            embedding = self._get_embedding(new_content)
            embedding_matrix: np.ndarray = getattr(self, f"{memory_type}_embedding_matrix")
            embedding_ids: List[str] = getattr(self, f"{memory_type}_embedding_ids")
            idx = embedding_ids.index(memory_id)
            embedding_matrix[idx] = embedding.reshape(-1)

        return {memory_id: new_content}

    def memory_delete(self, memory_type: str, memory_id: Optional[str] = None) -> None:
        if memory_type in ["semantic", "episodic"]:
            self._ensure_memory_type_enabled(memory_type)

        if memory_type == "core" and not self.including_core:
            raise ValueError("Core memory is not available. Set including_core=True to use core memory.")
        if memory_type == "core" and self.core is None:
            raise ValueError("Core memory is not initialized. Set including_core=True to use core memory.")

        if memory_type == "core":
            self.core = ""
            return

        mem_list = getattr(self, memory_type)
        for i, mem in enumerate(mem_list):
            if memory_id in mem:
                mem_list.pop(i)
                break

        embedding_matrix: np.ndarray = getattr(self, f"{memory_type}_embedding_matrix")
        embedding_ids: List[str] = getattr(self, f"{memory_type}_embedding_ids")
        try:
            idx = embedding_ids.index(memory_id)
            setattr(
                self,
                f"{memory_type}_embedding_matrix",
                np.delete(embedding_matrix, idx, axis=0),
            )
            embedding_ids.pop(idx)
        except ValueError:
            pass

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def memory_search(
        self,
        memory_type: str,
        query: str,
        top_k: Optional[int] = None,
        min_score: float = 0.0,
        search_method: str = "bm25",
    ) -> List[Tuple[Dict[str, str], float]]:
        if memory_type == "core":
            raise ValueError("Core memory doesn't support searching.")
        if memory_type not in ["semantic", "episodic"]:
            raise ValueError(f"Invalid memory_type: {memory_type}")

        self._ensure_memory_type_enabled(memory_type)

        mem_list = getattr(self, memory_type)
        if not mem_list or not query.strip():
            return []

        if search_method == "bm25":
            return self._search_bm25(memory_type, query, top_k, min_score)
        if search_method in ("text-embedding", "embedding"):
            return self._search_embedding(memory_type, query, top_k, min_score)
        raise ValueError(f"Unknown search method: {search_method}")

    def _search_bm25(
        self,
        memory_type: str,
        query: str,
        top_k: Optional[int],
        min_score: float,
    ) -> List[Tuple[Dict[str, str], float]]:
        mem_list = getattr(self, memory_type)
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        documents: List[Tuple[str, str]] = []
        doc_contents: List[str] = []
        for mem in mem_list:
            for memory_id, content in mem.items():
                documents.append((memory_id, content))
                doc_contents.append(content)

        if not documents:
            return []

        tokenized_corpus = [self._tokenize(c) for c in doc_contents]
        bm25 = BM25Okapi(tokenized_corpus)
        doc_scores = bm25.get_scores(query_tokens)

        results: List[Tuple[Dict[str, str], float]] = []
        for i, (memory_id, content) in enumerate(documents):
            score = float(doc_scores[i])
            if score >= min_score:
                results.append(({memory_id: content}, score))

        results.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            results = results[:top_k]
        return results

    def _search_embedding(
        self,
        memory_type: str,
        query: str,
        top_k: Optional[int],
        min_score: float,
    ) -> List[Tuple[Dict[str, str], float]]:
        mem_list = getattr(self, memory_type)
        embedding_matrix: np.ndarray = getattr(self, f"{memory_type}_embedding_matrix")
        embedding_ids: List[str] = getattr(self, f"{memory_type}_embedding_ids")

        if not mem_list or embedding_matrix.shape[0] == 0:
            return []

        query_embedding = self._get_embedding(query)
        if np.allclose(query_embedding, 0):
            return []

        similarities = _cosine_similarity_query_matrix(query_embedding, embedding_matrix)

        id_to_content: Dict[str, str] = {}
        for mem in mem_list:
            id_to_content.update(mem)

        results: List[Tuple[Dict[str, str], float]] = []
        for memory_id, similarity in zip(embedding_ids, similarities):
            if float(similarity) >= min_score and memory_id in id_to_content:
                results.append(({memory_id: id_to_content[memory_id]}, float(similarity)))

        results.sort(key=lambda x: x[1], reverse=True)
        if top_k is not None:
            results = results[:top_k]
        return results

    def to_serializable(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "including_core": self.including_core,
            "core": self.core,
            "semantic": self.semantic,
            "episodic": self.episodic,
            "semantic_embedding_ids": list(self.semantic_embedding_ids),
            "episodic_embedding_ids": list(self.episodic_embedding_ids),
        }

    @staticmethod
    def from_serializable(
        data: Dict[str, Any],
        embed_one: Optional[Callable[[str], np.ndarray]] = None,
    ) -> "AlphaMemory":
        inc = bool(data.get("including_core", False))
        m = AlphaMemory(including_core=inc, embed_one=embed_one)
        m.core = data.get("core")
        if not inc:
            m.core = None
        m.semantic = [dict(x) for x in data.get("semantic", [])]
        m.episodic = [dict(x) for x in data.get("episodic", [])]
        m.semantic_embedding_ids = list(data.get("semantic_embedding_ids", []))
        m.episodic_embedding_ids = list(data.get("episodic_embedding_ids", []))
        return m

    def rebuild_embeddings_from_content(self) -> None:
        """Recompute embedding matrices from memory texts (after load or dim mismatch)."""
        if self._embed_one is None:
            return
        for mt in ("semantic", "episodic"):
            mem_list = getattr(self, mt)
            ids_attr = f"{mt}_embedding_ids"
            texts: List[str] = []
            ordered_ids: List[str] = []
            for mem in mem_list:
                for mid, content in mem.items():
                    ordered_ids.append(mid)
                    texts.append(content)
            getattr(self, ids_attr).clear()
            getattr(self, ids_attr).extend(ordered_ids)
            if not texts:
                setattr(self, f"{mt}_embedding_matrix", np.empty((0, 0), dtype=np.float32))
                continue
            rows = [self._embed_one(t) for t in texts]
            mat = np.vstack([r.reshape(1, -1).astype(np.float32) for r in rows])
            setattr(self, f"{mt}_embedding_matrix", mat)
