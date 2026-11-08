"""Domain solver for `search_XXXX` tasks using local BM25 + entity graph traversal over `workspace/kb/*.md`."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from harness.jiuwen_agent import JiuwenAgentGateway, extract_code_or_text
from harness.model_router import ModelRouter
from harness.task_scanner import TaskFeatures, is_ignored_fs_entry

_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")
_ENTITY_ID_RE = re.compile(r"\b(?:person_\d+|company_\d+|project_\d+|city_\d+)\b", re.IGNORECASE)


class SearchDomainSolver:
    """Builds a local zero-token BM25 + 2-hop entity index over `workspace/kb/*.md`
    and sends only the top-ranked snippets (500-800 tokens) to the fast `-noreason` model.
    """

    def __init__(
        self,
        gateway: JiuwenAgentGateway,
        router: ModelRouter,
        system_prompt: str,
    ) -> None:
        self.gateway = gateway
        self.router = router
        self.system_prompt = system_prompt

    async def solve(self, tf: TaskFeatures, qualifying_mode: bool = False) -> str | None:
        kb_dir = tf.task_dir / "workspace" / "kb"
        if not kb_dir.exists():
            kb_dir = tf.task_dir / "workspace"
        if not kb_dir.exists():
            return None

        md_files = [p for p in sorted(kb_dir.glob("*.md")) if not is_ignored_fs_entry(p.name)]
        if not md_files:
            return None

        snippets = self._retrieve_top_snippets(tf.prompt, md_files, top_k=5)
        if not snippets:
            return None

        model = self.router.select_model(tf, attempt=0, qualifying_mode=qualifying_mode)
        is_reason = not model.endswith("-noreason")
        max_toks = self.router.max_tokens_for_domain("search", is_reasoning_model=is_reason)

        context_block = "\n\n".join(f"[{name}]\n{body}" for name, body in snippets)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Question ({tf.task_id}): {tf.prompt}\n\n"
                    f"Retrieved Knowledge Base Snippets:\n{context_block}\n\n"
                    "Provide ONLY the exact final answer on a single line."
                ),
            },
        ]

        raw_resp = await self.gateway.chat_completion(
            task_id=tf.task_id,
            model=model,
            messages=messages,
            max_tokens=max_toks,
            temperature=0.0,
        )
        cleaned = extract_code_or_text(raw_resp).strip()
        if not cleaned or "UNSOLVABLE" in cleaned.upper():
            return None
        return cleaned.splitlines()[-1].strip()

    @staticmethod
    def _retrieve_top_snippets(
        query: str,
        md_files: list[Path],
        top_k: int = 5,
    ) -> list[tuple[str, str]]:
        docs: list[tuple[str, str, list[str], set[str]]] = []
        df: Counter[str] = Counter()
        total_len = 0

        for path in md_files:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            tokens = [t.lower() for t in _WORD_RE.findall(text)]
            ents = {e.lower() for e in _ENTITY_ID_RE.findall(text)}
            ents.add(path.stem.lower())
            docs.append((path.name, text, tokens, ents))
            total_len += len(tokens)
            for term in set(tokens):
                df[term] += 1

        if not docs:
            return []

        n_docs = len(docs)
        avgdl = max(1.0, total_len / n_docs)
        q_tokens = [t.lower() for t in _WORD_RE.findall(query)]
        seed_entities = {e.lower() for e in _ENTITY_ID_RE.findall(query)}

        # 1-hop and 2-hop entity expansion across the KB graph
        hop1_entities = set(seed_entities)
        for _, _, _, ents in docs:
            if ents & seed_entities:
                hop1_entities.update(ents)

        k1 = 1.5
        b = 0.75
        scored: list[tuple[float, str, str]] = []

        for name, text, tokens, ents in docs:
            tf_counter = Counter(tokens)
            doc_len = len(tokens)
            score = 0.0
            for qt in q_tokens:
                if qt not in tf_counter:
                    continue
                n_qt = df[qt]
                idf = math.log(1.0 + (n_docs - n_qt + 0.5) / (n_qt + 0.5))
                f = tf_counter[qt]
                score += idf * (f * (k1 + 1.0)) / (f + k1 * (1.0 - b + b * (doc_len / avgdl)))

            if ents & seed_entities:
                score += 15.0
            elif ents & hop1_entities:
                score += 6.0

            scored.append((score, name, text[:450]))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [(name, body) for _, name, body in scored[:top_k]]
