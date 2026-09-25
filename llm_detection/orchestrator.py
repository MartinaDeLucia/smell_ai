from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import pandas as pd

from llm_detection.providers import LLMProvider
from llm_detection.types import (
    DetectionTarget,
    LLMCatalog,
    LLMSmellFinding,
    NormalizationMode,
    PromptMode,
)


@dataclass(frozen=True)
class OrchestratorStats:
    prompts_sent: int = 0
    targets_processed: int = 0
    smells_processed: int = 0


@dataclass(frozen=True)
class GenerationTrace:
    """Raw generation data kept for reproducibility/debugging."""

    filename: str
    smell_id: str
    response: str
    native_reasoning: str = ""
    metadata: dict[str, Any] | None = None


class LLMOrchestrator:
    """Orchestrates LLM detection using smell prompts and a provider.

    This class is additive: it does not alter the existing AST-based pipeline.
    """

    def __init__(self, provider: LLMProvider, catalog: LLMCatalog):
        self.provider = provider
        self.catalog = catalog
        self.last_traces: list[GenerationTrace] = []

    def _generate(self, prompt: str, filename: str, smell_id: str) -> str:
        generation = self.provider.generate_with_metadata(prompt)
        self.last_traces.append(
            GenerationTrace(
                filename=filename,
                smell_id=smell_id,
                response=generation.response,
                native_reasoning=generation.native_reasoning,
                metadata=dict(generation.metadata),
            )
        )
        return generation.response

    @staticmethod
    def _code_with_line_numbers(code: str) -> str:
        lines = (code or "").splitlines()
        # 1-based line numbering to match typical editors
        return "\n".join(f"{i}: {line}" for i, line in enumerate(lines, start=1))

    def build_prompt(
        self,
        smell_id: str,
        target: DetectionTarget,
        prompt_mode: PromptMode,
    ) -> str:
        smell = self.catalog.get_smell(smell_id)
        smell_prompt = smell.get_prompt(prompt_mode)

        numbered_code = self._code_with_line_numbers(target.code)

        # NOTE: keep ONE single output contract here (not duplicated in smell prompts).
        # The smell prompt should focus on definition + rules; this block enforces schema.
        return (
            f"{smell_prompt}\n\n"
            "You are a code smell detector.\n"
            "OUTPUT FORMAT (STRICT):\n"
            "Return ONLY valid JSON.\n"
            "Do NOT include explanations, markdown, or extra text.\n\n"
            "The JSON schema MUST be exactly:\n"
            "{\n"
            '  "findings": [\n'
            "    {\n"
            '      "function_name": "<name of the function or method where the smell occurs, or null if global>",\n'
            '      "line": <line number where the smell starts>,\n'
            '      "description": "<short summary of the detected smell>",\n'
            '      "reasoning": "<concise evidence-based rationale: violated rule + concrete code evidence>",\n'
            '      "additional_info": "<optional refactoring hint or summary>"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "IMPORTANT:\n"
            "- The top-level JSON object MUST contain ONLY the key 'findings'.\n"
            "- Do NOT wrap the JSON in ``` fences.\n\n"
            "GUIDELINES:\n"
            "- If no smell is detected, return: { \"findings\": [] }\n"
            "- If multiple occurrences exist, return one item per occurrence in 'findings' (do not group them).\n"
            "- Never return per-function keys (e.g., {\"func\": {...}}). Always return a 'findings' array.\n"
            "- Keep the response concise. If you are unsure, return fewer findings but keep valid JSON.\n"
            "- Use precise line numbers.\n"
            "- The code is provided with 1-based line numbers as a prefix like '12: ...'.\n"
            "  When you report 'line', use that exact prefix number.\n"
            "- For every finding, 'reasoning' is REQUIRED and must be concise (1-3 sentences).\n"
            "- In 'reasoning', explain which smell rule is matched and point to the concrete code evidence.\n"
            "- Do not expose a hidden chain-of-thought; provide only the short, user-facing justification needed to verify the finding.\n"
            "- Be conservative: avoid false positives.\n\n"
            f"FILENAME: {target.filename}\n"
            "CODE (numbered):\n"
            f"{numbered_code}\n"
        )

    def detect(
        self,
        targets: Sequence[DetectionTarget],
        smell_ids: Sequence[str],
        prompt_mode: PromptMode = PromptMode.DRAFT_IF_AVAILABLE,
        *,
        normalize_mode: NormalizationMode = NormalizationMode.STRICT,
    ) -> tuple[list[LLMSmellFinding], OrchestratorStats]:
        findings: list[LLMSmellFinding] = []
        prompts_sent = 0
        self.last_traces = []

        for target in targets:
            for smell_id in smell_ids:
                smell = self.catalog.get_smell(smell_id)
                if not smell.is_ready_for_detection():
                    continue

                prompt = self.build_prompt(smell_id, target, prompt_mode)
                raw = self._generate(prompt, target.filename, smell_id)
                prompts_sent += 1

                findings.extend(
                    self._normalize_response(
                        raw,
                        target.filename,
                        smell_id,
                        normalize_mode=normalize_mode,
                    )
                )

        stats = OrchestratorStats(
            prompts_sent=prompts_sent,
            targets_processed=len(targets),
            smells_processed=len(smell_ids),
        )
        return findings, stats

    def detect_for_prompt_engineering(
        self,
        targets: Sequence[DetectionTarget],
        smell_id: str,
        prompt_mode: PromptMode,
        *,
        normalize_mode: NormalizationMode = NormalizationMode.SALVAGE,
    ) -> tuple[list[LLMSmellFinding], OrchestratorStats]:
        """UC02 helper: allows testing draft prompt before saving as default."""
        findings: list[LLMSmellFinding] = []
        prompts_sent = 0
        self.last_traces = []

        for target in targets:
            prompt = self.build_prompt(smell_id, target, prompt_mode)
            raw = self._generate(prompt, target.filename, smell_id)
            prompts_sent += 1
            findings.extend(
                self._normalize_response(
                    raw,
                    target.filename,
                    smell_id,
                    normalize_mode=normalize_mode,
                )
            )

        stats = OrchestratorStats(
            prompts_sent=prompts_sent,
            targets_processed=len(targets),
            smells_processed=1,
        )
        return findings, stats

    def detect_for_prompt_engineering_with_raw(
        self,
        targets: Sequence[DetectionTarget],
        smell_id: str,
        prompt_mode: PromptMode,
        *,
        normalize_mode: NormalizationMode = NormalizationMode.SALVAGE,
    ) -> tuple[list[LLMSmellFinding], OrchestratorStats, dict[str, str]]:
        """UC02 helper: like detect_for_prompt_engineering but returns raw responses per file."""
        findings: list[LLMSmellFinding] = []
        raw_by_filename: dict[str, str] = {}
        prompts_sent = 0
        self.last_traces = []

        for target in targets:
            prompt = self.build_prompt(smell_id, target, prompt_mode)
            raw = self._generate(prompt, target.filename, smell_id)
            raw_by_filename[target.filename] = raw
            prompts_sent += 1
            findings.extend(
                self._normalize_response(
                    raw,
                    target.filename,
                    smell_id,
                    normalize_mode=normalize_mode,
                )
            )

        stats = OrchestratorStats(
            prompts_sent=prompts_sent,
            targets_processed=len(targets),
            smells_processed=1,
        )
        return findings, stats, raw_by_filename

    @staticmethod
    def _try_parse_json_payload(raw: str) -> Any | None:
        """Best-effort JSON parsing: strips common fences and extracts the first JSON object/array."""
        if raw is None:
            return None

        text = str(raw).strip()
        if not text:
            return None

        # Fast path
        try:
            return json.loads(text)
        except Exception:
            pass

        # Strip markdown fences (best-effort)
        if "```" in text:
            lines = text.splitlines()
            # drop leading fence
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            # drop trailing fence
            if lines and lines[-1].rstrip().endswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Extract first balanced JSON object/array
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            if start == -1:
                continue

            depth = 0
            in_string = False
            escape = False
            for i in range(start, len(text)):
                ch = text[i]
                if escape:
                    escape = False
                    continue
                if ch == "\\":
                    escape = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue

                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : i + 1].strip()
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break

        return None

    def _normalize_response(
            self,
            raw: str,
            filename: str,
            smell_id: str,
            *,
            normalize_mode: NormalizationMode,
    ) -> list[LLMSmellFinding]:
        smell = self.catalog.get_smell(smell_id)

        def _safe_str(value: Any) -> str:
            return "" if value is None else str(value)

        def _safe_list(value: Any) -> list[dict[str, Any]] | None:
            if not isinstance(value, list):
                return None
            if not all(isinstance(x, dict) for x in value):
                return None
            return value

        def _safe_single_finding(value: Any) -> dict[str, Any] | None:
            if not isinstance(value, dict):
                return None
            if "line" in value or "line_number" in value:
                return value
            return None

        payload = self._try_parse_json_payload(raw)
        if payload is None:
            if normalize_mode == NormalizationMode.STRICT:
                return []
            # SALVAGE: keep a trace as a non-finding (line=-1) for debugging
            return [
                LLMSmellFinding(
                    filename=filename,
                    function_name="",
                    smell_name=smell.display_name,
                    line=-1,
                    description=smell.description or smell.display_name,
                    additional_info="Unparseable LLM response; see raw_response",
                    smell_id=smell_id,
                    raw_response=raw,
                )
            ]

        # STRICT: accept ONLY {"findings": [ ... ]} with dict items
        if normalize_mode == NormalizationMode.STRICT:
            if not isinstance(payload, dict):
                return []
            strict_findings = payload.get("findings")
            if not isinstance(strict_findings, list):
                return []
            if not all(isinstance(x, dict) for x in strict_findings):
                return []

            out: list[LLMSmellFinding] = []
            for item in strict_findings:
                line = item.get("line", -1)
                if line == -1 and "line_number" in item:
                    line = item.get("line_number", -1)

                try:
                    line_int = int(line)
                except Exception:
                    continue
                if line_int <= 0:
                    continue

                confidence = item.get("confidence")
                confidence_f = None
                if confidence is not None:
                    try:
                        confidence_f = float(confidence)
                    except Exception:
                        confidence_f = None

                out.append(
                    LLMSmellFinding(
                        filename=filename,
                        function_name=_safe_str(item.get("function_name", "")),
                        smell_name=smell.display_name,
                        line=line_int,
                        description=_safe_str(
                            item.get("description", smell.description or smell.display_name)
                        ),
                        reasoning=_safe_str(
                            item.get(
                                "reasoning",
                                item.get("rationale", item.get("explanation", "")),
                            )
                        ),
                        additional_info=_safe_str(item.get("additional_info", "")),
                        smell_id=smell_id,
                        confidence=confidence_f,
                        raw_response=raw,
                    )
                )
            return out

        # ---------------- SALVAGE ----------------

        # Start from "findings"; if missing, try fallback keys.
        findings_payload: Any = payload.get("findings") if isinstance(payload, dict) else None
        source_key: str | None = None
        schema_invalid = False  # JSON valid but doesn't match expected schema

        if isinstance(payload, dict) and findings_payload is None:
            for k, v in payload.items():
                candidate = _safe_list(v)
                if candidate is None:
                    single = _safe_single_finding(v)
                    if single is None:
                        continue
                    findings_payload = [single]
                    source_key = str(k)
                    break

                if any(("line" in it or "line_number" in it) for it in candidate):
                    findings_payload = candidate
                    source_key = str(k)
                    break

            if findings_payload is None:
                # JSON is valid, but no "findings" and no recognizable fallback.
                schema_invalid = True
                findings_payload = []

        if not isinstance(findings_payload, list):
            findings_payload = []

        # If schema is invalid, return a single diagnostic row (line=-1) so the user
        # immediately understands this is not "no smells", but "bad response schema".
        if schema_invalid:
            return [
                LLMSmellFinding(
                    filename=filename,
                    function_name="",
                    smell_name=smell.display_name,
                    line=-1,
                    description="Invalid LLM response schema (missing 'findings')",
                    additional_info="Expected: {'findings': [...]} — see raw_response",
                    smell_id=smell_id,
                    raw_response=raw,
                )
            ]

        out: list[LLMSmellFinding] = []
        for item in findings_payload:
            if not isinstance(item, dict):
                continue

            line = item.get("line", -1)
            if line == -1 and "line_number" in item:
                line = item.get("line_number", -1)
            try:
                line_int = int(line)
            except Exception:
                continue
            if line_int <= 0:
                continue

            confidence = item.get("confidence")
            confidence_f = None
            if confidence is not None:
                try:
                    confidence_f = float(confidence)
                except Exception:
                    confidence_f = None

            desc = item.get("description")
            if not desc:
                desc = (
                    f"Recovered from non-standard JSON schema key '{source_key}'"
                    if source_key
                    else (smell.description or smell.display_name)
                )

            out.append(
                LLMSmellFinding(
                    filename=filename,
                    function_name=_safe_str(item.get("function_name", "")),
                    smell_name=smell.display_name,
                    line=line_int,
                    description=_safe_str(desc),
                    reasoning=_safe_str(
                        item.get(
                            "reasoning",
                            item.get("rationale", item.get("explanation", "")),
                        )
                    ),
                    additional_info=_safe_str(
                        item.get(
                            "additional_info",
                            item.get("code_snippet", item.get("code", "")),
                        )
                    ),
                    smell_id=smell_id,
                    confidence=confidence_f,
                    raw_response=raw,
                )
            )

        return out

    @staticmethod
    def findings_to_dataframe(findings: Iterable[LLMSmellFinding]) -> pd.DataFrame:
        rows = [f.to_overview_row() for f in findings]
        columns = [
            "filename",
            "function_name",
            "smell_name",
            "line",
            "description",
            "reasoning",
            "additional_info",
        ]
        return pd.DataFrame(rows, columns=columns)
