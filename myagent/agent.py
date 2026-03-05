from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.parse import urlparse
from urllib.request import Request
from urllib.request import urlopen

from google import genai
from google.adk.agents.llm_agent import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.tools import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from pydantic import BaseModel
from pydantic import Field


def _load_local_env_file() -> None:
    """Load myagent/.env into process env when runtime doesn't auto-load it."""
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if not os.getenv(key):
                os.environ[key] = value
    except OSError as err:
        print(f"[env] could not load {env_path.name}: {type(err).__name__}")


def _get_configured_model() -> str:
    return (
        os.getenv("GEMINI_MODEL")
        or os.getenv("GOOGLE_GENAI_MODEL")
        or "gemini-3.1-flash-lite-preview"
    ).strip() or "gemini-3.1-flash-lite-preview"


def _get_fallback_model() -> str:
    fallback = (os.getenv("GEMINI_FALLBACK_MODEL") or "").strip()
    if fallback:
        return fallback
    if "preview" in _get_configured_model().lower():
        return "gemini-2.5-flash"
    return ""


def _get_google_api_key() -> str:
    return (
        os.getenv("GOOGLE_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or ""
    ).strip()


def _build_runtime_config_error() -> str:
    if _get_google_api_key():
        return ""
    return (
        "à¸¢à¸±à¸‡à¹„à¸¡à¹ˆà¸žà¸š GOOGLE_API_KEY à¸ªà¸³à¸«à¸£à¸±à¸šà¹€à¸£à¸µà¸¢à¸à¹‚à¸¡à¹€à¸”à¸¥\n"
        "à¹ƒà¸«à¹‰à¸•à¸±à¹‰à¸‡à¸„à¹ˆà¸²à¹ƒà¸™ myagent/.env à¹à¸¥à¹‰à¸§à¸£à¸µà¸ªà¸•à¸²à¸£à¹Œà¸• adk web"
    )


_load_local_env_file()

_MAX_ARTICLE_CHARS = int(os.getenv("MAX_ARTICLE_CHARS", "14000") or "14000")
_MAX_TTS_CHARS = 1800
_DEFAULT_MODEL = _get_configured_model()
_FALLBACK_MODEL = _get_fallback_model()
_AUTO_FALLBACK_ON_503 = (
    os.getenv("AUTO_FALLBACK_ON_503", "1").strip().lower()
    in {"1", "true", "yes", "on"}
)
_CURRENT_RUNTIME_MODEL = _DEFAULT_MODEL
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_NOTION_PAGE_URL_RE = re.compile(
    r"https?://(?:www\.)?notion\.so/\S+",
    re.IGNORECASE,
)
_NOTION_HINT_RE = re.compile(r"(?:\bnotion\b|à¹‚à¸™à¸Šà¸±à¸™|à¹‚à¸™à¸Šà¸±à¹ˆà¸™)", re.IGNORECASE)
_NOTION_MCP_INIT_ERROR = ""
_RUNTIME_CONFIG_ERROR = _build_runtime_config_error()
_NOTION_PAGE_ID_RE = re.compile(r"([0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})")


class ResearchInput(BaseModel):
    url: str = Field(..., description="News URL from user.")


class ResearchOutput(BaseModel):
    status: Literal["ok", "error"]
    original_url: str = ""
    source_url: str = ""
    title: str = ""
    published_time: str = ""
    method: str = ""
    content: str = ""
    error: str = ""


class AnalysisInput(BaseModel):
    research: ResearchOutput


class AnalysisOutput(BaseModel):
    status: Literal["ok", "error"]
    source_url: str = ""
    title_th: str = ""
    summary_points: list[str] = Field(default_factory=list)
    translated_body_th: str = ""
    key_entities: list[str] = Field(default_factory=list)
    key_numbers: list[str] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    error: str = ""


class WriterInput(BaseModel):
    analysis: AnalysisOutput
    research: ResearchOutput


class WriterOutput(BaseModel):
    status: Literal["ok", "error"]
    source_url: str = ""
    final_markdown_th: str = ""
    narration_text_th: str = ""
    error: str = ""


class NarratorInput(BaseModel):
    writer: WriterOutput


class NarratorOutput(BaseModel):
    status: Literal["ok", "error"]
    source_url: str = ""
    final_markdown_th: str = ""
    audio_filename: str = ""
    audio_version: str = ""
    audio_truncated: bool = False
    error: str = ""


def _normalize_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return url


def _extract_latest_user_text(llm_request: LlmRequest) -> str:
    for content in reversed(llm_request.contents or []):
        if getattr(content, "role", "") != "user":
            continue
        texts = [
            part.text.strip()
            for part in (content.parts or [])
            if getattr(part, "text", None)
        ]
        if texts:
            return " ".join(texts)
    return ""


def _download_text(url: str, timeout: int = 25) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/133.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=timeout) as response:
        data = response.read()
    return data.decode("utf-8", errors="ignore")


def _extract_jina_payload(markdown: str) -> dict[str, str]:
    title_match = re.search(r"^Title:\s*(.+)$", markdown, flags=re.MULTILINE)
    source_match = re.search(
        r"^URL Source:\s*(.+)$", markdown, flags=re.MULTILINE
    )
    published_match = re.search(
        r"^Published Time:\s*(.+)$", markdown, flags=re.MULTILINE
    )

    body = markdown
    split_token = "Markdown Content:"
    if split_token in markdown:
        body = markdown.split(split_token, 1)[1].strip()

    return {
        "title": title_match.group(1).strip() if title_match else "",
        "source_url": source_match.group(1).strip() if source_match else "",
        "published_time": (
            published_match.group(1).strip() if published_match else ""
        ),
        "content": _prepare_article_content(body)[:_MAX_ARTICLE_CHARS],
    }


def _prepare_article_content(text: str) -> str:
    content = (text or "").strip()
    if not content:
        return ""
    # Remove repeated whitespace and obvious noisy separators from reader dumps.
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"[ \t]{2,}", " ", content)
    content = re.sub(r"(?im)^related article.*$", "", content)
    content = re.sub(r"(?im)^advertisement.*$", "", content)
    return content.strip()


def fetch_news_content(url: str) -> dict[str, str]:
    """Fetch readable news content from URL with fallback strategies."""
    normalized = _normalize_url(url)
    if not normalized:
        return {
            "ok": "false",
            "method": "none",
            "original_url": "",
            "source_url": "",
            "title": "",
            "published_time": "",
            "content": "",
            "error": "Empty URL",
        }

    parsed = urlparse(normalized)
    if not parsed.netloc:
        return {
            "ok": "false",
            "method": "none",
            "original_url": normalized,
            "source_url": "",
            "title": "",
            "published_time": "",
            "content": "",
            "error": "Invalid URL",
        }

    stripped = normalized.replace("https://", "").replace("http://", "")
    jina_url = f"https://r.jina.ai/http://{stripped}"

    try:
        jina_text = _download_text(jina_url)
        payload = _extract_jina_payload(jina_text)
        if payload["content"]:
            return {
                "ok": "true",
                "method": "jina_reader",
                "original_url": normalized,
                "source_url": payload["source_url"] or normalized,
                "title": payload["title"],
                "published_time": payload["published_time"],
                "content": payload["content"],
                "error": "",
            }
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as err:
        jina_error = f"{type(err).__name__}: {err}"
    else:
        jina_error = "Reader returned empty content"

    try:
        raw_html = _download_text(normalized)
        if raw_html:
            plain = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw_html)
            plain = re.sub(r"(?is)<style.*?>.*?</style>", " ", plain)
            plain = re.sub(r"(?s)<[^>]+>", " ", plain)
            plain = re.sub(r"\s+", " ", plain).strip()
            if plain:
                return {
                    "ok": "true",
                    "method": "direct_html_fallback",
                    "original_url": normalized,
                    "source_url": normalized,
                    "title": "",
                    "published_time": "",
                    "content": _prepare_article_content(plain)[:_MAX_ARTICLE_CHARS],
                    "error": "",
                }
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as err:
        fallback_error = f"{type(err).__name__}: {err}"
    else:
        fallback_error = "Direct fetch returned empty content"

    return {
        "ok": "false",
        "method": "none",
        "original_url": normalized,
        "source_url": "",
        "title": "",
        "published_time": "",
        "content": "",
        "error": f"jina={jina_error}; fallback={fallback_error}",
    }


def update_workflow_stage(
    stage: str, note: str = "", tool_context: ToolContext | None = None
) -> dict[str, str]:
    """Update workflow stage in session state."""
    if tool_context is None:
        return {"ok": "false", "stage": stage, "timestamp_utc": "", "error": "Missing tool_context"}
    now = datetime.now(timezone.utc).isoformat()
    history = tool_context.state.get("workflow_history", [])
    if not isinstance(history, list):
        history = []

    entry = {"stage": stage, "note": note, "timestamp_utc": now}
    history.append(entry)
    history = history[-25:]

    if stage == "Draft":
        # New workflow run starts here; reset one-audio-per-run guard.
        tool_context.state["workflow_run_id"] = now
        tool_context.state["audio_generated_run_id"] = ""
        tool_context.state["last_tts_cache"] = {}

    tool_context.state["workflow_stage"] = stage
    tool_context.state["workflow_history"] = history
    return {"ok": "true", "stage": stage, "timestamp_utc": now}


def _clean_tts_text(text: str, max_chars: int = _MAX_TTS_CHARS) -> tuple[str, bool]:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return "", False
    if len(cleaned) <= max_chars:
        return cleaned, False
    clipped = cleaned[:max_chars]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return f"{clipped} ...", True


def _chunk_tts_text(text: str, chunk_size: int = 180) -> list[str]:
    words = text.split(" ")
    if not words:
        return []
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(word) > chunk_size:
            for i in range(0, len(word), chunk_size):
                chunks.append(word[i : i + chunk_size])
            current = ""
        else:
            current = word
    if current:
        chunks.append(current)
    return chunks


def _download_tts_chunk(text: str, lang: str = "th") -> bytes:
    query = urlencode(
        {"ie": "UTF-8", "q": text, "tl": lang, "client": "tw-ob"}
    )
    url = f"https://translate.google.com/translate_tts?{query}"
    request = Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/133.0.0.0 Safari/537.36"
            )
        },
    )
    with urlopen(request, timeout=25) as response:
        return response.read()


def _synthesize_thai_speech_mp3(text: str) -> bytes:
    chunks = _chunk_tts_text(text, chunk_size=180)
    if not chunks:
        return b""
    parts = [_download_tts_chunk(chunk, lang="th") for chunk in chunks]
    return b"".join(parts)


async def create_thai_speech(
    text: str, tool_context: ToolContext
) -> dict[str, str]:
    """Create Thai MP3 narration and attach it as an ADK artifact."""
    prepared, truncated = _clean_tts_text(text)
    if not prepared:
        return {
            "ok": "false",
            "error": "Empty narration text",
            "filename": "",
            "version": "",
            "chars_used": "0",
            "truncated": "false",
        }

    run_id = str(tool_context.state.get("workflow_run_id", "")).strip()
    audio_run_id = str(tool_context.state.get("audio_generated_run_id", "")).strip()
    if run_id and audio_run_id == run_id:
        # Hard guard: one workflow run can emit only one audio artifact.
        return {
            "ok": "true",
            "error": "duplicate_audio_suppressed",
            "filename": "",
            "version": "",
            "chars_used": str(len(prepared)),
            "truncated": "true" if truncated else "false",
        }

    # Deduplicate repeated TTS calls for the same narration text.
    text_fingerprint = hashlib.sha256(prepared.encode("utf-8")).hexdigest()
    cached = tool_context.state.get("last_tts_cache", {})
    if isinstance(cached, dict):
        if (
            cached.get("fingerprint") == text_fingerprint
            and cached.get("filename")
            and cached.get("version")
        ):
            # Same text in the same session: do not emit another audio response.
            return {
                "ok": "true",
                "error": "duplicate_audio_suppressed",
                "filename": "",
                "version": "",
                "chars_used": str(cached.get("chars_used", len(prepared))),
                "truncated": str(cached.get("truncated", "false")),
            }

    try:
        audio_bytes = _synthesize_thai_speech_mp3(prepared)
        if not audio_bytes:
            return {
                "ok": "false",
                "error": "TTS service returned empty audio",
                "filename": "",
                "version": "",
                "chars_used": str(len(prepared)),
                "truncated": "true" if truncated else "false",
            }
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as err:
        return {
            "ok": "false",
            "error": f"{type(err).__name__}: {err}",
            "filename": "",
            "version": "",
            "chars_used": str(len(prepared)),
            "truncated": "true" if truncated else "false",
        }

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"thai_news_tts_{ts}.mp3"
    artifact = types.Part(
        inline_data=types.Blob(mime_type="audio/mpeg", data=audio_bytes)
    )
    version = await tool_context.save_artifact(
        filename=filename,
        artifact=artifact,
        custom_metadata={"kind": "tts", "language": "th"},
    )
    try:
        # Reduce extra UI summarization events after this tool finishes.
        tool_context.actions.skip_summarization = True
    except Exception:
        pass
    tool_context.state["last_tts_cache"] = {
        "fingerprint": text_fingerprint,
        "filename": filename,
        "version": str(version),
        "chars_used": str(len(prepared)),
        "truncated": "true" if truncated else "false",
    }
    tool_context.state["audio_generated_run_id"] = run_id or "__legacy__"
    return {
        "ok": "true",
        "error": "",
        "filename": filename,
        "version": str(version),
        "chars_used": str(len(prepared)),
        "truncated": "true" if truncated else "false",
    }


_RESEARCH_INSTRUCTION = """
You are Research Agent.
Task:
1) Call update_workflow_stage with stage='Draft'.
2) Call fetch_news_content with the input URL.
3) Return JSON matching ResearchOutput exactly.

Output requirements:
- status='ok' when content is available, otherwise status='error'.
- Preserve source_url, title, published_time, method.
- content must be the fetched readable article text.
- Do not output markdown or extra text, output JSON only.
""".strip()

_ANALYZER_INSTRUCTION = """
You are Analyzer Agent.
Input is JSON matching AnalysisInput.
Task:
1) Call update_workflow_stage with stage='Reviewed'.
2) If research.status='error', return status='error' and pass error forward.
3) If research.status='ok', analyze only research.content and produce:
   - title_th
   - summary_points (exactly 5 items, Thai)
   - translated_body_th (Thai detailed translation that keeps all key paragraphs and facts; if source is very long, prioritize core sections and keep output around 3,000-6,000 Thai chars)
   - key_entities, key_numbers, risk_notes
4) Return JSON matching AnalysisOutput exactly.

Rules:
- Keep names, dates, numbers faithful to source.
- No hallucination.
- JSON only.
""".strip()

_WRITER_INSTRUCTION = """
You are Writer Agent.
Input is JSON matching WriterInput.
Task:
1) Call update_workflow_stage with stage='Approved'.
2) If analysis.status='error' or research.status='error', return status='error'.
3) Build final_markdown_th using this order:
   - headline in Thai
   - 5 bullet summary in Thai
   - full Thai translation from analysis.translated_body_th
   - source URL
4) Build narration_text_th for TTS from the same content, concise and <= 1200 chars.
5) Return JSON matching WriterOutput exactly.

Rules:
- Thai language only for narrative fields.
- JSON only.
""".strip()

_NARRATOR_INSTRUCTION = """
You are Narrator Agent.
Input is JSON matching NarratorInput.
Task:
1) If writer.status='error', return status='error'.
2) Call create_thai_speech with writer.narration_text_th.
3) Call update_workflow_stage with stage='Audio Ready'.
4) Return JSON matching NarratorOutput exactly.

Rules:
- Keep final_markdown_th unchanged from writer input.
- Keep audio_filename and audio_version as empty strings to avoid duplicate
  inline audio player rendering in ADK Web.
- If TTS fails, return status='error' with error detail.
- JSON only.
""".strip()

_ROOT_INSTRUCTION = """
You are an orchestrator for a 3-agent news pipeline:
Research -> Analyzer -> Writer.

Behavior:
1) If user asks Notion operation (keyword "Notion" or notion.so URL):
   - if user asks to save latest summary, call save_latest_summary_to_notion with the page URL
   - if tool call succeeds, confirm in Thai what was saved and where
   - if tool call fails, explain the exact failure briefly in Thai and ask for a valid target URL
2) If user provides a non-Notion news URL:
   - call research_agent with {"url": "..."}
   - if ok, call analyzer_agent with {"research": research_result}
   - if ok, call writer_agent with {"analysis": analysis_result, "research": research_result}
   - then call update_workflow_stage with stage='Published'
   - respond in Thai with writer_result.final_markdown_th
   - do not call create_thai_speech directly (audio is generated automatically by system callback).
3) If any stage fails:
   - explain failure in Thai briefly
   - ask user for another URL
4) If no URL is provided:
   - ask user to provide a URL in Thai.

Rules:
- Always answer user in Thai.
- Do not fabricate facts.
- Never call create_thai_speech directly from model instructions.
- Notion intent has higher priority than generic URL handling.
- For Notion requests, do not run news pipeline unless user explicitly asks to summarize new URL.
- Never end your turn with tool calls only.
- Always produce a complete Thai final message for the user.
""".strip()


def _root_on_tool_error_callback(
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    error: Exception,
) -> dict | None:
    """Convert tool crashes to structured error payload for model recovery."""
    _ = args
    _ = tool_context
    return {
        "status": "error",
        "error": f"tool_error:{tool.name}:{type(error).__name__}",
    }


def _build_text_response(message: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model", parts=[types.Part.from_text(text=message)]
        ),
        turn_complete=True,
    )


def _coerce_mapping(value: object) -> dict:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            return {}
    if hasattr(value, "dict"):
        try:
            dumped = value.dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            return {}
    if isinstance(value, str):
        try:
            dumped = json.loads(value)
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            return {}
    return {}


def _iter_state_items(callback_context: CallbackContext):
    try:
        keys = list(callback_context.state.keys())
    except Exception:
        return
    for key in keys:
        try:
            value = callback_context.state.get(key)
        except Exception:
            continue
        yield str(key), value


def _find_article_mapping(value: object, depth: int = 0) -> dict:
    if depth > 5:
        return {}
    data = _coerce_mapping(value)
    if data:
        content = (data.get("content") or "").strip()
        status = str(data.get("status") or "").strip().lower()
        error = (data.get("error") or "").strip()
        if content or status == "error" or error:
            return data
        for nested in data.values():
            found = _find_article_mapping(nested, depth + 1)
            if found:
                return found
        return {}
    if isinstance(value, list):
        for item in value:
            found = _find_article_mapping(item, depth + 1)
            if found:
                return found
    return {}


def _find_analysis_mapping(value: object, depth: int = 0) -> dict:
    if depth > 5:
        return {}
    data = _coerce_mapping(value)
    if data:
        body = (data.get("translated_body_th") or "").strip()
        status = str(data.get("status") or "").strip().lower()
        error = (data.get("error") or "").strip()
        if body or status == "error" or error:
            return data
        for nested in data.values():
            found = _find_analysis_mapping(nested, depth + 1)
            if found:
                return found
        return {}
    if isinstance(value, list):
        for item in value:
            found = _find_analysis_mapping(item, depth + 1)
            if found:
                return found
    return {}


def _read_research_output(callback_context: CallbackContext) -> dict:
    preferred_keys = [
        "research_output",
        "research",
        "research_result",
        "research_agent_output",
        "research_agent.research_output",
    ]
    for key in preferred_keys:
        try:
            raw = callback_context.state.get(key, {})
        except Exception:
            raw = {}
        found = _find_article_mapping(raw)
        if found:
            return found

    for _, raw in _iter_state_items(callback_context):
        found = _find_article_mapping(raw)
        if found:
            return found
    return {}


def _read_analysis_output(callback_context: CallbackContext) -> dict:
    preferred_keys = [
        "analysis_output",
        "analysis",
        "analysis_result",
        "analyzer_agent_output",
        "analyzer_agent.analysis_output",
    ]
    for key in preferred_keys:
        try:
            raw = callback_context.state.get(key, {})
        except Exception:
            raw = {}
        found = _find_analysis_mapping(raw)
        if found:
            return found

    for _, raw in _iter_state_items(callback_context):
        found = _find_analysis_mapping(raw)
        if found:
            return found
    return {}


def _find_writer_mapping(value: object, depth: int = 0) -> dict:
    if depth > 5:
        return {}
    data = _coerce_mapping(value)
    if data:
        final_markdown = (data.get("final_markdown_th") or "").strip()
        narration = (data.get("narration_text_th") or "").strip()
        status = str(data.get("status") or "").strip().lower()
        error = (data.get("error") or "").strip()
        if final_markdown or narration or status == "error" or error:
            return data
        for nested in data.values():
            found = _find_writer_mapping(nested, depth + 1)
            if found:
                return found
        return {}
    if isinstance(value, list):
        for item in value:
            found = _find_writer_mapping(item, depth + 1)
            if found:
                return found
    return {}


def _read_writer_output(callback_context: CallbackContext) -> dict:
    preferred_keys = [
        "writer_output",
        "writer",
        "writer_result",
        "writer_agent_output",
        "writer_agent.writer_output",
    ]
    for key in preferred_keys:
        try:
            raw = callback_context.state.get(key, {})
        except Exception:
            raw = {}
        found = _find_writer_mapping(raw)
        if found:
            return found

    for _, raw in _iter_state_items(callback_context):
        found = _find_writer_mapping(raw)
        if found:
            return found
    return {}


def _build_narration_from_markdown(markdown_text: str) -> str:
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", markdown_text or "")
    text = re.sub(r"(?m)^\s*[-*•]\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    clipped, _ = _clean_tts_text(text, max_chars=1200)
    return clipped


def _fallback_generate_thai_report_from_research(research: dict) -> str:
    source_url = (
        (research.get("source_url") or research.get("original_url") or "").strip()
    )
    content = (research.get("content") or "").strip()
    if not content:
        return ""
    api_key = _get_google_api_key()
    if not api_key:
        return ""

    truncated = content[:9000]
    prompt = (
        "สร้างรายงานข่าวภาษาไทยจากข้อมูลข่าวดิบด้านล่าง\n"
        "รูปแบบผลลัพธ์:\n"
        "1) หัวข้อข่าวภาษาไทย 1 บรรทัด\n"
        "2) สรุปย่อ 5 bullet\n"
        "3) เนื้อหาข่าวภาษาไทยแบบเต็มและละเอียด (คงข้อเท็จจริง ตัวเลข และชื่อบุคคล/องค์กร)\n"
        "4) บรรทัดสุดท้ายเป็น 'แหล่งข่าว: <url>'\n"
        "ห้ามแต่งข้อมูลใหม่ และตอบเป็นข้อความไทยล้วน\n\n"
        f"URL: {source_url}\n"
        "เนื้อหาต้นฉบับ:\n"
        f"{truncated}"
    )

    models_to_try: list[str] = []
    for candidate in (_CURRENT_RUNTIME_MODEL, _FALLBACK_MODEL, _DEFAULT_MODEL):
        model_name = (candidate or "").strip()
        if model_name and model_name not in models_to_try:
            models_to_try.append(model_name)

    client = genai.Client(api_key=api_key)
    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=8192,
                    response_modalities=["TEXT"],
                ),
            )
            text = (getattr(response, "text", None) or "").strip()
            if text:
                return text
        except Exception:
            continue
    return ""


def _extract_first_url(text: str) -> str:
    match = _URL_RE.search(text or "")
    if not match:
        return ""
    return _normalize_url(match.group(0))


def _read_pipeline_error(callback_context: CallbackContext) -> str:
    preferred_keys = [
        "writer_output",
        "analysis_output",
        "research_output",
        "writer_result",
        "analysis_result",
        "research_result",
    ]
    for key in preferred_keys:
        try:
            raw = callback_context.state.get(key, {})
        except Exception:
            raw = {}
        data = _coerce_mapping(raw)
        if not data:
            continue
        error_text = str(data.get("error") or "").strip()
        status_text = str(data.get("status") or "").strip().lower()
        if error_text:
            return error_text
        if status_text == "error":
            return f"{key} returned status=error"
    return ""


def _looks_like_failure_message(text: str) -> bool:
    lower = (text or "").strip().lower()
    if not lower:
        return False
    failure_tokens = [
        "error",
        "failed",
        "unavailable",
        "ข้อผิดพลาด",
        "ไม่สามารถ",
        "ล้มเหลว",
        "กรุณาลองอีกครั้ง",
        "ขออภัย",
    ]
    return any(token in lower for token in failure_tokens)


def _looks_like_non_report_output(text: str) -> bool:
    body = (text or "").strip()
    if not body:
        return True
    lower = body.lower()
    generic_tokens = [
        "capabilities:",
        "please send a news url",
        "send a news url",
        "send url",
        "ส่งลิงก์ข่าวมาได้เลย",
        "กรุณาส่ง url",
    ]
    if any(token in lower for token in generic_tokens):
        return True

    has_report_markers = (
        ("แหล่งข่าว" in body)
        or ("http://" in lower)
        or ("https://" in lower)
        or ("•" in body)
        or ("\n-" in body)
        or ("\n## " in body)
        or ("\n# " in body)
    )
    if not has_report_markers and len(body) < 260:
        return True
    return False


async def _ensure_auto_tts_artifact(
    callback_context: CallbackContext,
) -> tuple[str, str]:
    writer = _read_writer_output(callback_context)
    narration = (writer.get("narration_text_th") or "").strip()
    if not narration:
        return "", ""

    prepared, truncated = _clean_tts_text(narration)
    if not prepared:
        return "", ""

    run_id = str(callback_context.state.get("workflow_run_id", "")).strip() or "__legacy__"
    audio_run_id = str(callback_context.state.get("audio_generated_run_id", "")).strip()
    cached = callback_context.state.get("last_tts_cache", {})
    if (
        audio_run_id == run_id
        and isinstance(cached, dict)
        and cached.get("filename")
        and cached.get("version")
    ):
        return str(cached.get("filename")), str(cached.get("version"))

    text_fingerprint = hashlib.sha256(prepared.encode("utf-8")).hexdigest()
    if (
        isinstance(cached, dict)
        and cached.get("fingerprint") == text_fingerprint
        and cached.get("filename")
        and cached.get("version")
    ):
        callback_context.state["audio_generated_run_id"] = run_id
        return str(cached.get("filename")), str(cached.get("version"))

    try:
        audio_bytes = _synthesize_thai_speech_mp3(prepared)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return "", ""
    if not audio_bytes:
        return "", ""

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"thai_news_tts_{ts}.mp3"
    artifact = types.Part(
        inline_data=types.Blob(mime_type="audio/mpeg", data=audio_bytes)
    )
    try:
        version = await callback_context.save_artifact(
            filename=filename,
            artifact=artifact,
            custom_metadata={"kind": "tts", "language": "th"},
        )
    except Exception:
        return "", ""

    callback_context.state["last_tts_cache"] = {
        "fingerprint": text_fingerprint,
        "filename": filename,
        "version": str(version),
        "chars_used": str(len(prepared)),
        "truncated": "true" if truncated else "false",
    }
    callback_context.state["audio_generated_run_id"] = run_id
    callback_context.state["workflow_stage"] = "Audio Ready"
    return filename, str(version)


def _append_full_article_if_needed(
    callback_context: CallbackContext, current_text: str
) -> str:
    try:
        stage = str(callback_context.state.get("workflow_stage", "")).strip()
    except Exception:
        stage = ""
    if stage and stage not in {"Published", "Audio Ready"}:
        return current_text

    analysis = _read_analysis_output(callback_context)
    thai_body = (analysis.get("translated_body_th") or "").strip()
    if not thai_body:
        return current_text

    if thai_body[:160] and thai_body[:160] in current_text:
        return current_text

    research = _read_research_output(callback_context)
    source_url = (
        research.get("source_url")
        or research.get("original_url")
        or ""
    ).strip()
    title = (research.get("title") or "").strip()
    published = (research.get("published_time") or "").strip()

    meta_lines: list[str] = []
    if title:
        meta_lines.append(f"ชื่อข่าว: {title}")
    if source_url:
        meta_lines.append(f"ที่มา: {source_url}")
    if published:
        meta_lines.append(f"เวลาเผยแพร่: {published}")

    appendix = "## เนื้อหาข่าวฉบับเต็ม (ภาษาไทย)\n"
    if meta_lines:
        appendix += "\n".join(meta_lines) + "\n\n"
    appendix += thai_body
    return f"{current_text.rstrip()}\n\n---\n\n{appendix}"


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_notion_mcp_enabled() -> bool:
    return _is_truthy(os.getenv("ENABLE_NOTION_MCP"))


def _get_notion_bearer_token() -> str:
    return (
        os.getenv("NOTION_MCP_BEARER_TOKEN")
        or os.getenv("NOTION_API_KEY")
        or ""
    ).strip()


def _extract_notion_page_id(raw_url: str) -> str:
    text = (raw_url or "").strip()
    if not text:
        return ""
    match = _NOTION_PAGE_ID_RE.search(text)
    if not match:
        return ""
    page_id = match.group(1).replace("-", "").lower()
    if len(page_id) != 32:
        return ""
    return f"{page_id[0:8]}-{page_id[8:12]}-{page_id[12:16]}-{page_id[16:20]}-{page_id[20:32]}"


def _build_notion_headers(token: str, notion_version: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Notion-Version": notion_version,
        "Content-Type": "application/json",
    }


def _notion_request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict | None = None,
    timeout: int = 20,
) -> tuple[dict, str]:
    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="ignore")
        parsed = json.loads(raw or "{}")
        if isinstance(parsed, dict):
            return parsed, ""
        return {}, "Invalid JSON response from Notion API"
    except HTTPError as err:
        try:
            detail = err.read().decode("utf-8", errors="ignore")
        except Exception:
            detail = ""
        return {}, f"HTTP {err.code}: {detail[:240]}"
    except (URLError, TimeoutError, OSError, ValueError) as err:
        return {}, f"{type(err).__name__}: {err}"


def _to_notion_text_blocks(markdown_text: str) -> list[dict]:
    blocks: list[dict] = []
    for raw_line in (markdown_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        block_type = "paragraph"
        content = line
        if line.startswith("## "):
            block_type = "heading_2"
            content = line[3:].strip()
        elif line.startswith("# "):
            block_type = "heading_1"
            content = line[2:].strip()
        elif line.startswith(("-", "*", "•")):
            block_type = "bulleted_list_item"
            content = line[1:].strip()
        if not content:
            continue
        while content:
            chunk = content[:1600]
            content = content[1600:]
            blocks.append(
                {
                    "object": "block",
                    "type": block_type,
                    block_type: {
                        "rich_text": [
                            {"type": "text", "text": {"content": chunk}}
                        ]
                    },
                }
            )
    return blocks


def _build_notion_direct_init_error() -> str:
    token = _get_notion_bearer_token()
    if not token:
        return "missing NOTION_MCP_BEARER_TOKEN (or NOTION_API_KEY)"
    notion_version = os.getenv("NOTION_VERSION", "2022-06-28").strip()
    ok, err = _validate_notion_bearer_token(token, notion_version)
    if not ok:
        return err
    return ""


def save_latest_summary_to_notion(
    notion_page_url: str,
    tool_context: ToolContext,
) -> dict[str, str]:
    """Save latest writer_output.final_markdown_th into a Notion page."""
    page_id = _extract_notion_page_id(notion_page_url)
    if not page_id:
        return {
            "ok": "false",
            "error": "Invalid Notion page URL",
            "page_id": "",
            "blocks_appended": "0",
        }

    writer = _read_writer_output(tool_context)
    content = (writer.get("final_markdown_th") or "").strip()
    if not content:
        return {
            "ok": "false",
            "error": "No latest summary in session (writer_output is empty)",
            "page_id": page_id,
            "blocks_appended": "0",
        }

    token = _get_notion_bearer_token()
    if not token:
        return {
            "ok": "false",
            "error": "Missing NOTION_MCP_BEARER_TOKEN (or NOTION_API_KEY)",
            "page_id": page_id,
            "blocks_appended": "0",
        }
    notion_version = os.getenv("NOTION_VERSION", "2022-06-28").strip()
    headers = _build_notion_headers(token, notion_version)

    _, read_error = _notion_request_json(
        "GET",
        f"https://api.notion.com/v1/pages/{page_id}",
        headers=headers,
    )
    if read_error:
        return {
            "ok": "false",
            "error": f"Cannot access page: {read_error}",
            "page_id": page_id,
            "blocks_appended": "0",
        }

    blocks = _to_notion_text_blocks(content)
    if not blocks:
        return {
            "ok": "false",
            "error": "Summary text could not be converted to Notion blocks",
            "page_id": page_id,
            "blocks_appended": "0",
        }

    appended = 0
    for i in range(0, len(blocks), 50):
        chunk = blocks[i : i + 50]
        _, append_error = _notion_request_json(
            "PATCH",
            f"https://api.notion.com/v1/blocks/{page_id}/children",
            headers=headers,
            payload={"children": chunk},
        )
        if append_error:
            return {
                "ok": "false",
                "error": f"Append failed after {appended} blocks: {append_error}",
                "page_id": page_id,
                "blocks_appended": str(appended),
            }
        appended += len(chunk)

    try:
        tool_context.actions.skip_summarization = True
    except Exception:
        pass
    tool_context.state["notion_last_saved_page_id"] = page_id
    tool_context.state["notion_last_saved_blocks"] = str(appended)
    return {
        "ok": "true",
        "error": "",
        "page_id": page_id,
        "blocks_appended": str(appended),
    }


def _validate_notion_bearer_token(
    bearer_token: str, notion_version: str
) -> tuple[bool, str]:
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Notion-Version": notion_version,
    }
    request = Request("https://api.notion.com/v1/users/me", headers=headers)
    try:
        with urlopen(request, timeout=8) as response:
            status = getattr(response, "status", 200)
        if status >= 400:
            return False, f"Token validation failed: HTTP {status}"
        return True, ""
    except HTTPError as err:
        return False, f"Token validation failed: HTTP {err.code}"
    except (URLError, TimeoutError, OSError, ValueError) as err:
        return False, f"Token validation failed: {type(err).__name__}"


def _root_before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Fast-path for non-URL prompts to avoid wasting model quota."""
    _ = callback_context
    user_text = _extract_latest_user_text(llm_request)
    callback_context.state["last_user_text"] = user_text
    latest_url = _extract_first_url(user_text)
    if latest_url:
        callback_context.state["last_user_url"] = latest_url
    callback_context.state["last_user_intent_notion"] = (
        "1" if _NOTION_HINT_RE.search(user_text) else "0"
    )
    if not user_text:
        return _build_text_response(
            "ส่งลิงก์ข่าวมาได้เลย ผมจะสรุป แปลไทย และทำไฟล์เสียง MP3 ให้ใน Artifacts"
        )

    # Route Notion intent first, even when user text contains URLs.
    if _NOTION_HINT_RE.search(user_text):
        if not _is_notion_mcp_enabled():
            return _build_text_response(
                "Notion integration is disabled.\n"
                "Set ENABLE_NOTION_MCP=1 in .env and restart adk web to enable it."
            )
        if _NOTION_MCP_INIT_ERROR:
            return _build_text_response(
                "Notion integration is enabled but not ready.\n"
                f"{_NOTION_MCP_INIT_ERROR}\n"
                "Check token/permissions and restart adk web."
            )
        return None

    if _NOTION_PAGE_URL_RE.search(user_text):
        return _build_text_response(
            "This URL looks like a Notion page URL.\n"
            "If you want me to save the latest summary there, please include 'Notion' or 'save to Notion' in your message."
        )

    if _URL_RE.search(user_text):
        if _RUNTIME_CONFIG_ERROR:
            return _build_text_response(_RUNTIME_CONFIG_ERROR)
        return None

    return _build_text_response(
        "ความสามารถ:\n"
        "1) อ่านข่าวจาก URL\n"
        "2) สรุปข่าว + ใส่เนื้อหาไทยแบบเต็ม\n"
        "3) แปลเป็นภาษาไทย\n"
        "4) สร้างไฟล์เสียง MP3 ใน Artifacts\n"
        "5) บันทึกสรุปลง Notion (เมื่อเปิดใช้งาน)\n\n"
        "ส่งลิงก์ข่าวมาได้เลย เช่น https://example.com/news/..."
    )


async def _root_after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """Guarantee visible text and auto-generate one TTS artifact per run."""
    if llm_response.error_code:
        return None
    if llm_response.partial:
        return None

    original_text = ""
    has_function_event = False
    content = llm_response.content
    if content and content.parts:
        text_parts = [
            (part.text or "").strip() for part in content.parts
            if (part.text or "").strip()
        ]
        has_function_event = any(
            part.function_call or part.function_response
            for part in content.parts
        )
        original_text = "\n\n".join(text_parts).strip()

    try:
        notion_intent = (
            str(callback_context.state.get("last_user_intent_notion", "0")).strip()
            == "1"
        )
    except Exception:
        notion_intent = False

    # Let ADK complete tool-call loop first.
    if has_function_event and not original_text:
        return None

    final_text = original_text
    research_output = _read_research_output(callback_context)
    writer_output = _read_writer_output(callback_context)
    writer_markdown = (writer_output.get("final_markdown_th") or "").strip()

    if not final_text and writer_markdown and not notion_intent:
        final_text = writer_markdown

    try:
        stage = str(callback_context.state.get("workflow_stage", "")).strip()
    except Exception:
        stage = ""

    # Let normal tool-call loop continue unless we already have enough output.
    if (
        not final_text
        and has_function_event
        and stage not in {"Approved", "Audio Ready", "Published"}
    ):
        return None

    research_ok = bool((research_output.get("content") or "").strip())
    pipeline_error = _read_pipeline_error(callback_context)
    try:
        last_user_url = str(callback_context.state.get("last_user_url", "")).strip()
    except Exception:
        last_user_url = ""

    research_for_recovery = research_output
    if (
        not research_ok
        and last_user_url
        and (
            (final_text and _looks_like_failure_message(final_text))
            or not final_text
        )
    ):
        fetched = fetch_news_content(last_user_url)
        if (fetched.get("ok") or "").strip().lower() == "true":
            print("[pipeline] recovered research_output via direct refetch")
            research_for_recovery = {
                "status": "ok",
                "original_url": fetched.get("original_url", ""),
                "source_url": fetched.get("source_url", ""),
                "title": fetched.get("title", ""),
                "published_time": fetched.get("published_time", ""),
                "method": fetched.get("method", ""),
                "content": fetched.get("content", ""),
                "error": "",
            }
            callback_context.state["research_output"] = research_for_recovery
            research_ok = bool((research_for_recovery.get("content") or "").strip())

    need_recovery = (
        research_ok
        and not writer_markdown
        and (
            (final_text and _looks_like_failure_message(final_text))
            or (final_text and _looks_like_non_report_output(final_text))
            or not final_text
        )
    )

    if need_recovery:
        recovered_markdown = _fallback_generate_thai_report_from_research(
            research_for_recovery
        )
        if recovered_markdown:
            print("[pipeline] recovered final markdown via fallback generator")
            final_text = recovered_markdown
            source_url = (
                research_for_recovery.get("source_url")
                or research_for_recovery.get("original_url")
                or ""
            ).strip()
            callback_context.state["writer_output"] = {
                "status": "ok",
                "source_url": source_url,
                "final_markdown_th": recovered_markdown,
                "narration_text_th": _build_narration_from_markdown(
                    recovered_markdown
                ),
                "error": "",
            }
            writer_markdown = recovered_markdown

    if (
        final_text
        and _looks_like_failure_message(final_text)
        and research_ok
        and not writer_markdown
    ):
        hint = "ระบบดึงเนื้อหาข่าวจาก URL ได้แล้ว แต่ขั้นสรุป/แปลมีปัญหาชั่วคราว"
        if pipeline_error:
            hint = f"{hint} ({pipeline_error})"
        return _build_text_response(
            f"{hint}\nกรุณาส่งลิงก์เดิมอีกครั้ง หรือลองอีกครั้งใน 10-30 วินาที"
        )

    if not final_text:
        if notion_intent:
            return _build_text_response(
                "ยังไม่ได้ผลลัพธ์การบันทึก Notion ในรอบนี้\n"
                "ลองสั่งอีกครั้ง: บันทึกสรุปล่าสุดลง Notion หน้านี้ <notion-url>"
            )
        if research_ok:
            detail = f" ({pipeline_error})" if pipeline_error else ""
            return _build_text_response(
                f"ดึงข่าวจาก URL ได้แล้ว แต่ขั้นสรุป/แปลขัดข้องชั่วคราว{detail}\n"
                "กรุณาลองส่งลิงก์เดิมอีกครั้งใน 10-30 วินาที"
            )
        if last_user_url:
            return _build_text_response(
                f"ยังดึงข่าวจากลิงก์นี้ไม่ได้: {last_user_url}\n"
                "โปรดลองลิงก์บทความเต็ม (ไม่ใช่หน้า redirect/short URL) หรือส่งลิงก์อื่น"
            )
        return _build_text_response(
            "เกิดข้อผิดพลาดระหว่างสรุปข่าวครับ กรุณาส่งลิงก์อีกครั้ง"
        )

    if not notion_intent:
        final_text = _append_full_article_if_needed(callback_context, final_text)

        audio_filename, _audio_version = await _ensure_auto_tts_artifact(
            callback_context
        )
        if audio_filename and "Artifacts" not in final_text:
            final_text = (
                f"{final_text.rstrip()}\n\n"
                f"(ไฟล์เสียงแนบใน Artifacts: {audio_filename})"
            )

    if not notion_intent and not _looks_like_failure_message(final_text):
        callback_context.state["workflow_stage"] = "Published"

    if final_text != original_text:
        return _build_text_response(final_text)
    return None


research_agent = Agent(
    name="research_agent",
    model=_DEFAULT_MODEL,
    description="Fetch and normalize article content from URL.",
    instruction=_RESEARCH_INSTRUCTION,
    tools=[fetch_news_content, update_workflow_stage],
    input_schema=ResearchInput,
    output_schema=ResearchOutput,
    output_key="research_output",
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
)

analyzer_agent = Agent(
    name="analyzer_agent",
    model=_DEFAULT_MODEL,
    description="Analyze and translate fetched news to Thai insights.",
    instruction=_ANALYZER_INSTRUCTION,
    tools=[update_workflow_stage],
    input_schema=AnalysisInput,
    output_schema=AnalysisOutput,
    output_key="analysis_output",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=8192,
    ),
)

writer_agent = Agent(
    name="writer_agent",
    model=_DEFAULT_MODEL,
    description="Compose final Thai report and narration script.",
    instruction=_WRITER_INSTRUCTION,
    tools=[update_workflow_stage],
    input_schema=WriterInput,
    output_schema=WriterOutput,
    output_key="writer_output",
    generate_content_config=types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=8192,
    ),
)

narrator_agent = Agent(
    name="narrator_agent",
    model=_DEFAULT_MODEL,
    description="Generate Thai MP3 narration artifact from final script.",
    instruction=_NARRATOR_INSTRUCTION,
    tools=[create_thai_speech, update_workflow_stage],
    input_schema=NarratorInput,
    output_schema=NarratorOutput,
    output_key="narrator_output",
    generate_content_config=types.GenerateContentConfig(temperature=0.1),
)

_root_tools = [
    AgentTool(research_agent),
    AgentTool(analyzer_agent),
    AgentTool(writer_agent),
    save_latest_summary_to_notion,
    update_workflow_stage,
]
_NOTION_MCP_INIT_ERROR = _build_notion_direct_init_error()


def _switch_runtime_model(new_model: str) -> bool:
    global _CURRENT_RUNTIME_MODEL
    target = (new_model or "").strip()
    if not target or target == _CURRENT_RUNTIME_MODEL:
        return False

    switched = False
    for agent_name in (
        "research_agent",
        "analyzer_agent",
        "writer_agent",
        "narrator_agent",
        "root_agent",
    ):
        agent = globals().get(agent_name)
        if agent is None:
            continue
        try:
            if getattr(agent, "model", "") != target:
                agent.model = target
                switched = True
        except Exception:
            continue

    if switched:
        _CURRENT_RUNTIME_MODEL = target
    return switched


if not _is_notion_mcp_enabled():
    print("Notion Integration: disabled")
elif _NOTION_MCP_INIT_ERROR:
    print(f"Notion Integration: enabled (init failed: {_NOTION_MCP_INIT_ERROR})")
else:
    print("Notion Integration: enabled (init ok)")

print(f"Model: {_DEFAULT_MODEL}")
if _FALLBACK_MODEL:
    print(
        f"Model fallback on 503: {_FALLBACK_MODEL} "
        f"(enabled={str(_AUTO_FALLBACK_ON_503).lower()})"
    )
if _RUNTIME_CONFIG_ERROR:
    print("Google API key: missing")
else:
    print("Google API key: detected")


def _root_on_model_error_callback(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
    error: Exception,
) -> LlmResponse | None:
    """Return actionable model error messages."""
    _ = callback_context
    _ = llm_request
    error_text = str(error)
    lower = error_text.lower()
    short_error = error_text.replace("\n", " ").strip()
    if len(short_error) > 220:
        short_error = f"{short_error[:220]}..."

    if (
        "resource_exhausted" in lower
        or "429" in lower
        or "quota" in lower
    ):
        message = (
            "Model quota is temporarily exhausted.\n"
            "Please wait and try again, or switch to an API key with available quota."
        )
    elif (
        "503" in lower
        or "unavailable" in lower
        or "high demand" in lower
        or "backend unavailable" in lower
    ):
        switched = False
        if _AUTO_FALLBACK_ON_503 and _FALLBACK_MODEL:
            switched = _switch_runtime_model(_FALLBACK_MODEL)
        if switched:
            message = (
                "Model backend ของตัวหลักกำลังหนาแน่น (503 UNAVAILABLE)\n"
                f"ระบบสลับโมเดลอัตโนมัติเป็น {_FALLBACK_MODEL} แล้ว\n"
                "กรุณาส่ง URL เดิมอีกครั้งได้ทันที"
            )
            return _build_text_response(message)
        model_hint = ""
        if "preview" in _CURRENT_RUNTIME_MODEL.lower():
            model_hint = (
                "\nCurrent model is a preview variant and may return 503 during peak demand."
            )
        message = (
            "Model backend is temporarily overloaded (503 UNAVAILABLE).\n"
            "Please retry in 10-30 seconds."
            f"{model_hint}\n"
            "If it keeps failing, change API key/project and restart adk web."
        )
    elif (
        "api key not valid" in lower
        or "unauthenticated" in lower
        or "permission denied" in lower
        or "invalid api key" in lower
    ):
        message = (
            "Model call failed because the API key is invalid or unauthorized.\n"
            "Check GOOGLE_API_KEY in myagent/.env and restart adk web."
        )
    elif (
        "model" in lower
        and (
            "not found" in lower
            or "unsupported" in lower
            or "invalid argument" in lower
        )
    ):
        message = (
            f"Configured model ({_CURRENT_RUNTIME_MODEL}) may be unsupported.\n"
            "Set GEMINI_MODEL=gemini-3.1-flash-lite-preview in myagent/.env and restart adk web."
        )
    else:
        message = (
            "An unexpected error occurred while processing the news pipeline.\n"
            f"Details: {short_error}"
        )
    return _build_text_response(message)

root_agent = Agent(
    name="news_pipeline_orchestrator",
    model=_DEFAULT_MODEL,
    description="Orchestrate multi-agent Thai news read/translate/audio pipeline.",
    instruction=_ROOT_INSTRUCTION,
    tools=_root_tools,
    before_model_callback=_root_before_model_callback,
    after_model_callback=_root_after_model_callback,
    on_model_error_callback=_root_on_model_error_callback,
    on_tool_error_callback=_root_on_tool_error_callback,
    generate_content_config=types.GenerateContentConfig(
        response_modalities=["TEXT"],
        temperature=0.2,
        max_output_tokens=8192,
    ),
)





