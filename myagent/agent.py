from __future__ import annotations

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

from google.adk.agents.llm_agent import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import AgentTool
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_toolset import (
    StreamableHTTPConnectionParams,
)
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
        or "gemini-2.5-flash"
    ).strip() or "gemini-2.5-flash"


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

_MAX_ARTICLE_CHARS = 30000
_MAX_TTS_CHARS = 1800
_DEFAULT_MODEL = _get_configured_model()
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_NOTION_HINT_RE = re.compile(r"(?:\bnotion\b|à¹‚à¸™à¸Šà¸±à¸™|à¹‚à¸™à¸Šà¸±à¹ˆà¸™)", re.IGNORECASE)
_NOTION_MCP_INIT_ERROR = ""
_SEARCH_MCP_INIT_ERROR = ""
_WEB_MCP_INIT_ERROR = ""
_RUNTIME_CONFIG_ERROR = _build_runtime_config_error()


class SafeMcpToolset(McpToolset):
    """MCP toolset that fails open (returns no tools) instead of crashing."""

    async def get_tools(
        self,
        readonly_context: ReadonlyContext | None = None,
    ) -> list[BaseTool]:
        try:
            return await super().get_tools(readonly_context=readonly_context)
        except BaseException as err:  # pragma: no cover - defensive guard
            if isinstance(err, (KeyboardInterrupt, SystemExit)):
                raise
            print(
                f"[notion_mcp] unavailable in this request: {type(err).__name__}: {err}"
            )
            return []


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
        "content": body[:_MAX_ARTICLE_CHARS],
    }


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
                    "content": plain[:_MAX_ARTICLE_CHARS],
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
   - translated_body_th (Thai, readable, factual)
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
2) If analysis.status='error', return status='error'.
3) Build final_markdown_th using this order:
   - headline in Thai
   - 5 bullet summary in Thai
   - full Thai translation
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
- Fill audio_filename and audio_version from tool result.
- If TTS fails, return status='error' with error detail.
- JSON only.
""".strip()

_ROOT_INSTRUCTION = """
You are an orchestrator for a 4-agent news pipeline:
Research -> Analyzer -> Writer -> Narrator.

Behavior:
1) If user provides a news URL:
   - call research_agent with {"url": "..."}
   - if ok, call analyzer_agent with {"research": research_result}
   - if ok, call writer_agent with {"analysis": analysis_result}
   - if ok, call narrator_agent with {"writer": writer_result}
   - then call update_workflow_stage with stage='Published'
   - respond in Thai with narrator_result.final_markdown_th
   - add one short line that audio file is attached in Artifacts.
2) If user asks for external search/crawling/notion operations:
   - for web/news search, use tools with prefix "search_"
   - for page fetching/crawling, use tools with prefix "web_"
   - for workspace notes/pages, use tools with prefix "notion_"
   - summarize the action result in Thai
3) If any stage fails:
   - explain failure in Thai briefly
   - ask user for another URL
4) If no URL is provided:
   - ask user to provide a URL in Thai.

Rules:
- Always answer user in Thai.
- Do not fabricate facts.
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


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _is_notion_mcp_enabled() -> bool:
    return _is_truthy(os.getenv("ENABLE_NOTION_MCP"))


def _is_search_mcp_enabled() -> bool:
    return _is_truthy(os.getenv("ENABLE_SEARCH_MCP"))


def _is_web_mcp_enabled() -> bool:
    return _is_truthy(os.getenv("ENABLE_WEB_MCP"))


def _get_notion_bearer_token() -> str:
    return (
        os.getenv("NOTION_MCP_BEARER_TOKEN")
        or os.getenv("NOTION_API_KEY")
        or ""
    ).strip()


def _get_bearer_token(*keys: str) -> str:
    for key in keys:
        value = (os.getenv(key) or "").strip()
        if value:
            return value
    return ""


def _parse_tool_filter(raw_value: str) -> list[str] | None:
    values = [x.strip() for x in (raw_value or "").split(",") if x.strip()]
    return values or None


def _build_bearer_header_provider(bearer_token: str):
    def _header_provider(
        readonly_context: ReadonlyContext,
    ) -> dict[str, str]:
        _ = readonly_context
        headers: dict[str, str] = {}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        return headers

    return _header_provider


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


def _build_notion_header_provider(
    bearer_token: str, notion_version: str
):
    def _header_provider(
        readonly_context: ReadonlyContext,
    ) -> dict[str, str]:
        _ = readonly_context
        headers: dict[str, str] = {}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        if notion_version:
            headers["Notion-Version"] = notion_version
        return headers

    return _header_provider


def _build_notion_toolset() -> tuple[McpToolset | None, str]:
    if not _is_notion_mcp_enabled():
        return None, ""

    notion_url = (
        os.getenv("NOTION_MCP_URL", "https://mcp.notion.com/mcp").strip()
    )
    if not notion_url:
        return None, "NOTION_MCP_URL is empty"

    bearer_token = _get_notion_bearer_token()
    if not bearer_token:
        # Hosted Notion MCP requires auth. Skip tool registration if missing.
        return None, "missing NOTION_MCP_BEARER_TOKEN"

    notion_version = os.getenv("NOTION_VERSION", "2022-06-28").strip()
    is_valid, validation_error = _validate_notion_bearer_token(
        bearer_token=bearer_token,
        notion_version=notion_version,
    )
    if not is_valid:
        return None, validation_error

    raw_filter = os.getenv("NOTION_MCP_TOOL_FILTER", "").strip()
    tool_filter = [x.strip() for x in raw_filter.split(",") if x.strip()]
    if not tool_filter:
        tool_filter = None

    header_provider = None
    if bearer_token or notion_version:
        header_provider = _build_notion_header_provider(
            bearer_token=bearer_token,
            notion_version=notion_version,
        )

    return SafeMcpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=notion_url,
            timeout=15.0,
            sse_read_timeout=300.0,
        ),
        tool_filter=tool_filter,
        tool_name_prefix="notion",
        header_provider=header_provider,
    ), ""


def _build_optional_mcp_toolset(
    *,
    enabled: bool,
    url_env: str,
    default_url: str,
    token_env_keys: tuple[str, ...],
    tool_filter_env: str,
    tool_name_prefix: str,
    require_bearer: bool = False,
) -> tuple[McpToolset | None, str]:
    if not enabled:
        return None, ""

    mcp_url = (os.getenv(url_env, default_url) or "").strip()
    if not mcp_url:
        return None, f"{url_env} is empty"

    bearer_token = _get_bearer_token(*token_env_keys)
    if require_bearer and not bearer_token:
        return None, f"missing {token_env_keys[0]}"

    header_provider = None
    if bearer_token:
        header_provider = _build_bearer_header_provider(bearer_token)

    tool_filter = _parse_tool_filter(os.getenv(tool_filter_env, ""))
    return SafeMcpToolset(
        connection_params=StreamableHTTPConnectionParams(
            url=mcp_url,
            timeout=15.0,
            sse_read_timeout=300.0,
        ),
        tool_filter=tool_filter,
        tool_name_prefix=tool_name_prefix,
        header_provider=header_provider,
    ), ""


def _build_search_toolset() -> tuple[McpToolset | None, str]:
    return _build_optional_mcp_toolset(
        enabled=_is_search_mcp_enabled(),
        url_env="SEARCH_MCP_URL",
        default_url="",
        token_env_keys=("SEARCH_MCP_BEARER_TOKEN",),
        tool_filter_env="SEARCH_MCP_TOOL_FILTER",
        tool_name_prefix="search",
    )


def _build_web_toolset() -> tuple[McpToolset | None, str]:
    return _build_optional_mcp_toolset(
        enabled=_is_web_mcp_enabled(),
        url_env="WEB_MCP_URL",
        default_url="",
        token_env_keys=("WEB_MCP_BEARER_TOKEN",),
        tool_filter_env="WEB_MCP_TOOL_FILTER",
        tool_name_prefix="web",
    )


# Override callback text with stable MCP fallback behavior.
_SEARCH_HINT_RE = re.compile(
    r"(?:\bsearch\b|google|news search|find sources|\u0e04\u0e49\u0e19\u0e2b\u0e32|\u0e2b\u0e32\u0e02\u0e48\u0e32\u0e27)",
    re.IGNORECASE,
)
_WEB_HINT_RE = re.compile(
    r"(?:\bcrawl(?:er|ing)?\b|\bfetch\b|scrape|extract webpage|web page|\u0e14\u0e36\u0e07\u0e40\u0e27\u0e47\u0e1a|\u0e2d\u0e48\u0e32\u0e19\u0e40\u0e27\u0e47\u0e1a)",
    re.IGNORECASE,
)
_NOTION_HINT_RE = re.compile(
    r"(?:\bnotion\b|\u0e42\u0e19\u0e0a\u0e31\u0e19|\u0e42\u0e19\u0e0a\u0e31\u0e48\u0e19)",
    re.IGNORECASE,
)


def _root_before_model_callback(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    """Fast-path for non-URL prompts to avoid wasting model quota."""
    _ = callback_context
    user_text = _extract_latest_user_text(llm_request)
    if not user_text:
        return _build_text_response(
            "Send a news URL and I will summarize, translate to Thai, and create an MP3 in Artifacts."
        )

    if _URL_RE.search(user_text):
        if _RUNTIME_CONFIG_ERROR:
            return _build_text_response(_RUNTIME_CONFIG_ERROR)
        return None

    if _SEARCH_HINT_RE.search(user_text):
        if not _is_search_mcp_enabled():
            return _build_text_response(
                "Search MCP is disabled.\n"
                "Set ENABLE_SEARCH_MCP=1 and SEARCH_MCP_URL in .env, then restart adk web."
            )
        if _SEARCH_MCP_INIT_ERROR:
            return _build_text_response(
                "Search MCP is enabled but not ready.\n"
                f"{_SEARCH_MCP_INIT_ERROR}\n"
                "Check MCP connection settings and restart adk web."
            )
        return None

    if _WEB_HINT_RE.search(user_text):
        if not _is_web_mcp_enabled():
            return _build_text_response(
                "Web MCP is disabled.\n"
                "Set ENABLE_WEB_MCP=1 and WEB_MCP_URL in .env, then restart adk web."
            )
        if _WEB_MCP_INIT_ERROR:
            return _build_text_response(
                "Web MCP is enabled but not ready.\n"
                f"{_WEB_MCP_INIT_ERROR}\n"
                "Check MCP connection settings and restart adk web."
            )
        return None

    if _NOTION_HINT_RE.search(user_text):
        if not _is_notion_mcp_enabled():
            return _build_text_response(
                "Notion MCP is temporarily disabled in stable mode.\n"
                "Set ENABLE_NOTION_MCP=1 in .env and restart adk web to enable it."
            )
        if _NOTION_MCP_INIT_ERROR:
            return _build_text_response(
                "Notion MCP is enabled but not ready.\n"
                f"{_NOTION_MCP_INIT_ERROR}\n"
                "Check token/permissions and restart adk web."
            )
        return None

    return _build_text_response(
        "Capabilities:\n"
        "1) Read news from URL\n"
        "2) Summarize and translate to Thai\n"
        "3) Create MP3 audio in Artifacts\n"
        "4) Use Search/Web/Notion MCP when enabled\n\n"
        "Please send a news URL, e.g. https://example.com/news/..."
    )


def _root_after_model_callback(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """Guarantee visible fallback text when model returns an empty reply."""
    _ = callback_context
    if llm_response.error_code:
        return None
    if llm_response.partial:
        return None

    content = llm_response.content
    if content and content.parts:
        has_text = any((part.text or "").strip() for part in content.parts)
        has_function_event = any(
            part.function_call or part.function_response
            for part in content.parts
        )
        if has_text or has_function_event:
            return None

    return _build_text_response(
        "I am ready to read news, but this turn returned an empty response.\n"
        "Please send the news URL again, for example: https://example.com/news/..."
    )


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
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
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
    generate_content_config=types.GenerateContentConfig(temperature=0.2),
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
    AgentTool(narrator_agent),
    update_workflow_stage,
]

search_toolset, _SEARCH_MCP_INIT_ERROR = _build_search_toolset()
if search_toolset:
    _root_tools.append(search_toolset)

web_toolset, _WEB_MCP_INIT_ERROR = _build_web_toolset()
if web_toolset:
    _root_tools.append(web_toolset)

notion_toolset, _NOTION_MCP_INIT_ERROR = _build_notion_toolset()
if notion_toolset:
    _root_tools.append(notion_toolset)

if not _is_search_mcp_enabled():
    print("Search MCP: disabled")
elif _SEARCH_MCP_INIT_ERROR:
    print(f"Search MCP: enabled (init failed: {_SEARCH_MCP_INIT_ERROR})")
else:
    print("Search MCP: enabled (init ok)")

if not _is_web_mcp_enabled():
    print("Web MCP: disabled")
elif _WEB_MCP_INIT_ERROR:
    print(f"Web MCP: enabled (init failed: {_WEB_MCP_INIT_ERROR})")
else:
    print("Web MCP: enabled (init ok)")

if not _is_notion_mcp_enabled():
    print("Notion MCP: disabled")
elif _NOTION_MCP_INIT_ERROR:
    print(f"Notion MCP: enabled (init failed: {_NOTION_MCP_INIT_ERROR})")
else:
    print("Notion MCP: enabled (init ok)")

print(f"Model: {_DEFAULT_MODEL}")
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

    if "resource_exhausted" in lower or "429" in lower or "quota" in lower:
        message = (
            "Model quota is temporarily exhausted.\n"
            "Please wait and try again, or switch to an API key with available quota."
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
            f"Configured model ({_DEFAULT_MODEL}) may be unsupported.\n"
            "Set GEMINI_MODEL=gemini-2.5-flash in myagent/.env and restart adk web."
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
        max_output_tokens=4096,
    ),
)

