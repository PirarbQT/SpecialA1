# AI News Reader (ADK Web + Notion)

This project provides a Thai news workflow on ADK Web:
- Read article from URL
- Summarize and translate to Thai
- Include full Thai body in output
- Generate Thai MP3 narration (Artifacts)
- Save latest summary to Notion via direct Notion API tool

## Integration scope
- Search MCP / Web MCP were removed from runtime
- Notion save is now a single dedicated tool (`save_latest_summary_to_notion`)

## Run
```powershell
$env:UV_LINK_MODE='copy'
uv sync
uv run adk web
```

## Environment (`myagent/.env`)
```env
GOOGLE_GENAI_USE_VERTEXAI=0
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
GEMINI_MODEL=gemini-3.1-flash-lite-preview
AUTO_FALLBACK_ON_503=1
GEMINI_FALLBACK_MODEL=gemini-2.5-flash
MAX_ARTICLE_CHARS=14000

ENABLE_NOTION_MCP=1
NOTION_MCP_BEARER_TOKEN=YOUR_NOTION_INTEGRATION_TOKEN
# or NOTION_API_KEY=YOUR_NOTION_INTEGRATION_TOKEN
NOTION_VERSION=2022-06-28
```

## Make Notion write work
1. Create Notion integration token in Notion Developers.
2. Put token into `NOTION_MCP_BEARER_TOKEN` (or `NOTION_API_KEY`).
3. Share target page/database with that integration.
4. Restart `uv run adk web`.

## Expected startup logs
- `Notion Integration: enabled (init ok)` = ready
- `missing NOTION_MCP_BEARER_TOKEN...` = token missing
- `Token validation failed ...` = token invalid

## Example prompts
```text
สรุปและแปลข่าวนี้เป็นภาษาไทย พร้อมทำเสียงอ่าน
https://edition.cnn.com/...
```

```text
บันทึกสรุปล่าสุดลง Notion หน้านี้ https://www.notion.so/...
```
