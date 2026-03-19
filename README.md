# 📖 AI Story Teller

A multi-agent LangGraph pipeline that generates stories and illustrates them with AI images — built entirely with **free APIs**.

## Architecture

```
START → Orchestrator → Story Writer → Orchestrator → Image Gen → Orchestrator → END
```

| Agent | Model | Provider |
|-------|-------|----------|
| Orchestrator | deterministic routing | — |
| Story Writer | llama-3.3-70b-versatile | Groq (free) |
| Image Generator | FLUX.1-schnell | HuggingFace (free) |

## Project Structure

```
Story_to_images_by_Agents/
├── agents.py        ← all agent logic (orchestrator, story writer, image gen)
├── main.py          ← LangGraph workflow wiring + CLI entry point
├── app.py           ← Streamlit web UI
├── requirements.txt
├── .env             ← your API keys (never commit this)
├── .env.example     ← template
└── .gitignore
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Get free API keys

**Groq** (story generation):
- Go to https://console.groq.com
- Create account → API Keys → Create key
- Copy the key starting with `gsk_...`

**HuggingFace** (image generation):
- Go to https://huggingface.co/settings/tokens
- Click **"Create new token"**
- Token type: **Fine-grained**
- Enable permission: ✅ **"Make calls to Inference Providers"**
- Copy the key starting with `hf_...`

### 3. Create your `.env` file
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

> ⚠️ No quotes around the values. Just plain text.

### 4. Run the app

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**CLI mode:**
```bash
python main.py
```

> ⚠️ Never run `python app.py` — Streamlit apps must be launched with `streamlit run`.

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `HTTP 410` from HuggingFace | Old API endpoint | Use `router.huggingface.co` (already fixed) |
| `HTTP 403` from HuggingFace | Token missing permissions | Create Fine-grained token with "Inference Providers" enabled |
| `HTTP 503` from HuggingFace | Model is cold-starting | Auto-retries after wait — normal on first call |
| Streamlit warnings in terminal | Ran with `python app.py` | Use `streamlit run app.py` instead |

## Notes

- First image generation may take ~20–30 seconds (FLUX model cold start) — subsequent calls are faster
- The virtual environment folder (`agent1/`) should sit **outside** the project folder
- Delete `__pycache__/` if you update `agents.py` and changes don't seem to apply
