import os
import json
import base64
import re
import time
import requests
from typing import List, Optional, TypedDict, Literal

import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage

dotenv.load_dotenv()

# ─── Configure APIs ───────────────────────────────────────────────────────────
story_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.8)

HF_TOKEN  = os.getenv("HF_TOKEN")
HF_URL    = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


# ─── Shared State ─────────────────────────────────────────────────────────────
class StoryState(TypedDict):
    topic:          Optional[str]
    story:          Optional[str]
    image_prompts:  Optional[List[str]]
    image_b64:      Optional[List[str]]
    messages:       List[AnyMessage]
    next_step:      Optional[str]
    error:          Optional[str]


# ─── 1. Orchestrator ──────────────────────────────────────────────────────────
def orchestrator(state: StoryState) -> StoryState:
    has_story  = bool(state.get("story"))
    has_images = bool(state.get("image_b64"))

    if not has_story:
        state["next_step"] = "story_writer"
    elif not has_images:
        state["next_step"] = "image_gen"
    else:
        state["next_step"] = "end"

    print(f"[Orchestrator] → {state['next_step']}")
    return state


def orch_router(state: StoryState) -> Literal["story_writer", "image_gen", "end"]:
    return state["next_step"]  # type: ignore


# ─── 2. Story Writer ──────────────────────────────────────────────────────────
STORY_SYS = """You are a creative story writer.
Given a topic, write an engaging short story (300-500 words).
Then provide exactly 3 vivid image prompts capturing key scenes.

Respond ONLY with valid JSON — no markdown fences, no extra text:
{
  "story": "<full story text>",
  "image_prompts": ["<prompt 1>", "<prompt 2>", "<prompt 3>"]
}"""

def story_writer(state: StoryState) -> StoryState:
    print("[Story Writer] Generating story...")
    messages = [
        SystemMessage(content=STORY_SYS),
        HumanMessage(content=f"Write a story about: {state['topic']}"),
    ]

    response = story_llm.invoke(messages)
    raw = response.content.strip()
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()

    try:
        data = json.loads(raw)
        state["story"]         = data["story"]
        state["image_prompts"] = data["image_prompts"]
        print(f"[Story Writer] Done. {len(state['image_prompts'])} prompts ready.")
    except (json.JSONDecodeError, KeyError):
        state["story"] = raw
        state["image_prompts"] = [
            f"A dramatic scene from a story about {state['topic']}, cinematic lighting",
            f"A key moment in a story about {state['topic']}, illustrated art style",
            f"The climax of a story about {state['topic']}, vivid colors",
        ]

    state["messages"] = state.get("messages", []) + [response]
    return state


# ─── 3. Image Generator (HuggingFace FLUX.1-schnell — free) ──────────────────
def _hf_generate(prompt: str, retries: int = 3) -> bytes:
    """Call HF Inference API with retry on model-loading 503."""
    for attempt in range(retries):
        resp = requests.post(
            HF_URL,
            headers=HF_HEADERS,
            json={"inputs": prompt},
            timeout=120,
        )
        if resp.status_code == 200:
            return resp.content
        elif resp.status_code == 503:
            # Model is loading — wait and retry
            wait = int(resp.headers.get("X-Wait-For-Model", 20))
            print(f"  Model loading, retrying in {wait}s...")
            time.sleep(wait)
        else:
            raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
    raise Exception("Max retries reached — model still loading")


def image_gen(state: StoryState) -> StoryState:
    """Generate images using HuggingFace FLUX.1-schnell (free tier)."""
    print("[Image Gen] Generating images via HuggingFace FLUX.1-schnell...")
    prompts   = state.get("image_prompts") or []
    b64_images: List[str] = []

    for i, prompt in enumerate(prompts):
        print(f"[Image Gen] Image {i+1}/{len(prompts)}...")
        try:
            img_bytes = _hf_generate(prompt)
            img_b64   = base64.b64encode(img_bytes).decode("utf-8")
            b64_images.append(f"data:image/jpeg;base64,{img_b64}")
            print(f"[Image Gen] Image {i+1} done ✓")
            time.sleep(2)   # small pause between requests
        except Exception as e:
            print(f"[Image Gen] Error on prompt {i+1}: {e}")
            b64_images.append(f"ERROR:{e}")

    state["image_b64"] = b64_images
    print(f"[Image Gen] Done. {len(b64_images)} images generated.")
    return state