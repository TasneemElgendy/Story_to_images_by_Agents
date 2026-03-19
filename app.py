import streamlit as st
from main import build_workflow
from agent import StoryState

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Story Teller",
    page_icon="📖",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #888;
        margin-bottom: 2rem;
        font-size: 1.05rem;
    }
    .story-box {
        background: #1e1e2e;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1.5rem 2rem;
        font-size: 1.05rem;
        line-height: 1.8;
        color: #cdd6f4;
    }
    .prompt-badge {
        background: #313244;
        border-radius: 6px;
        padding: 0.4rem 0.8rem;
        font-size: 0.85rem;
        color: #a6e3a1;
        margin-bottom: 0.4rem;
        display: inline-block;
    }
    .step-badge {
        background: #45475a;
        border-radius: 20px;
        padding: 0.3rem 1rem;
        font-size: 0.8rem;
        color: #cba6f7;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📖 AI Story Teller</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Multi-agent pipeline · Story Writer + Image Generator</div>', unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "graph" not in st.session_state:
    st.session_state.graph = build_workflow()

if "result" not in st.session_state:
    st.session_state.result = None

# ── Input panel ───────────────────────────────────────────────────────────────
with st.container():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        topic = st.text_input(
            "✨ Enter your story topic",
            placeholder="e.g. A lonely astronaut who discovers life on Mars…",
            label_visibility="visible",
        )
        generate_btn = st.button("🚀 Generate Story & Images", use_container_width=True, type="primary")

# ── Run pipeline ──────────────────────────────────────────────────────────────
if generate_btn and topic.strip():
    st.session_state.result = None   # reset

    # Progress display
    status_area = st.empty()

    with st.spinner(""):
        status_area.markdown('<span class="step-badge">🤖 Orchestrator deciding…</span>', unsafe_allow_html=True)

        initial_state = StoryState(
            topic=topic.strip(),
            story=None,
            image_prompts=None,
            image_b64=None,
            messages=[],
            next_step=None,
            error=None,
        )

        # Stream progress via callbacks (simple approach: run sync)
        status_area.markdown('<span class="step-badge">✍️ Story Writer crafting your story…</span>', unsafe_allow_html=True)

        result = st.session_state.graph.invoke(initial_state)

        status_area.markdown('<span class="step-badge">🎨 Image Generator creating visuals…</span>', unsafe_allow_html=True)
        st.session_state.result = result
        status_area.empty()

elif generate_btn and not topic.strip():
    st.warning("Please enter a story topic first!")

# ── Display results ───────────────────────────────────────────────────────────
if st.session_state.result:
    result = st.session_state.result
    st.divider()

    # Story section
    st.subheader("📜 The Story")
    st.markdown(f'<div class="story-box">{result["story"]}</div>', unsafe_allow_html=True)

    st.divider()

    # Image prompts used
    with st.expander("🔍 Image prompts used by the agent"):
        for i, p in enumerate(result.get("image_prompts", []), 1):
            st.markdown(f'<span class="prompt-badge">Prompt {i}:</span> {p}', unsafe_allow_html=True)

    # Images section
    st.subheader("🖼️ Generated Illustrations")
    images = result.get("image_b64", [])

    if images:
        cols = st.columns(len(images))
        for col, img_data in zip(cols, images):
            with col:
                if img_data.startswith("ERROR:"):
                    st.error(f"Image generation failed:\n{img_data[6:]}")
                else:
                    st.image(img_data, use_container_width=True)
    else:
        st.info("No images were generated.")

    # Download story
    st.divider()
    st.download_button(
        label="⬇️ Download Story as .txt",
        data=result["story"],
        file_name=f"story_{topic[:30].replace(' ', '_')}.txt",
        mime="text/plain",
    )

# ── Sidebar: pipeline diagram ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗺️ Agent Pipeline")
    st.markdown("""
```
START
  │
  ▼
Orchestrator
  │
  ├──► Story Writer
  │         │
  │         └──► Orchestrator
  │
  ├──► Image Gen (FLUX.1-schnell)
  │         │
  │         └──► Orchestrator
  │
  └──► END
```
    """)
    st.markdown("---")
    st.markdown("**Models used:**")
    st.markdown("- 🟣 `llama-3.3-70b` via Groq (story)")
    st.markdown("- 🔵 `FLUX.1-schnell` (images)")
    st.markdown("---")
    st.markdown("**Set in `.env`:**")
    st.code("GROQ_API_KEY=...\nHF_TOKEN=...", language="bash")