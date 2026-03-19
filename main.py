from langgraph.graph import StateGraph, START, END
import dotenv

dotenv.load_dotenv()

from agent import StoryState, orchestrator, orch_router, story_writer, image_gen


def build_workflow():
    wf = StateGraph(StoryState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    wf.add_node("orch",         orchestrator)
    wf.add_node("story_writer", story_writer)
    wf.add_node("image_gen",    image_gen)

    # ── Edges ──────────────────────────────────────────────────────────────
    wf.add_edge(START, "orch")

    wf.add_conditional_edges(
        "orch",
        orch_router,
        {
            "story_writer": "story_writer",
            "image_gen":    "image_gen",
            "end":          END,
        },
    )

    # Both worker nodes feed back into the orchestrator
    wf.add_edge("story_writer", "orch")
    wf.add_edge("image_gen",    "orch")

    return wf.compile()


# ── Quick CLI test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    graph = build_workflow()

    topic = input("Enter a story topic: ").strip()
    state = StoryState(
        topic=topic,
        story=None,
        image_prompts=None,
        image_b64=None,
        messages=[],
        next_step=None,
        error=None,
    )

    final = graph.invoke(state)

    print("\n" + "=" * 60)
    print("STORY\n" + "=" * 60)
    print(final["story"])

    print("\n" + "=" * 60)
    print("IMAGE PROMPTS")
    print("=" * 60)
    for i, p in enumerate(final["image_prompts"], 1):
        print(f"  {i}. {p}")

    print("\n" + "=" * 60)
    print(f"Generated {len(final['image_b64'])} image(s).")
    print("Run app.py (Streamlit) to view them visually.")