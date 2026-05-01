"""
Backpacking Assistant — LangChain + LangGraph chatbot powered by Gemini.

Single-file app. Tools:
  1. gear_search          — CSV lookup with rich filters
  2. backpacking_knowledge — RAG over markdown guides
  3. pack_calculator      — total weight + cost, 20%-of-body-weight check
  4. trip_logger          — append a structured note to logs/trips.jsonl

QUICK START (Mac/Linux, in a fresh terminal):
    cd ~/Desktop/backpacking-bot
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    export GOOGLE_API_KEY=your-key-here
    python app.py
"""

import os
import json
import datetime
import pathlib
from typing import List, Optional

# Auto-load GOOGLE_API_KEY from a .env file if present, so users don't have
# to re-run `export` in every terminal session.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; export works fine too.

import gradio as gr
import pandas as pd

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document

from langgraph.prebuilt import create_react_agent

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================================================================
# PATHS
# =============================================================================

ROOT = pathlib.Path(__file__).parent
GEAR_CSV_PATH = ROOT / "data" / "gear.csv"
GUIDES_DIR = ROOT / "guides"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
TRIP_LOG = LOG_DIR / "trips.jsonl"
CONV_LOG = LOG_DIR / "conversations.jsonl"


# =============================================================================
# LOAD DATA
# =============================================================================

GEAR_DF = pd.read_csv(GEAR_CSV_PATH)


def _build_vectorstore() -> FAISS:
    """Embed the markdown guides and build an in-memory FAISS index."""
    docs = []
    for md_path in sorted(GUIDES_DIR.glob("*.md")):
        text = md_path.read_text(encoding="utf-8")
        docs.append(Document(page_content=text, metadata={"source": md_path.name}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    return FAISS.from_documents(chunks, embeddings)


VECTORSTORE = _build_vectorstore()


# =============================================================================
# TOOLS
# =============================================================================

@tool
def gear_search(
    category: Optional[str] = None,
    item_type: Optional[str] = None,
    company: Optional[str] = None,
    max_weight_oz: Optional[float] = None,
    max_price_usd: Optional[float] = None,
    essential_level: Optional[str] = None,
    season_rating: Optional[int] = None,
    waterproof_only: bool = False,
    keyword: Optional[str] = None,
) -> str:
    """
    Search the backpacking gear catalog (190 items, 19 categories).
    Use whenever the user wants gear recommendations, comparisons, or items
    matching a budget / weight / category constraint.

    Args:
        category: Top-level category. Options: 'Shelter', 'Sleeping', 'Backpack',
            'Cooking', 'Hydration', 'Clothing', 'Footwear', 'Electronics',
            'Safety', 'Navigation', 'Food', 'Storage', 'Hygiene', 'Accessories',
            'Winter Gear', 'Fishing', 'Pet Gear', 'Camera Gear', 'Luxury'.
        item_type: More specific (e.g., 'Tent', 'Sleeping Bag', 'Stove',
            'Water Filter', 'Headlamp', 'Rain Jacket').
        company: Filter by brand (e.g., 'Osprey', 'MSR', 'Patagonia').
        max_weight_oz: Filter to items at or below this weight in OUNCES.
        max_price_usd: Filter to items at or below this price in USD.
        essential_level: 'High' (must-have) or 'Medium' (nice-to-have).
        season_rating: 1 (winter), 3 (three-season), or 4 (four-season).
        waterproof_only: If True, only include items marked Yes/Waterproof.
        keyword: Free-text match against item_type, model, or notes.

    Returns:
        A formatted list of up to 10 matching items.
    """
    df = GEAR_DF.copy()

    if category:
        df = df[df["category"].str.contains(category, case=False, na=False)]
    if item_type:
        df = df[df["item_type"].str.contains(item_type, case=False, na=False)]
    if company:
        df = df[df["company"].str.contains(company, case=False, na=False)]
    if max_weight_oz is not None:
        df = df[df["weight_oz"] <= max_weight_oz]
    if max_price_usd is not None:
        df = df[df["price_usd"] <= max_price_usd]
    if essential_level:
        df = df[df["essential_level"].str.lower() == essential_level.lower()]
    if season_rating is not None:
        df = df[df["season_rating"] == season_rating]
    if waterproof_only:
        df = df[df["waterproof"].str.lower().isin(["yes", "waterproof"])]
    if keyword:
        mask = (
            df["item_type"].str.contains(keyword, case=False, na=False)
            | df["model"].str.contains(keyword, case=False, na=False)
            | df["notes"].str.contains(keyword, case=False, na=False)
        )
        df = df[mask]

    if df.empty:
        return (
            "No items match those filters. Try loosening category, weight, "
            "price, or other constraints."
        )

    df = df.sort_values(["category", "price_usd"]).head(10)
    lines = [
        f"- {r.company} {r.model} ({r.item_type}, {r.category}): "
        f"{r.weight_oz} oz, ${r.price_usd:.2f}, "
        f"{r.season_rating}-season, {r.essential_level} priority"
        for r in df.itertuples()
    ]
    return "\n".join(lines)


@tool
def backpacking_knowledge(question: str) -> str:
    """
    Answer general backpacking how-to and best-practice questions using the
    knowledge base of guides (Leave No Trace, layering, food planning, water
    treatment). Use when the user asks 'how do I...', 'what's the rule
    for...', or any conceptual / educational question that isn't about
    specific products.

    Args:
        question: The user's natural-language question.

    Returns:
        Relevant excerpts from the guides, with the source file noted.
    """
    results = VECTORSTORE.similarity_search(question, k=3)
    if not results:
        return "No relevant guidance found in the knowledge base."
    out = []
    for r in results:
        src = r.metadata.get("source", "guide")
        out.append(f"[{src}]\n{r.page_content.strip()}")
    return "\n\n---\n\n".join(out)


@tool
def pack_calculator(items: List[str], body_weight_lb: Optional[float] = None) -> str:
    """
    Calculate total weight and total cost for a list of gear items, and flag
    if the pack exceeds the recommended 20% of the user's body weight.

    Args:
        items: A list of item names (model names work best, e.g.,
            "Copper Spur HV UL2", "PocketRocket Deluxe"). Case-insensitive
            substring match against company + model + item type.
        body_weight_lb: Optional. If provided, compares pack weight to the
            20%-of-body-weight rule of thumb.

    Returns:
        A breakdown with totals and a safety note if applicable.
    """
    if not items:
        return "Please provide at least one item to calculate."

    matched_rows = []
    not_found = []
    for name in items:
        combined = (
            GEAR_DF["company"].fillna("") + " "
            + GEAR_DF["model"].fillna("") + " "
            + GEAR_DF["item_type"].fillna("")
        )
        hits = GEAR_DF[combined.str.contains(name, case=False, na=False)]
        if hits.empty:
            not_found.append(name)
        else:
            matched_rows.append(hits.iloc[0])

    if not matched_rows:
        return (
            "None of those items were found in the catalog. "
            "Try gear_search first to get exact names."
        )

    total_oz = sum(r["weight_oz"] for r in matched_rows)
    total_cost = sum(r["price_usd"] for r in matched_rows)
    total_lb = total_oz / 16.0

    lines = [
        f"- {r['company']} {r['model']}: {r['weight_oz']} oz, ${r['price_usd']:.2f}"
        for r in matched_rows
    ]
    summary = (
        f"\n**Pack totals**\n"
        f"- Items: {len(matched_rows)}\n"
        f"- Weight: {total_oz:.1f} oz ({total_lb:.2f} lb)\n"
        f"- Cost: ${total_cost:.2f}"
    )

    safety_note = ""
    if body_weight_lb is not None:
        limit_lb = body_weight_lb * 0.20
        if total_lb > limit_lb:
            safety_note = (
                f"\n\nWARNING: Pack is {total_lb:.2f} lb -- over the "
                f"20%-of-body-weight guideline ({limit_lb:.1f} lb for "
                f"{body_weight_lb} lb). Consider lighter alternatives."
            )
        else:
            safety_note = (
                f"\n\nOK: Pack is {total_lb:.2f} lb -- within the "
                f"20%-of-body-weight guideline ({limit_lb:.1f} lb for "
                f"{body_weight_lb} lb)."
            )

    missing_note = ""
    if not_found:
        missing_note = f"\n\nNot found in catalog: {', '.join(not_found)}"

    return "\n".join(lines) + summary + safety_note + missing_note


@tool
def trip_logger(summary: str, trip_name: Optional[str] = None) -> str:
    """
    Save a short note about the user's planned trip to a log file. Use this
    when the user explicitly asks to save, log, or remember something about
    a trip.

    Args:
        summary: Free-text summary of what to log.
        trip_name: Optional short name like 'Wonderland Trail Aug 2026'.

    Returns:
        Confirmation with timestamp.
    """
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "trip_name": trip_name or "untitled",
        "summary": summary,
    }
    with TRIP_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return f"Logged at {entry['timestamp']} under '{entry['trip_name']}'."


TOOLS = [gear_search, backpacking_knowledge, pack_calculator, trip_logger]


# =============================================================================
# AGENT — using LangGraph's prebuilt ReAct agent (works on any LangChain version)
# =============================================================================

SYSTEM_PROMPT = """You are a friendly, knowledgeable backpacking assistant.

You help users plan trips, choose gear within a budget, calculate pack weight,
and answer wilderness-skill questions.

Tool routing rules:
- For specific gear recommendations, comparisons, or filtering by budget,
  weight, brand, or essential-level -> use gear_search.
- For how-to, best-practice, and rule questions (LNT, layering, food, water)
  -> use backpacking_knowledge.
- For totaling weight or cost across multiple items, or checking pack weight
  vs. body weight -> use pack_calculator.
- When the user asks to save, log, or remember a trip -> use trip_logger.

If a question doesn't need a tool (casual chat, clarification), reply directly.
Be concise. When you give recommendations, briefly explain why.
The catalog stores weight in OUNCES. If the user gives weight in grams or
kilograms, convert to ounces (1 oz = 28.35 g) before calling tools.
"""

LLM = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
AGENT = create_react_agent(LLM, TOOLS, prompt=SYSTEM_PROMPT)


# =============================================================================
# CONVERSATION LOGGING
# =============================================================================

def log_turn(user_msg: str, bot_msg: str) -> None:
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "user": user_msg,
        "bot": bot_msg,
    }
    with CONV_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def respond(message: str, history: list) -> str:
    """Handles both Gradio history formats:
    - Older (tuples): list of [user_msg, bot_msg] pairs
    - Newer (messages): list of {'role', 'content'} dicts
    """
    msgs = []
    for turn in history:
        # New-style: dict with role + content
        if isinstance(turn, dict):
            role = turn.get("role")
            content = turn.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                msgs.append(AIMessage(content=content))
        # Old-style: [user_msg, bot_msg] pair
        elif isinstance(turn, (list, tuple)) and len(turn) >= 2:
            user_msg, bot_msg = turn[0], turn[1]
            if user_msg:
                msgs.append(HumanMessage(content=user_msg))
            if bot_msg:
                msgs.append(AIMessage(content=bot_msg))
    msgs.append(HumanMessage(content=message))

    try:
        result = AGENT.invoke({"messages": msgs})
        # LangGraph returns the full message list; the last message is the reply.
        final = result["messages"][-1]
        reply = final.content if hasattr(final, "content") else str(final)
        if isinstance(reply, list):
            reply = "\n".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in reply
            ).strip()
    except Exception as e:
        reply = f"Something went wrong: {e}"

    log_turn(message, reply)
    return reply


EXAMPLES = [
    "I'm a beginner. Recommend a backpack under $300.",
    "What's the rule for catholes when there's no toilet?",
    "Calculate pack weight for: Exos 58, Copper Spur HV UL2, NeoAir XLite NXT, Revelation 20, PocketRocket Deluxe. I weigh 160 lb.",
    "Save a note: planning a 4-day Wonderland Trail trip in August.",
    "How many calories should I plan per day?",
    "Show me ultralight backpacks under 35 oz.",
    "What Patagonia clothing is in the catalog?",
]


with gr.Blocks(title="Backpacking Assistant") as demo:
    gr.Markdown(
        "# Backpacking Assistant\n"
        "Plan trips, pick gear, calculate pack weight, and learn backcountry "
        "skills. Powered by Gemini + LangChain over a 190-item gear catalog "
        "and a small guide library."
    )
    gr.ChatInterface(
        fn=respond,
        examples=EXAMPLES,
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch()
