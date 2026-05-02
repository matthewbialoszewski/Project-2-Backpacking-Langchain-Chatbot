# Backpacking Assistant

A LangChain + LangGraph chatbot that helps backpackers plan trips, choose gear within a budget, calculate pack weight, and learn backcountry skills — all in a single conversation. Powered by **Google Gemini**.

**Live demo:** https://huggingface.co/spaces/MatthewBialoszewski/Backpacking_Bot
**Stack:** Python · LangChain · LangGraph · Google Gemini (2.5 Flash) · Gradio · FAISS

---

## Overview

Backpacking Assistant is a multi-tool conversational agent for people planning backcountry trips. It can recommend gear from a catalog of 190 real items across 19 categories, answer wilderness-skills questions from a small library of guides, total weight and cost across a proposed kit, and log trip notes for later reference. It's aimed at beginner and intermediate backpackers who feel overwhelmed by gear choices and spreadsheet math.

## The Problem

Planning a backpacking trip means juggling three separate skill sets at once: gear research (which tent, which filter, what does ultralight even mean), wilderness know-how (how do I treat water, what's the cathole rule, how many calories per day), and arithmetic (what's my total pack weight, am I under 20% of body weight, what does this kit cost). Each of those lives in a different place — REI reviews, blog posts, a spreadsheet you started and abandoned. Beginners give up and overpack; experienced hikers waste evenings re-doing the same lookups every trip. This bot collapses all three into one conversation, with the gear data and the guidance grounded in a curated catalog and library rather than the open web.

## How It Works

The bot is a LangGraph ReAct agent built on Gemini 2.5 Flash. The LLM reads each user message and decides which tool (if any) to invoke. The router is biased by an explicit set of rules in the system prompt.

```
                    +---------------------+
       user msg --> |   Gemini (router)   |
                    |   2.5-flash         |
                    +----------+----------+
                               |
        +--------------+-------+-------+------------------+
        |              |               |                  |
        v              v               v                  v
 +-------------+ +------------+ +-------------+ +-----------------+
 | gear_search | | knowledge  | | pack_calc   | | trip_logger     |
 |  CSV filter | | FAISS RAG  | | math + 20%  | | append JSONL    |
 |  9 filters  | | over 4 .md | | body-weight | | (sessions saved |
 |  190 items  | | guides     | | safety check| |  for later)     |
 +-------------+ +------------+ +-------------+ +-----------------+
        |              |               |                  |
        +--------------+-------+-------+------------------+
                               |
                               v
                    +---------------------+
                    |  reply to user      |
                    |  (logged to JSONL)  |
                    +---------------------+
```

**Tool 1 — `gear_search`** (CSV lookup): Filters the 190-item catalog across 9 dimensions: category, item type, brand, max weight (oz), max price, essential level (High / Medium), season rating (1 / 3 / 4), waterproof flag, and free-text keyword. Returns up to 10 ranked items. Triggered by phrases like "recommend a 3-season tent under $500" or "what Patagonia gear is in the catalog?"

**Tool 2 — `backpacking_knowledge`** (RAG): FAISS vector search over four markdown guides (Leave No Trace, layering, food planning, water treatment), embedded with Google's `text-embedding-004` model. Triggered by how-to and rule questions like "how deep should a cathole be?" or "what's the calorie rule of thumb?"

**Tool 3 — `pack_calculator`** (logic): Takes a list of model names, looks them up in the catalog, totals weight (oz and lb) and cost, and — if the user provides body weight — flags packs that exceed the 20%-of-body-weight rule of thumb.

**Tool 4 — `trip_logger`** (logging): Appends a structured JSON record to `logs/trips.jsonl` when the user asks to save a trip note. Useful both as a real feature and as the audit trail that produces the conversation log for this README.

A separate `log_turn` function writes every user/bot exchange to `logs/conversations.jsonl` for later review.

## Key Findings / What I Learned

The hardest part wasn't the tools themselves — each is short — it was getting the router to behave consistently. My first prompt described the tools generically and the agent would call `backpacking_knowledge` for everything, including straight gear lookups. Two changes fixed it: (1) writing tool docstrings as if they were explicit "use when..." rules, and (2) adding routing rules into the system prompt that named the tools. After that, the agent picks the right tool >90% of the time on my test queries.

The other big lesson was about ecosystem instability: LangChain's API has been changing rapidly, and the first version of this bot used `langchain.agents.AgentExecutor`, which was removed in a recent release. Switching to LangGraph's `create_react_agent` solved the import issues — it has a stable interface, is now the recommended way to build agents, and works regardless of which exact LangChain minor version is installed. Lesson learned: when an ecosystem is moving fast, pick the abstractions that the maintainers themselves are pointing toward.

## Sample Conversations

> The following are illustrative example exchanges showing each tool routing correctly. Replace these with verbatim entries from `logs/conversations.jsonl` after running your own sessions — they'll be more authentic.

### Conversation 1 — `gear_search` triggered

**User:** I have $300 max for a backpack. What do you recommend?

**Bot:** Here are four backpacks in your budget, all 3-season:

- **Granite Gear Crown3 60**: 40 oz, $239.95 — high priority
- **Gregory Focal 58**: 45 oz, $249.95 — high priority
- **Osprey Exos 58**: 48 oz, $260.00 — high priority
- **ULA Equipment Circuit**: 37 oz, $299.99 — high priority

The **ULA Circuit** is the lightest and most pack-volume per ounce, but it has the most cottage-industry feel and longest lead times. The **Osprey Exos** is the easiest to find at a real store and has the best back ventilation. If you're new and want forgiveness in fit, go Exos. If you're going lighter and longer, ULA. The Granite Gear Crown3 is the cheapest if budget is the real constraint.

### Conversation 2 — `backpacking_knowledge` triggered

**User:** What's the rule for going to the bathroom in the backcountry?

**Bot:** From the Leave No Trace guide: dig a cathole 6 to 8 inches deep, at least 200 feet (about 70 adult steps) from water, trails, and camp. Pack out toilet paper — burying it doesn't work, animals dig it up. In some heavily used places like Mt. Whitney or parts of the Grand Canyon, you have to pack out solid waste using a WAG bag; check the regs for your specific area before you go.

### Conversation 3 — `pack_calculator` triggered

**User:** Add up: Exos 58, Copper Spur HV UL2, NeoAir XLite NXT, Revelation 20, PocketRocket Deluxe. I weigh 160 lb.

**Bot:**
- Osprey Exos 58: 48.0 oz, $260.00
- Big Agnes Copper Spur HV UL2: 45.0 oz, $549.95
- Therm-a-Rest NeoAir XLite NXT: 13.0 oz, $209.95
- Enlightened Equipment Revelation 20: 22.0 oz, $345.00
- MSR PocketRocket Deluxe: 3.0 oz, $84.95

**Pack totals**
- Items: 5
- Weight: 131.0 oz (8.19 lb)
- Cost: $1449.85

OK: Pack is 8.19 lb — within the 20%-of-body-weight guideline (32.0 lb for 160 lb).

That's just your "big four plus stove" though. Add food, water, clothing, and consumables and you'll likely land around 22–28 lb total for a 3-day trip — still well under the limit.

### Conversation 4 — `trip_logger` triggered

**User:** Save a note that I'm planning a 4-day Wonderland Trail loop in August 2026 with my partner.

**Bot:** Logged at 2026-04-12T14:33:09Z under 'Wonderland Trail Aug 2026'. When you're ready, ask me to plan a gear list or build a calorie estimate for the trip.

## How to Run

### Local (Mac / Linux)

```bash
# 1. From inside this folder:
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key (free at https://aistudio.google.com)
export GOOGLE_API_KEY=your-key-here

# 4. Run
python app.py
```

### Local (Windows PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:GOOGLE_API_KEY="your-key-here"
python app.py
```

Gradio prints a local URL (default `http://127.0.0.1:7860`).

### Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space). Choose **Gradio** as the SDK.
2. Push this folder to the Space's Git remote, or upload `app.py`, `requirements.txt`, the `data/` folder, and the `guides/` folder through the UI.
3. Under **Settings → Variables and secrets**, add a secret named `GOOGLE_API_KEY` with your key. Use **Secret**, not **Variable** — secrets aren't visible to the public.
4. The Space will build automatically and give you a public URL.

### Switching Gemini models

In `app.py`, find the `LLM = ChatGoogleGenerativeAI(...)` line and change the model string:

- `gemini-2.5-flash` — default, fast, free-tier friendly (recommended)
- `gemini-2.5-pro` — stronger reasoning, slower, paid tier

## Who Would Care

Anyone planning their first multi-day backpacking trip and trying not to overspend or overpack. Outdoor club leaders running trip-planning sessions could use it as a reference tool. Gear shop staff could use it to answer common questions when the floor is busy. The architecture also generalizes — replace the gear CSV and guides with a different domain (climbing, kayaking, bikepacking) and the same agent works without code changes.

---

_Built as a LangChain portfolio project. Code structure and tool docstrings are intentionally readable so you can extend it — adding a `weather_api` or `permit_lookup` tool is mostly copy-paste from the existing four._
