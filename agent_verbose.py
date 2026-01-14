# LangGraph Reddit Interview Intelligence Agent (Iterative, Planning, Refinement, Patterns + Named Problems)
# -----------------------------------------------------------------------------------------
# Goals
# - No Reddit API keys required: uses Reddit's public JSON endpoints with strict rate limiting
# - Agentic workflow (LangGraph): Plan -> Gather -> Filter -> Evaluate -> Refine -> (loop) -> Cluster -> Report
# - Outputs BOTH:
#   (A) algorithmic patterns (e.g., sliding window, BFS/DFS, DP)
#   (B) named problems (e.g., Two Sum, LRU Cache) when present
#
# Notes
# - This is intended for small-scale research/prototyping. Be respectful: low volume + sleep between requests.
# - In production, replace JSON endpoints with authenticated APIs + caching + robust compliance controls.

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict, Tuple

import requests
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI


# -----------------------------
# Configuration
# -----------------------------

DOMAIN = os.environ.get("AGENT_DOMAIN", "coding_interviews")
# You can try: coding_interviews | stocks | health

SEED_QUERIES = {
    "coding_interviews": [
        "coding interview",
        "leetcode",
        "OA",
        "system design",
        "two sum",
        "sliding window",
        "DP",
    ],
    "stocks": [
        "stocks",
        "earnings",
        "options",
        "SPY",
        "NVDA",
        "AAPL",
        "short squeeze",
    ],
    "health": [
        "longevity",
        "sleep",
        "metformin",
        "zone 2",
        "knee pain",
        "supplements",
        "blood work",
    ],
}

SEED_SUBREDDITS = {
    "coding_interviews": ["leetcode", "codinginterview", "cscareerquestions"],
    "stocks": ["stocks", "wallstreetbets", "investing"],
    "health": ["fitness", "longevity", "biohackers"],
}

# Data collection knobs
MAX_SUBREDDITS = 12
POST_LIMIT_PER_SUB = 120
TIME_FILTER = "year"  # top.json supports: hour, day, week, month, year, all
REQUEST_SLEEP_SEC = 1.0

# Filtering knobs
MIN_SCORE = 8
MIN_TITLE_WORDS = 4
MIN_QUALITY_POSTS = 50
MAX_ITERATIONS = 3

# LLM knobs
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = 0
MAX_TITLES_FOR_CLUSTERING = 140


# -----------------------------
# LLM
# -----------------------------

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=LLM_TEMPERATURE)


# -----------------------------
# State
# -----------------------------

class AgentState(TypedDict, total=False):
    domain: str
    plan: str
    subreddits: List[str]
    queries: List[str]
    iteration: int

    # data
    posts: List[Dict]
    filtered_posts: List[Dict]

    # outputs
    report_json: Dict


# -----------------------------
# Utilities
# -----------------------------

HEADERS = {
    "User-Agent": "agentic-reddit/0.2 (educational prototype; low-volume; contact: none)"
}


def log(msg: str) -> None:
    """Lightweight progress logger with timestamps (prints immediately)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _safe_json_loads(s: str) -> Optional[object]:
    """Attempt to parse JSON even if wrapped in ```json fences."""
    if not s:
        return None
    # Strip code fences
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"```\s*$", "", s)
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        return None


def _dedupe_posts(posts: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for p in posts:
        key = p.get("permalink") or (p.get("subreddit"), p.get("title"), p.get("created_utc"))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


# -----------------------------
# Keyword heuristics (domain-specific)
# -----------------------------

CODING_INCLUDE = [
    "leetcode", "lc", "oa", "online assessment", "coding question", "interview question",
    "two sum", "lru", "merge intervals", "top k", "kth", "sliding window",
    "two pointers", "binary search", "heap", "priority queue", "bfs", "dfs", "graph",
    "dynamic programming", "dp", "interval", "trie", "union find", "disjoint set",
    "backtracking", "recursion", "bitmask", "prefix sum",
]

CODING_EXCLUDE = [
    "salary", "tc", "comp", "compensation", "offer", "recruiter", "layoff", "laid off",
    "job market", "rejected", "rejection", "resume", "promotion", "p1", "p2", "p3",
]

# Simple named-problem hints (you will also get named problems from the LLM)
NAMED_PROBLEM_HINTS = [
    "two sum", "lru cache", "merge intervals", "valid parentheses", "group anagrams",
    "longest substring", "minimum window", "top k", "kth largest", "word ladder",
]


def is_quality_post(title: str, score: int, domain: str) -> bool:
    t = (title or "").strip().lower()

    if score < MIN_SCORE:
        return False
    if len(t.split()) < MIN_TITLE_WORDS:
        return False

    if domain == "coding_interviews":
        # Include coding signals and exclude obvious career talk
        if any(x in t for x in CODING_EXCLUDE):
            return False
        return any(x in t for x in CODING_INCLUDE) or any(ch in t for ch in ["[oa]", "[interview]", "(oa)"])

    # For stocks/health, keep a simpler heuristic; the LLM clustering will do the heavy lifting.
    # You can refine these similarly later.
    return True


# -----------------------------
# Nodes
# -----------------------------


def plan_strategy(state: AgentState) -> AgentState:
    log("[plan] Generating analysis plan via LLM...")
    domain = state.get("domain") or DOMAIN
    queries = state.get("queries") or SEED_QUERIES.get(domain, SEED_QUERIES["coding_interviews"])[:]
    subs = state.get("subreddits") or SEED_SUBREDDITS.get(domain, SEED_SUBREDDITS["coding_interviews"])[:]

    prompt = f"""
You are a research agent.
Create a concise plan to analyze Reddit discussions for domain = '{domain}'.
The plan must include:
1) how to discover additional relevant subreddits
2) how to gather posts responsibly (rate limits)
3) how to filter to high-quality domain-relevant posts
4) how to decide if more data is needed, and loop
5) how to cluster into BOTH (a) recurring themes/patterns and (b) named items when present

Return a numbered list. Keep it under 12 lines.
"""

    response = llm.invoke(prompt)
    log("[plan] Plan generated.")

    state["domain"] = domain
    state["queries"] = queries
    state["subreddits"] = subs
    state["iteration"] = 0
    state["plan"] = response.content

    # reset data containers
    state["posts"] = []
    state["filtered_posts"] = []
    return state



def discover_subreddits(state: AgentState) -> AgentState:
    log(f"[discover] Searching for related subreddits (domain={state.get('domain')}, seed_queries={len(state.get('queries', []))})...")
    """
    Discover additional subreddits using Reddit's public subreddit search endpoint.
    This replaces "search web" while still being keyless.

    Endpoint: https://www.reddit.com/subreddits/search.json?q=...&limit=...
    """
    domain = state["domain"]
    queries = state["queries"]

    found: List[str] = []
    for q in queries[:5]:
        log(f"[discover] Querying subreddit search: q='{q}'")
        url = "https://www.reddit.com/subreddits/search.json"
        params = {"q": q, "limit": 10}
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=10)
            if r.status_code != 200:
                time.sleep(REQUEST_SLEEP_SEC)
                continue
            data = r.json()
            children = data.get('data', {}).get('children', [])
            log(f"[discover]   -> status=200, results={len(children)}")
            for child in children:
                d = child.get("data", {})
                name = d.get("display_name")
                if name:
                    found.append(name)
        except Exception:
            pass

        time.sleep(REQUEST_SLEEP_SEC)

    # Merge + cap
    merged = list(dict.fromkeys(state.get("subreddits", []) + found))

    # Optional: light domain guard for coding (avoid huge unrelated subs)
    if domain == "coding_interviews":
        block = {"pics", "funny", "askreddit", "todayilearned", "news", "worldnews"}
        merged = [s for s in merged if s.lower() not in block]

    state["subreddits"] = merged[:MAX_SUBREDDITS]
    log(f"[discover] Using {len(state['subreddits'])} subreddits: {state['subreddits']}")
    return state



def fetch_reddit_posts(state: AgentState) -> AgentState:
    log(f"[fetch] Fetching top posts (time_filter={TIME_FILTER}, limit_per_sub={POST_LIMIT_PER_SUB})...")
    """
    Fetch posts using public subreddit listing endpoints (no auth).

    We use /r/{sub}/top.json?t=... for a recent time window.
    """
    posts = state.get("posts", [])

    for sub in state["subreddits"]:
        log(f"[fetch] /r/{sub}: requesting /top.json")
        url = f"https://www.reddit.com/r/{sub}/top.json"
        params = {"t": TIME_FILTER, "limit": POST_LIMIT_PER_SUB}
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
            if resp.status_code != 200:
                log(f"[fetch] /r/{sub}: non-200 status={resp.status_code}; skipping")
                time.sleep(REQUEST_SLEEP_SEC)
                continue
            data = resp.json()
            children = data.get('data', {}).get('children', [])
            log(f"[fetch] /r/{sub}: received {len(children)} posts")
        except Exception:
            time.sleep(REQUEST_SLEEP_SEC)
            continue

        for child in children:
            d = child.get("data", {})
            posts.append(
                {
                    "subreddit": sub,
                    "title": d.get("title", ""),
                    "score": int(d.get("score", 0) or 0),
                    "num_comments": int(d.get("num_comments", 0) or 0),
                    "created_utc": d.get("created_utc"),
                    "permalink": d.get("permalink"),
                    "url": ("https://www.reddit.com" + d.get("permalink", "")) if d.get("permalink") else None,
                }
            )

        time.sleep(REQUEST_SLEEP_SEC)

    state["posts"] = _dedupe_posts(posts)
    log(f"[fetch] Total unique posts collected so far: {len(state['posts'])}")
    return state



def filter_posts(state: AgentState) -> AgentState:
    log(f"[filter] Filtering posts for quality (min_score={MIN_SCORE}, min_title_words={MIN_TITLE_WORDS})...")
    domain = state["domain"]
    filtered = []

    for p in state.get("posts", []):
        title = p.get("title", "")
        score = int(p.get("score", 0) or 0)
        if is_quality_post(title, score, domain):
            filtered.append(p)

    log(f"[filter] Kept {len(filtered)} of {len(state.get('posts', []))} posts after heuristics.")

    # Sort for quality (score then comments)
    filtered.sort(key=lambda x: (x.get("score", 0), x.get("num_comments", 0)), reverse=True)

    state["filtered_posts"] = filtered
    return state



def evaluate_quality(state: AgentState) -> str:
    """
    Decide whether to refine (find more subs / gather more) or proceed to clustering.
    """
    iteration = int(state.get("iteration", 0) or 0)
    n = len(state.get("filtered_posts", []))

    if n >= MIN_QUALITY_POSTS:
        log(f"[gate] Enough quality posts ({n} >= {MIN_QUALITY_POSTS}). Proceeding to clustering.")
        return "cluster"

    if iteration >= MAX_ITERATIONS:
        log(f"[gate] Reached max iterations ({iteration} >= {MAX_ITERATIONS}) with {n} quality posts. Proceeding to clustering.")
        return "cluster"

    log(f"[gate] Not enough quality posts ({n} < {MIN_QUALITY_POSTS}). Refining sources (iteration {iteration + 1}/{MAX_ITERATIONS}).")
    return "refine"

def refine_sources(state: AgentState) -> AgentState:
    log(f"[refine] Refinement step starting (current_iteration={state.get('iteration', 0)})...")
    """
    When signal is insufficient:
    - ask LLM for additional query terms (optional)
    - discover more subreddits
    - increment iteration

    Note: We keep this bounded by MAX_ITERATIONS.
    """
    domain = state["domain"]

    # Ask for a few extra query terms to improve discovery.
    # Keep it deterministic and short.
    prompt = f"""
We are collecting Reddit discussions for domain='{domain}'.
Current queries: {state.get('queries', [])}
Suggest 3 additional short search queries to find relevant subreddits.
Return JSON array of strings only.
"""
    log("[refine] Asking LLM for additional subreddit search queries...")
    # log(f"[cluster] Sending {len(titles)} titles to LLM for clustering...")
    candidates = state.get("candidates") or state.get("posts") or []
    log(f"[refine] Running LLM refine gate on {len(candidates)} candidates...")
    resp = llm.invoke(prompt)
    log("[cluster] LLM clustering complete.")
    extra = _safe_json_loads(resp.content)
    if isinstance(extra, list):
        log(f"[refine] LLM suggested {len(extra)} additional queries.")
        # Merge and cap
        merged_q = list(dict.fromkeys(state.get("queries", []) + [str(x) for x in extra]))
        state["queries"] = merged_q[:10]

    # Discover subs from Reddit search endpoint
    state = discover_subreddits(state)

    state["iteration"] = int(state.get("iteration", 0) or 0) + 1
    return state



def cluster_topics(state: AgentState) -> AgentState:
    log("[cluster] Clustering titles via LLM (this may take a bit)...")
    """
    Cluster into BOTH:
    - algorithmic patterns/themes
    - named problems/items (when present)

    Output format is strict JSON to make it easy to render.
    """
    domain = state["domain"]
    titles = [p.get("title", "") for p in state.get("filtered_posts", [])][:MAX_TITLES_FOR_CLUSTERING]

    # Provide lightweight hints for named problems (non-authoritative)
    hints = ", ".join(NAMED_PROBLEM_HINTS) if domain == "coding_interviews" else ""

    prompt = f"""
You are clustering high-quality Reddit post titles for domain='{domain}'.

Requirements:
1) Produce BOTH:
   A) 'patterns' (6-10 clusters): recurring algorithmic patterns/themes (e.g., sliding window, graphs, DP)
   B) 'named_items' (0-20): explicit named problems/items when present in the titles (e.g., Two Sum, LRU Cache)
2) Each cluster/item must include:
   - 'name'
   - 'description' (1-2 sentences)
   - 'evidence_titles' (3-6 titles from the list)
   - 'keywords' (3-8)
3) Also include a 'top_mentions' list for named_items with an estimated 'mention_count' (best-effort).
4) Return STRICT JSON only. No markdown.

Hints for named items (optional): {hints}

Titles:
""" + "\n".join(f"- {t}" for t in titles)

    resp = llm.invoke(prompt)
    parsed = _safe_json_loads(resp.content)

    if not isinstance(parsed, dict):
        # Fallback minimal report
        parsed = {
            "patterns": [],
            "named_items": [],
            "top_mentions": [],
            "note": "LLM output was not valid JSON; consider retrying or tightening prompt.",
        }

    report = {
        "meta": {
            "domain": domain,
            "subreddits_used": state.get("subreddits", []),
            "iterations": state.get("iteration", 0),
            "total_posts": len(state.get("posts", [])),
            "quality_posts": len(state.get("filtered_posts", [])),
            "time_filter": TIME_FILTER,
        },
        "results": parsed,
    }

    state["report_json"] = report
    return state


# -----------------------------
# Graph
# -----------------------------

graph = StateGraph(AgentState)

graph.add_node("plan", plan_strategy)
# Step 1 (your request): "search" for forums/subreddits
graph.add_node("discover", discover_subreddits)
# Step 2: gather posts
graph.add_node("fetch", fetch_reddit_posts)
# Step 3: filter high quality
graph.add_node("filter", filter_posts)
# Step 4: if insufficient, refine and loop
graph.add_node("refine", refine_sources)
# Step 5: cluster
graph.add_node("cluster", cluster_topics)

# Entry
graph.set_entry_point("plan")

# Edges
graph.add_edge("plan", "discover")
graph.add_edge("discover", "fetch")
graph.add_edge("fetch", "filter")

graph.add_conditional_edges(
    "filter",
    evaluate_quality,
    {
        "refine": "refine",
        "cluster": "cluster",
    },
)

graph.add_edge("refine", "fetch")
graph.add_edge("cluster", END)

agent = graph.compile()


# -----------------------------
# Execution
# -----------------------------

if __name__ == "__main__":
    log(f"[main] Starting agent run (domain={DOMAIN})")
    final_state = agent.invoke({"domain": DOMAIN})
    log("[main] Agent run complete. Rendering report...")

    print("=== AGENT STRATEGY ===")
    print(final_state.get("plan", ""))

    print("\n=== SUBREDDITS USED ===")
    print(final_state.get("subreddits", []))

    report = final_state.get("report_json", {})

    print("\n=== REPORT SUMMARY ===")
    if report:
        meta = report.get("meta", {})
        print(json.dumps(meta, indent=2))

    print("\n=== CLUSTERED RESULTS (PATTERNS + NAMED ITEMS) ===")
    print(json.dumps(report.get("results", {}), indent=2))

    # Save to disk for easy sharing
    with open("report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nSaved report.json")
