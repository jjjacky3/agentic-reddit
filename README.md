# Reddit Coding Interview Research Agent



This project is an example of an **AI agent** (not just an AI-powered script) that autonomously researches Reddit discussions related to software engineering coding interviews and extracts the most frequent **problem patterns and named problems** to help with interview preparation.



The system combines deterministic data collection with LLM-based reasoning, using a **stateful, multi-step control loop** implemented via LangGraph.



---



## What the Agent Does (End-to-End)



### 1. Domain Definition and Seed Knowledge

The agent starts with a clearly defined **research domain** (default: `coding\_interviews`) and a small set of:



- Seed search queries (e.g., LeetCode, OA, DP, sliding window)

- Seed subreddits (e.g., `leetcode`, `codinginterview`, `cscareerquestions`)



These act as the agent’s initial prior. All subsequent decisions are guided by and stored in agent state.



---



### 2. Stateful Agent Workflow (LangGraph)

The system is built using **LangGraph**, not a linear script.



- A shared `AgentState` object carries the plan, queries, subreddits, iteration count, raw posts, filtered posts, and final report.

- Each node reads from and writes to this state.

- Execution can branch or loop based on intermediate results.



This makes the workflow **stateful, inspectable, and adaptive**.



---



### 3. Planning Step (LLM-Guided)

The agent first asks the LLM to generate a concise, explicit plan covering:



1\. Source discovery

2\. Data collection

3\. Quality filtering

4\. Evaluation and iteration

5\. Final clustering and synthesis



The plan is stored in state and later printed, making the reasoning process visible and auditable.



---



### 4. Subreddit Discovery

Using Reddit’s public search API, the agent:



- Searches for subreddits relevant to its current query set

- Merges discovered subreddits with seeds

- Applies domain-specific exclusions to remove irrelevant large communities

- Caps total subreddits to control cost and runtime



This step allows **adaptive source discovery** rather than relying on a fixed list.



---



### 5. Evidence Collection (Reddit Post Fetching)

For each selected subreddit, the agent:



- Fetches top posts over a specified time window

- Extracts structured metadata (title, score, comments, timestamps, URLs)

- Deduplicates posts across subreddits

- Respects rate limits via intentional sleeps



At this stage, the agent is gathering raw evidence, not yet reasoning about it.



---



### 6. Domain-Specific Filtering

Posts are filtered using heuristics tailored to coding interview content:



- Minimum score and title-length thresholds

- Inclusion keywords (e.g., “DP”, “graph”, “OA”, “LeetCode”)

- Exclusion keywords (e.g., compensation or career discussion)



The result is a smaller, higher-signal dataset focused on interview problems rather than general career chatter.



---



### 7. Self-Evaluation and Control Flow

The agent evaluates whether it has gathered **enough quality evidence**:



- If the number of filtered posts meets a threshold, it proceeds

- Otherwise, it enters a refinement loop

- A maximum iteration limit ensures bounded execution



This decision point is a key agent characteristic: the workflow is not strictly linear.



---



### 8. Strategy Refinement (LLM-Guided)

When evidence is insufficient, the agent asks the LLM to propose **additional search queries**.



- New queries are merged into state

- Subreddit discovery and fetching run again

- Iteration count is incremented



The agent is effectively **improving its own research strategy within a single run**, under explicit safety limits.



---



### 9. Clustering and Synthesis

Once enough signal is collected, the agent:



- Sends post titles to the LLM

- Requests strict JSON output only

- Produces:

&nbsp; - Problem *patterns* (e.g., sliding window, graphs, DP)

&nbsp; - *Named problems* when explicitly mentioned

&nbsp; - Estimated mention counts

&nbsp; - Evidence titles per cluster



The output is grounded in observed data, not free-form hallucination.



---



### 10. Final Report Artifact

The agent prints and saves a structured report (`report.json`) containing:



- The original plan

- Sources used

- Identified patterns

- Named problems

- Supporting evidence



This makes the run reproducible and suitable for downstream analysis.



---



## Why This Is an AI Agent (Not Just a Script)



This project goes beyond a typical “LLM-powered tool” in several key ways:



- **Stateful execution** across multiple steps

- **Explicit planning** before acting

- **Autonomous source discovery**

- **Feedback-driven iteration**

- **Bounded self-improvement**

- **Evidence-grounded synthesis**

- **Artifact-based output**



The LLM is used for reasoning and strategy, while deterministic code handles data collection and filtering. This separation enables transparency, control, and auditability.



---



## Summary



This agent demonstrates a full **agent lifecycle**:



1. Define a goal

2. Plan an approach

3. Discover sources

4. Gather evidence

5. Evaluate quality

6. Adapt strategy

7. Synthesize structured insights



It is designed as a practical, inspectable example of how to build AI agents that **reason, act, and adapt**, rather than simply responding to prompts.



