# TokenLess - Sample Input/Output

This document records a complete end-to-end session with the TokenLess Mentor, showing the full pipeline from a vague user prompt to a structured optimized output. Metrics are representative of the live evaluation run used for the final project.

## Session: React Stock Dashboard (EB-01)

### Step 1 - User Initial Message

```
I want to build a React dashboard that shows real-time stock prices, includes charts, and sends an alert when a stock moves more than 5%.
```

Raw token estimate: 47 tokens.

### Step 2 - Mentor Clarification

TokenLess asks focused questions across five turns:

1. Which React version and should it use TypeScript?
2. Which charting library should it use?
3. Where does the stock data come from?
4. Does the app need authentication?
5. Should tests be included?

Clarified requirements: React 18, TypeScript, Recharts, WebSocket API with endpoint TBD, no authentication, and tests optional for the first version.

### Step 3 - Final Optimized Prompt

~~~markdown
## Role
You are a senior React/TypeScript developer. Your expertise: real-time data dashboards, financial UI components.

## Task
Build a real-time stock price dashboard with Recharts visualization and plus-or-minus 5% price-movement alerts.

## Context
- Framework: React 18 + TypeScript
- Charting: Recharts
- Data source: WebSocket connection (endpoint TBD; use a placeholder URL)
- Alert trigger: any stock moves more than plus-or-minus 5% from its opening price

## Constraints
- No user authentication
- Functional components and React hooks only
- Handle WebSocket reconnect gracefully on disconnect

## Output Format

```json
{
  "files_to_create_or_modify": [],
  "implementation_steps": [],
  "acceptance_criteria": [],
  "tests": []
}
```

## Reminder
Implement the WebSocket data feed and the alert logic first; the Recharts component is secondary.
~~~

### Step 4 - Metrics

| Metric | Value |
|---|---|
| Raw prompt tokens | 47 |
| Optimized prompt tokens | 187 |
| Pipeline token cost | about 2,591 tokens |
| GPT-4o judge verdict | tie |
| Gemini judge verdict | optimized |
| Final winner | optimized |
| Quality score overall | 9.55 / 10 |
| Intent alignment | 9.75 / 10 |
| Logic coherence | 9.50 / 10 |
| Conciseness score | 9.00 / 10 |
| Format compliance | 10.00 / 10 |

## Where the System Fails: EB-02

Raw prompt: `Write a function`

The pipeline can run the full professional rewrite, but the prompt is too underspecified. Even after two self-correction retries, the quality score reached only 2.20 / 10. The quality gate blocked the output and returned the original prompt unchanged.

Takeaway: the mentor clarification phase is essential. A three-token prompt carries too little recoverable intent, and structural optimization cannot replace missing user requirements.

## Baseline Comparison

The raw dashboard prompt was sent directly to Gemini 2.5 Flash as the baseline. The baseline response was workable, but it did not explicitly include the WebSocket reconnection requirement or acceptance-criteria structure. The optimized prompt addressed these gaps with clearer implementation steps and constraints.
