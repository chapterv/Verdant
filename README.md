# Verdant Verdict

Verdant Verdict is a production-ready **Green Agent** for evaluating Web Agents in the AgentBeats/AAA workflow.

## What This Agent Does

It acts as an evaluator, not a task executor:

1. Receives benchmark tasks.
2. Sends tasks to an assessee agent via A2A.
3. Collects outputs and judges success with transparent rules.
4. Computes metrics and weighted score.
5. Reports multi-run reliability (mean/std/consistency).

## Design Overview

The system is split into four layers:

- Protocol Layer (`src/a2a_protocol.py`)
A2A client/server helpers, task submission, polling, retries.

- Evaluation Engine (`src/agent.py`)
Task orchestration, judging logic, metric aggregation, reliability stats.

- Service/Controller Layer (`src/service.py`)
FastAPI endpoints for health, reset/restart, assessment run/query, RPC.

- Benchmark Utilities (`src/webarena_benchmark.py`)
Task dataset adapters and internal task conversion.

## Why It Is Useful

- Standardized benchmark scoring for different agent implementations.
- Regression tracking across model/prompt/system versions.
- Deployable evaluator service for competition and internal QA.

## Repo Layout

```text
.
├── config/
├── src/
├── .dockerignore
├── Dockerfile
├── render.yaml
├── requirements.txt
├── run.py
├── run.sh
└── run.ps1
```

## Quick Start

```bash
python -m pip install -r requirements.txt
python run.py --config config/config.yaml
```

Health check:

```bash
curl http://localhost:8080/health
```
