# Verdant Verdict

Verdant Verdict is a production-ready evaluator stack for AgentBeats/AAA workflows.

## What This Project Does

It contains two cooperating services:

1. Green (`run.py`): benchmark evaluator and scoring controller.
2. Purple (`run_purple.py`): reference assessee agent with dual-pass verification.

## Design Overview

The system is split into five layers:

- Protocol Layer (`src/a2a_protocol.py`)
A2A client/server helpers, task submission, polling, retries.

- Evaluation Engine (`src/agent.py`)
Task orchestration, judging logic, metric aggregation, reliability stats.

- Green Service Layer (`src/service.py`)
FastAPI endpoints for health, reset/restart, assessment run/query, RPC.

- Purple Assessee Layer (`src/purple_agent.py`, `src/purple_service.py`)
Deterministic execution, dual-pass checks, and trace export.

- Benchmark Utilities (`src/webarena_benchmark.py`)
Task dataset adapters and internal task conversion.

## Why It Is Useful

- Standardized benchmark scoring for different agent implementations.
- Regression tracking across model/prompt/system versions.
- Deployable evaluator + assessee baseline for competition and internal QA.

## Repo Layout

```text
.
├── config/
├── docs/
├── src/
├── .dockerignore
├── Dockerfile
├── render.yaml
├── requirements.txt
├── run.py
├── run_purple.py
├── run.sh
└── run.ps1
```

## Quick Start

```bash
python -m pip install -r requirements.txt
```

Start Purple service (assessee):

```bash
python run_purple.py --config config/purple_config.yaml
```

Start Green service (evaluator):

```bash
python run.py --config config/config.yaml
```

Health checks:

```bash
curl http://localhost:8080/health
curl http://localhost:8001/health
```
