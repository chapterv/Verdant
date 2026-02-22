"""Benchmark quality analysis helpers for Type-1 submissions."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .agent import Task

PLACEHOLDER_HOSTS = (
    "example.com",
    "example.org",
    "example.net",
)


def _task_to_dict(task: Task) -> Dict[str, Any]:
    return {
        "id": task.id,
        "description": task.description,
        "target_url": task.target_url,
        "expected_result": task.expected_result,
        "timeout": task.timeout,
        "metadata": task.metadata,
    }


def analyze_tasks(tasks: Sequence[Task]) -> Dict[str, Any]:
    """Generate a deterministic quality report for task set analysis."""
    if not tasks:
        raise ValueError("Task list is empty")

    task_ids = [task.id for task in tasks]
    duplicate_ids = [task_id for task_id, count in Counter(task_ids).items() if count > 1]

    missing_expected = [task.id for task in tasks if not task.expected_result.strip()]
    missing_target = [task.id for task in tasks if not task.target_url.strip()]

    categories = Counter(str(task.metadata.get("category", "unknown")) for task in tasks)
    difficulties = Counter(str(task.metadata.get("difficulty", "unknown")) for task in tasks)

    placeholder_urls: List[str] = []
    for task in tasks:
        target_url = task.target_url.lower()
        if any(host in target_url for host in PLACEHOLDER_HOSTS):
            placeholder_urls.append(task.id)

    max_steps = [int(task.metadata.get("max_steps", 0)) for task in tasks if int(task.metadata.get("max_steps", 0)) > 0]
    timeout_values = [int(task.timeout) for task in tasks]

    report = {
        "total_tasks": len(tasks),
        "duplicate_task_ids": duplicate_ids,
        "missing_expected_result": missing_expected,
        "missing_target_url": missing_target,
        "placeholder_url_tasks": placeholder_urls,
        "category_distribution": dict(categories),
        "difficulty_distribution": dict(difficulties),
        "timeout_summary": {
            "min": min(timeout_values),
            "max": max(timeout_values),
            "mean": round(statistics.mean(timeout_values), 2),
        },
        "max_steps_summary": {
            "min": min(max_steps) if max_steps else None,
            "max": max(max_steps) if max_steps else None,
            "mean": round(statistics.mean(max_steps), 2) if max_steps else None,
        },
        "quality_flags": {
            "has_duplicates": bool(duplicate_ids),
            "has_missing_expected": bool(missing_expected),
            "has_missing_target": bool(missing_target),
            "has_placeholder_urls": bool(placeholder_urls),
        },
        "recommendations": _build_recommendations(
            duplicate_ids=duplicate_ids,
            missing_expected=missing_expected,
            missing_target=missing_target,
            placeholder_urls=placeholder_urls,
            categories=categories,
            difficulties=difficulties,
        ),
    }
    return report


def _build_recommendations(
    duplicate_ids: Sequence[str],
    missing_expected: Sequence[str],
    missing_target: Sequence[str],
    placeholder_urls: Sequence[str],
    categories: Counter,
    difficulties: Counter,
) -> List[str]:
    recommendations: List[str] = []

    if duplicate_ids:
        recommendations.append("Remove duplicate task IDs to avoid score contamination.")
    if missing_expected:
        recommendations.append("Fill expected_result for all tasks to enable deterministic judging.")
    if missing_target:
        recommendations.append("Provide target_url for all tasks or document why navigation is optional.")
    if placeholder_urls:
        recommendations.append("Replace placeholder domains via environment.url_overrides before production runs.")
    if len(categories) < 3:
        recommendations.append("Expand category coverage for stronger benchmark diversity.")
    if len(difficulties) < 3:
        recommendations.append("Add easy/medium/hard coverage to improve scale and realism.")

    if not recommendations:
        recommendations.append("No structural blockers detected. Proceed with manual validation sampling.")

    return recommendations


def save_report(report: Dict[str, Any], output_path: str) -> str:
    """Persist report as pretty JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def analyze_and_save(tasks: Iterable[Task], output_path: str) -> Dict[str, Any]:
    """Convenience wrapper for CLI/script usage."""
    report = analyze_tasks(list(tasks))
    save_report(report, output_path)
    return report
