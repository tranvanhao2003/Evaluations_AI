#!/usr/bin/env python3
"""Run a single evaluation stage locally or via Langfuse experiment API."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent directory to path so script works when run as `python Evaluation_AI/...`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Evaluation_AI.config import Config
from Evaluation_AI.core.dataset_utils import DatasetLoader
from Evaluation_AI.runners.evaluation_runner import EvaluationRunner
from Evaluation_AI.run_experiments_proper import get_evaluation_task


def run_local(stage: str, output_path: Path) -> int:
    """Run single-stage evaluation against local datasets."""
    test_cases = DatasetLoader.load_stage(stage)
    if not test_cases:
        print(f"No test cases found for stage '{stage}'")
        return 1

    runner = EvaluationRunner(use_mock=Config.USE_MOCK_BACKEND)
    results = []
    for case in test_cases:
        result = runner.run_test_case(case, skip_tracing=True)
        results.append(result.to_dict())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    passed = sum(1 for item in results if item.get("passed"))
    avg_score = sum(item.get("overall_score", 0.0) for item in results) / len(results)
    print(f"Stage: {stage}")
    print(f"Passed: {passed}/{len(results)} ({(passed / len(results)) * 100:.1f}%)")
    print(f"Average score: {avg_score:.3f}")
    print(f"Results written to: {output_path}")
    return 0


async def run_langfuse(stage: str, dataset: str) -> int:
    """Run a single experiment with Langfuse run_experiment()."""
    if not dataset:
        print("--dataset is required when using --langfuse")
        return 1

    from Evaluation_AI.runners.langfuse_experiment_runner import LangfuseExperimentRunner

    runner = LangfuseExperimentRunner()
    task_fn = await get_evaluation_task(stage)
    await runner.run_experiment_for_dataset(dataset_name=dataset, stage=stage, task_fn=task_fn)
    runner.flush()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run single stage evaluation")
    parser.add_argument("--stage", required=True, help="Stage name, e.g. script_generation")
    parser.add_argument("--dataset", default="", help="Langfuse dataset name (required with --langfuse)")
    parser.add_argument("--output", default="Evaluation_AI/results/test_results.json", help="Local output path")
    parser.add_argument("--langfuse", action="store_true", help="Run with Langfuse run_experiment()")
    args = parser.parse_args()

    if args.langfuse:
        if not Config.LANGFUSE_ENABLED:
            print("Langfuse is not configured. Falling back to local mode.")
        else:
            return asyncio.run(run_langfuse(args.stage, args.dataset))

    return run_local(args.stage, Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())
