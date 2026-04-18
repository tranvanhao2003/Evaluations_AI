#!/usr/bin/env python3
"""
Langfuse Experiment Runner - Use run_experiment() to properly record experiments
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Any, Callable
from langfuse import Langfuse
from langfuse.experiment import Evaluation
import os
import json
from types import SimpleNamespace

from Evaluation_AI.config import Config
from Evaluation_AI.core.dataset_utils import DatasetLoader, TestCase
from Evaluation_AI.runners.stage_metrics import get_stage_metrics


class LangfuseExperimentRunner:
    """Run experiments using Langfuse's run_experiment() API"""
    
    def __init__(self):
        self.client = Langfuse(
            public_key=Config.LANGFUSE_PUBLIC_KEY,
            secret_key=Config.LANGFUSE_SECRET_KEY,
            host=Config.LANGFUSE_HOST
        )

    @staticmethod
    def _serialize_turn_for_metadata(turn) -> Dict[str, Any]:
        """
        Keep metadata compact: only include optional fields when they are meaningful.
        This avoids storing expected_tool_call=null for quality-only evaluations.
        """
        payload: Dict[str, Any] = {
            "role": turn.role,
            "content": turn.content,
        }
        if turn.expected_output_contains:
            payload["expected_output_contains"] = turn.expected_output_contains
        if turn.expected_tool_call:
            payload["expected_tool_call"] = turn.expected_tool_call
        return payload

    @staticmethod
    def _compact_item_metadata(test_case: TestCase, stage: str) -> Dict[str, Any]:
        """
        Keep dataset item metadata below telemetry propagation limits.
        Langfuse/OpenTelemetry may drop large propagated attributes around 200 chars.
        """
        original_metadata = test_case.metadata or {}
        compact_metadata: Dict[str, Any] = {
            "test_id": test_case.id,
            "test_name": test_case.name,
            "stage": stage,
            "criteria": test_case.criteria or [],
            "turn_count": len(test_case.turns or []),
        }

        for key in (
            "category",
            "difficulty",
            "industry",
            "target_audience",
            "job_level",
            "template_id",
            "tts_provider",
            "stt_mode",
        ):
            value = original_metadata.get(key)
            if value not in (None, "", [], {}):
                compact_metadata[key] = value

        return compact_metadata

    @staticmethod
    def _runtime_item_metadata(full_metadata: Dict[str, Any], stage: str) -> Dict[str, Any]:
        """
        Metadata propagated to experiment spans must stay under OTel baggage limits.
        Keep only the fields that are useful for trace inspection; keep the full
        metadata on the dataset item itself.
        """
        runtime_metadata: Dict[str, Any] = {"stage": stage}

        common_keys = ("test_id", "template_id", "category", "difficulty")
        stage_keys = {
            "script_generation": ("industry", "target_audience", "job_level", "video_duration"),
            "keyword_generation": ("industry", "target_audience", "job_level"),
            "stt_transcription": ("stt_mode", "tts_provider", "language"),
            "stt_raw_transcription": ("stt_mode", "tts_provider", "language"),
            "voice_splitting": ("source_stage", "language"),
            "subtitle_splitting": ("source_stage", "language"),
        }

        ordered_keys = common_keys + stage_keys.get(stage, ())
        optional_keys: List[str] = []

        for key in ordered_keys:
            value = full_metadata.get(key)
            if value not in (None, "", [], {}):
                runtime_metadata[key] = value
                optional_keys.append(key)

        while LangfuseExperimentRunner._metadata_payload_too_large(runtime_metadata) and optional_keys:
            runtime_metadata.pop(optional_keys.pop(), None)

        return runtime_metadata

    def _build_experiment_item(self, item: Any, stage: str) -> Any:
        """
        Wrap Langfuse dataset items so traces only propagate a compact metadata subset,
        while task execution still has access to the full dataset metadata.
        """
        full_metadata = getattr(item, "metadata", None) or {}
        if isinstance(full_metadata, str):
            try:
                full_metadata = json.loads(full_metadata)
            except json.JSONDecodeError:
                full_metadata = {"raw": full_metadata}
        if not isinstance(full_metadata, dict):
            full_metadata = {}

        runtime_metadata = self._runtime_item_metadata(full_metadata, stage)

        return SimpleNamespace(
            id=getattr(item, "id", None),
            dataset_id=getattr(item, "dataset_id", None),
            input=getattr(item, "input", None),
            expected_output=getattr(item, "expected_output", None),
            metadata=runtime_metadata,
            full_metadata=full_metadata,
            name=full_metadata.get("test_name") or str(full_metadata.get("test_id") or getattr(item, "id", "unknown")),
        )

    @staticmethod
    def _metadata_payload_too_large(metadata: Any) -> bool:
        """Guard against Langfuse/OpenTelemetry propagation limits."""
        try:
            serialized = json.dumps(metadata or {}, ensure_ascii=False)
        except (TypeError, ValueError):
            serialized = str(metadata or "")
        return len(serialized) > 180

    def _dataset_needs_refresh(self, dataset: Any, test_cases: List[TestCase], stage: str) -> bool:
        """
        Refresh existing datasets when item metadata uses an outdated, oversized schema
        or when the dataset no longer matches the local test-case set.
        """
        try:
            items = list(dataset.items or [])
        except Exception:
            return True

        if not items:
            return True

        if len(items) != len(test_cases):
            print(
                f"♻️  Dataset '{dataset.name}' item count differs from local test cases "
                f"({len(items)} != {len(test_cases)}), refreshing..."
            )
            return True

        expected_ids = {test_case.id for test_case in test_cases}
        actual_ids = set()

        for item in items:
            item_metadata = getattr(item, "metadata", None) or {}
            if isinstance(item_metadata, str):
                try:
                    item_metadata = json.loads(item_metadata)
                except json.JSONDecodeError:
                    item_metadata = {"raw": item_metadata}

            if not isinstance(item_metadata, dict):
                print(f"♻️  Dataset '{dataset.name}' has non-dict item metadata, refreshing...")
                return True

            if "original_metadata" in item_metadata or "turns" in item_metadata:
                print(f"♻️  Dataset '{dataset.name}' uses old verbose metadata, refreshing...")
                return True

            if item_metadata.get("stage") not in (None, stage):
                print(f"♻️  Dataset '{dataset.name}' stage metadata is stale, refreshing...")
                return True

            item_test_id = item_metadata.get("test_id")
            if item_test_id:
                actual_ids.add(item_test_id)

        if actual_ids and actual_ids != expected_ids:
            print(f"♻️  Dataset '{dataset.name}' test ids differ from local dataset, refreshing...")
            return True

        return False

    def _delete_dataset_items(self, dataset: Any) -> None:
        """Delete existing dataset items so the refreshed dataset uses the new compact schema."""
        items = list(getattr(dataset, "items", []) or [])
        if not items:
            return

        print(f"🧹 Refreshing dataset '{dataset.name}' by deleting {len(items)} stale items...")
        for idx, item in enumerate(items, 1):
            self.client.api.dataset_items.delete(id=item.id)
            if idx % 10 == 0 or idx == len(items):
                print(f"   ... {idx}/{len(items)} items deleted")

    def _populate_dataset(self, dataset_name: str, stage: str, test_cases: List[TestCase]) -> None:
        """Upload current local test cases to Langfuse with compact item metadata."""
        if not test_cases:
            print(f"⚠️  No test cases found for stage '{stage}'")
            return

        print(f"📥 Adding {len(test_cases)} items to dataset...")
        for idx, test_case in enumerate(test_cases, 1):
            try:
                item_input = test_case.turns[0].content if test_case.turns else test_case.name
                item_metadata = self._compact_item_metadata(test_case, stage)

                self.client.create_dataset_item(
                    dataset_name=dataset_name,
                    id=f"{dataset_name}:{test_case.id}",
                    input=item_input,
                    expected_output=(
                        test_case._expected_output
                        if getattr(test_case, "_expected_output", None) is not None
                        else {"stage": stage}
                    ),
                    metadata=item_metadata,
                )
                if idx % 10 == 0 or idx == len(test_cases):
                    print(f"   ... {idx}/{len(test_cases)} items added")
            except Exception as e:
                print(f"⚠️  Error adding item {test_case.id}: {e}")
    
    def _create_or_get_dataset(self, dataset_name: str, stage: str) -> Dict[str, Any]:
        """
        Create dataset if it doesn't exist, otherwise load existing one
        """
        try:
            loader = DatasetLoader()
            test_cases = loader.load_stage(stage)

            # Try to get existing dataset
            try:
                dataset = self.client.get_dataset(dataset_name)
                if dataset:
                    if self._dataset_needs_refresh(dataset, test_cases, stage):
                        self._delete_dataset_items(dataset)
                        self._populate_dataset(dataset_name, stage, test_cases)
                        return self.client.get_dataset(dataset_name)

                    if list(dataset.items):
                        print(f"✅ Found existing dataset '{dataset_name}'")
                        return dataset
            except:
                pass
            
            # Create new dataset
            print(f"📝 Creating new dataset '{dataset_name}'...")
            dataset = self.client.create_dataset(
                name=dataset_name,
                description=f"Evaluation dataset for {stage}"
            )
            
            self._populate_dataset(dataset_name, stage, test_cases)
            
            return self.client.get_dataset(dataset_name)
        except Exception as e:
            print(f"❌ Error creating/loading dataset: {e}")
            return None
    
    async def run_experiment_for_dataset(
        self,
        dataset_name: str,
        stage: str,
        task_fn: Callable
    ):
        """
        Run experiment for a dataset using Langfuse run_experiment()
        """
        
        valid_metrics = list(get_stage_metrics(stage))

        def _extract_task_metrics(output: Any) -> Dict[str, float]:
            if not isinstance(output, dict):
                return {}
            raw_metrics = output.get("metrics", {})
            if not isinstance(raw_metrics, dict):
                return {}
            cleaned = {}
            for metric_name in valid_metrics:
                metric_value = raw_metrics.get(metric_name)
                try:
                    if metric_value is not None:
                        cleaned[metric_name] = float(metric_value)
                    else:
                        cleaned[metric_name] = 0.0
                except (TypeError, ValueError):
                    cleaned[metric_name] = 0.0
            return cleaned

        def _make_item_metric_evaluator(metric_name: str):
            def evaluator(*, output, **kwargs):
                metrics = _extract_task_metrics(output)
                if metric_name not in metrics:
                    return Evaluation(
                        name=metric_name,
                        value=0.0,
                        comment=f"Metric '{metric_name}' missing from task output",
                        data_type="NUMERIC",
                    )
                return Evaluation(
                    name=metric_name,
                    value=metrics[metric_name],
                    comment=f"{stage}:{metric_name}",
                    data_type="NUMERIC",
                )
            return evaluator

        item_evaluators = [_make_item_metric_evaluator(metric_name) for metric_name in valid_metrics]

        def score_run_evaluator(**kwargs):
            """Aggregate per-metric averages for the whole run."""
            try:
                item_results = kwargs.get("item_results") or []
                score_buckets: Dict[str, List[float]] = {metric_name: [] for metric_name in valid_metrics}

                for item_result in item_results:
                    task_output = getattr(item_result, "output", None)
                    metrics = _extract_task_metrics(task_output)
                    for metric_name, metric_value in metrics.items():
                        score_buckets[metric_name].append(metric_value)

                evaluations = []
                for metric_name, values in score_buckets.items():
                    if not values:
                        continue
                    avg_value = sum(values) / len(values)
                    evaluations.append(
                        Evaluation(
                            name=f"avg_{metric_name}",
                            value=avg_value,
                            comment=f"Average {metric_name} across {len(values)} items",
                            data_type="NUMERIC",
                        )
                    )

                if evaluations:
                    print(f"   ✅ Recorded {len(evaluations)} run-level metric averages to Langfuse")
                return evaluations
            except Exception as e:
                print(f"⚠️  Error in score_run_evaluator: {e}")
                return []
        
        dataset = self._create_or_get_dataset(dataset_name, stage)
        if not dataset:
            return
        
        try:
            items = list(dataset.items)
            if not items:
                return
            print(f"📦 Loaded {len(items)} items from dataset '{dataset_name}'")
        except Exception as e:
            return

        experiment_items = [self._build_experiment_item(item, stage) for item in items]
        
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        experiment_name = f"Eval | {dataset_name} | {stage} | {timestamp}"
        
        print(f"🚀 Starting experiment: {experiment_name}")
        try:
            self.client.run_experiment(
                name=experiment_name,
                data=experiment_items,
                task=task_fn,
                run_name=experiment_name,
                evaluators=item_evaluators,
                run_evaluators=[score_run_evaluator],
                max_concurrency=1
            )
            print(f"✅ Experiment completed: {experiment_name}")
        except Exception as e:
            print(f"❌ Error running experiment: {e}")
    
    def flush(self):
        try:
            self.client.flush()
            print("✅ Langfuse data flushed")
        except Exception as e:
            print(f"⚠️  Error flushing: {e}")

if __name__ == "__main__":
    runner = LangfuseExperimentRunner()
    asyncio.run(runner.run_experiment_for_dataset(
        dataset_name="jd_script_dataset",
        stage="script_generation",
        task_fn=lambda x: {"status": "ok"}
    ))
    runner.flush()
