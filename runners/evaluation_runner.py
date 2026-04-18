#!/usr/bin/env python3
"""
Evaluation Runner - Orchestrate test execution with Langfuse integration
"""

import json
import re
import click
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from Evaluation_AI.config import Config
from Evaluation_AI.core.dataset_utils import DatasetLoader, TestCase
from Evaluation_AI.core.langfuse_manager import (
    LangfuseExperimentManager,
    LangfuseDatasetManager,
    LangfuseMetricsAggregator,
    LangfuseReportGenerator
)
from Evaluation_AI.core.base_evaluator import TestCaseResult, TurnEvalResult
from Evaluation_AI.metrics.script_eval import (
    ScriptStructureEvaluator,
    ScriptWordCountEvaluator,
    ScriptRelevanceEvaluator,
    ScriptToneEvaluator,
)
from Evaluation_AI.metrics.subtitle_eval import (
    SubtitleCPLEvaluator,
    SubtitleOrphanWordsEvaluator,
    SubtitleSyncEvaluator,
)
from Evaluation_AI.metrics.stt_eval import (
    STTPunctuationEvaluator,
    STTAccuracyEvaluator,
    STTTimestampEvaluator,
)
from Evaluation_AI.metrics.keyword_eval import (
    KeywordRelevanceEvaluator,
    SearchabilityEvaluator,
    KeywordDiversityEvaluator,
)
from Evaluation_AI.metrics.voice_splitting_eval import (
    SemanticCompletenessEvaluator,
    DurationBalanceEvaluator,
    NaturalPauseEvaluator,
)
from Evaluation_AI.backend.client import BackendClient
from Evaluation_AI.runners.stage_metrics import get_stage_metrics, STAGE_METRICS

class EvaluationRunner:
    """Chạy evaluation cho test cases with Langfuse tracking"""
    
    def __init__(self, use_mock: bool = None, dataset_name: str = None):
        if use_mock is None:
            use_mock = Config.USE_MOCK_BACKEND
        self.backend = BackendClient(
            base_url=Config.BACKEND_URL,
            use_mock=use_mock,
            strict_real=(not use_mock and Config.REQUIRE_REAL_BACKEND),
        )
        self.experiment_manager = LangfuseExperimentManager()
        self.dataset_manager = LangfuseDatasetManager()
        self.dataset_name = dataset_name
        self.stage_results = {}
        
        # Init evaluators - Match metric names from stage_metrics.py exactly
        self.script_evaluators = {
            "relevance": ScriptRelevanceEvaluator(),
            "structure": ScriptStructureEvaluator(),
            "tone_of_voice": ScriptToneEvaluator(),
            "length_constraint": ScriptWordCountEvaluator(),
        }

        self.stt_evaluators = {
            "word_error_rate": STTAccuracyEvaluator(),
            "punctuation_capitalization": STTPunctuationEvaluator(),
            "timestamp_accuracy": STTTimestampEvaluator(),
        }
        
        self.voice_splitting_evaluators = {
            "semantic_completeness": SemanticCompletenessEvaluator(),
            "duration_balance": DurationBalanceEvaluator(),
            "natural_pause": NaturalPauseEvaluator(),
        }
        
        self.subtitle_evaluators = {
            "readability": SubtitleCPLEvaluator(),
            "synchronization": SubtitleSyncEvaluator(),
            "line_break_logic": SubtitleOrphanWordsEvaluator(),
        }
        
        self.keyword_evaluators = {
            "visual_relevance": KeywordRelevanceEvaluator(),
            "searchability": SearchabilityEvaluator(),
            "diversity": KeywordDiversityEvaluator(),
        }

    @staticmethod
    def _clamp_score(value: Any) -> float:
        """Normalize any score-like value to [0, 1]."""
        try:
            score = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))
    
    def run_test_case(self, test_case: TestCase, dataset_item_id: str = None, skip_tracing: bool = False) -> TestCaseResult:
        """Chạy 1 test case với tất cả turns"""
        
        result = TestCaseResult(test_case.id, test_case.name, test_case.stage)
        
        # Create Langfuse experiment trace (with fallback)
        trace_id = None
        if not skip_tracing:
            try:
                trace_id = self.experiment_manager.create_trace(
                    test_case.id, 
                    test_case.stage,
                    dataset_item_id=dataset_item_id
                )
                result.langfuse_trace_id = trace_id
            except Exception as e:
                pass  # Langfuse error - continue without it
        
        context = {}
        
        try:
            for turn_idx, turn in enumerate(test_case.turns):
                turn_result = self._run_turn(turn, turn_idx, test_case, context)
                result.add_turn(turn_result)
                
                # Log turn to Langfuse
                if trace_id:
                    try:
                        self.experiment_manager.log_turn(
                            trace_id,
                            turn_idx,
                            turn_result.to_dict()
                        )
                    except Exception:
                        pass  # Langfuse error - continue
                
                # Save context for next turn
                if "output" in context and turn_idx < len(test_case.turns) - 1:
                    context["last_output"] = context.get("output")
        
        except Exception as e:
            result.error = str(e)
        
        # Finalize
        result.finalize()
        
        # Collect all metrics
        all_metrics = {}
        if result.turn_results:
            for turn in result.turn_results:
                all_metrics.update(turn.metrics)
        
        # Log metrics to Langfuse
        if trace_id and all_metrics:
            try:
                self.experiment_manager.log_metrics(trace_id, all_metrics, test_case.stage)
            except Exception:
                pass  # Langfuse error - continue
        
        # End experiment with final result
        if trace_id:
            try:
                self.experiment_manager.end_experiment(
                    trace_id,
                    result.passed,
                    result.overall_score
                )
            except Exception:
                pass  # Langfuse error - continue
        
        # Track stage results
        if test_case.stage not in self.stage_results:
            self.stage_results[test_case.stage] = []
        self.stage_results[test_case.stage].append({
            "test_id": test_case.id,
            "metrics": all_metrics,
            "passed": result.passed,
            "score": result.overall_score
        })
        
        return result
    
    def _run_turn(self, turn, turn_idx: int, test_case: TestCase, context: dict) -> TurnEvalResult:
        """Chạy 1 turn"""
        
        turn_result = TurnEvalResult(turn_idx, turn.content)
        
        try:
            # Execute backend
            output = self._execute_backend(turn, test_case, context)
            context["output"] = output
            if getattr(test_case, "_expected_output", None) is not None:
                context["expected_output"] = test_case._expected_output
            
            # If this was a script generation turn, save it as ground truth for future turns (like STT)
            if test_case.stage == "script_generation":
                if isinstance(output, dict):
                    # Combine hook, body, cta if they exist
                    script_text = output.get("full_script")
                    if isinstance(script_text, str) and script_text.strip():
                        context["generated_script"] = script_text.strip()
                    else:
                        script_parts = [output.get(k, "") for k in ["hook", "body", "cta"] if output.get(k)]
                        context["generated_script"] = " ".join(script_parts)
                else:
                    context["generated_script"] = str(output)
            
            # Check 1: Tool call match
            if turn.expected_tool_call:
                tool_match = self._check_tool_call(output, turn.expected_tool_call)
                turn_result.checks["tool_call_match"] = tool_match
                if not tool_match:
                    turn_result.passed = False
            
            # Check 2: Output contains
            if turn.expected_output_contains:
                output_str = json.dumps(output) if isinstance(output, (dict, list)) else str(output)
                contains_all = all(
                    str(item).lower() in output_str.lower()
                    for item in turn.expected_output_contains
                )
                turn_result.checks["output_contains"] = contains_all
                if not contains_all:
                    turn_result.passed = False
            
            # Check 3: Metrics from criteria
            metric_names = self._resolve_metric_names(test_case)
            if metric_names:
                metrics = self._evaluate_metrics(
                    output, metric_names, test_case, turn, context=context
                )
                turn_result.metrics = metrics
                
                # Check if pass threshold
                for metric_name, score in metrics.items():
                    threshold = Config.get_threshold(metric_name)
                    if score < threshold:
                        turn_result.passed = False
            
            # Calculate score
            check_scores = [1.0 if v else 0.0 for v in turn_result.checks.values()]
            metric_scores = list(turn_result.metrics.values())
            all_scores = check_scores + metric_scores
            turn_result.score = self._clamp_score(
                (sum(all_scores) / len(all_scores)) if all_scores else 0.0
            )
            
        except Exception as e:
            turn_result.passed = False
            turn_result.error = str(e)
            turn_result.score = 0.0
        
        return turn_result
    
    def _execute_backend(self, turn, test_case, context):
        """Execute backend call"""
        if not turn.expected_tool_call:
            return self._execute_stage_backend(turn.content, test_case, context)
        
        tool_name = turn.expected_tool_call.get("name")
        args = turn.expected_tool_call.get("arguments", {})
        
        # Replace placeholders
        for key, val in args.items():
            if isinstance(val, str) and val.startswith("$"):
                placeholder = val[1:].lower()
                args[key] = context.get(placeholder, val)
        
        # Call backend
        if tool_name == "generate_script":
            return self.backend.generate_script(**args)
        elif tool_name == "transcribe_text":
            return self.backend.transcribe_text(**args)
        elif tool_name == "transcribe_raw_text":
            return self.backend.transcribe_raw_text(**args)
        elif tool_name == "split_voice":
            return self.backend.split_voice(**args)
        elif tool_name == "split_subtitles":
            return self.backend.split_subtitles(**args)
        elif tool_name == "generate_keywords":
            return self.backend.generate_keywords(**args)
        else:
            return {"status": "unknown"}

    def _parse_turn_content(self, content: str):
        raw = str(content or "").strip()
        if not raw:
            return ""
        try:
            return json.loads(raw)
        except Exception:
            return raw

    def _execute_stage_backend(self, turn_content: str, test_case: TestCase, context: dict):
        """Execute stage backend based on dataset input when no explicit tool call is provided."""
        parsed = self._parse_turn_content(turn_content)
        stage = test_case.stage

        if stage == "script_generation":
            if isinstance(parsed, dict):
                jd_text = str(parsed.get("jd_content") or parsed.get("input") or parsed.get("text") or "").strip()
            else:
                jd_text = str(parsed).strip()
            template_id = 1
            metadata = test_case.metadata or {}
            if isinstance(metadata, dict):
                template_meta = metadata.get("template_id")
                if isinstance(template_meta, int):
                    template_id = template_meta
            video_duration = metadata.get("video_duration", 60) if isinstance(metadata, dict) else 60
            if not jd_text:
                return ""
            template_id = self.backend.resolve_script_template_id(
                template_id=template_id,
                metadata=metadata if isinstance(metadata, dict) else {},
                case_name=test_case.name,
            )
            return self.backend.generate_script(
                jd_text=jd_text,
                template_id=template_id,
                video_duration=int(video_duration) if isinstance(video_duration, (int, float)) else 60,
            )

        if stage == "stt_transcription":
            transcript_source = ""
            stt_kwargs = {}
            metadata = test_case.metadata or {}
            if isinstance(parsed, dict):
                transcript_source = str(
                    parsed.get("text")
                    or parsed.get("transcript")
                    or parsed.get("script_text")
                    or parsed.get("content")
                    or ""
                ).strip()
                for key in ("audio_url", "audio_path", "language", "session_id"):
                    value = parsed.get(key)
                    if value is not None:
                        stt_kwargs[key] = value
            else:
                transcript_source = str(parsed).strip()

            if isinstance(metadata, dict):
                for key in ("tts_provider", "voice_id", "language", "gender", "speed", "split_by_sentence"):
                    value = metadata.get(key)
                    if value is not None:
                        stt_kwargs.setdefault(key, value)

            stt_kwargs.setdefault("tts_provider", "vieneu")
            stt_kwargs.setdefault("language", "vi")

            if not transcript_source:
                transcript_source = self._extract_text_from_turn_content(turn_content)
            return self.backend.transcribe_text(text=transcript_source, **stt_kwargs)

        if stage == "stt_raw_transcription":
            transcript_source = ""
            stt_kwargs = {}
            metadata = test_case.metadata or {}
            if isinstance(parsed, dict):
                transcript_source = str(
                    parsed.get("text")
                    or parsed.get("transcript")
                    or parsed.get("script_text")
                    or parsed.get("content")
                    or ""
                ).strip()
                for key in ("audio_url", "audio_path", "language", "session_id"):
                    value = parsed.get(key)
                    if value is not None:
                        stt_kwargs[key] = value
            else:
                transcript_source = str(parsed).strip()

            if isinstance(metadata, dict):
                for key in ("tts_provider", "voice_id", "language", "gender", "speed", "split_by_sentence"):
                    value = metadata.get(key)
                    if value is not None:
                        stt_kwargs.setdefault(key, value)

            stt_kwargs.setdefault("tts_provider", "vieneu")
            stt_kwargs.setdefault("language", "vi")

            if not transcript_source:
                transcript_source = self._extract_text_from_turn_content(turn_content)
            return self.backend.transcribe_raw_text(text=transcript_source, **stt_kwargs)

        if stage == "voice_splitting":
            script_text = ""
            if isinstance(parsed, dict):
                script_text = str(parsed.get("script_text") or parsed.get("text") or "").strip()
            else:
                script_text = str(parsed).strip()
            if not script_text:
                script_text = self._extract_text_from_turn_content(turn_content)
            return self.backend.split_voice(text=script_text)

        if stage == "subtitle_splitting":
            subtitle_text = self._extract_text_from_turn_content(turn_content)
            word_timings = []
            if isinstance(parsed, dict):
                raw_word_timings = parsed.get("word_timings")
                if isinstance(raw_word_timings, list):
                    word_timings = raw_word_timings
            return self.backend.split_subtitles(text=subtitle_text, word_timings=word_timings)

        if stage == "keyword_generation":
            text = self._extract_text_from_turn_content(turn_content)
            captions = None
            job_category = "recruitment"
            if isinstance(parsed, dict):
                raw_captions = parsed.get("captions")
                if isinstance(raw_captions, list):
                    captions = raw_captions
                job_val = parsed.get("job_category")
                if isinstance(job_val, str) and job_val.strip():
                    job_category = job_val.strip()
            return self.backend.generate_keywords(
                text=text,
                captions=captions,
                job_category=job_category,
            )

        return self._build_stage_fallback_output(turn_content, stage)

    def _extract_text_from_turn_content(self, content: str) -> str:
        """Extract primary text from dataset turn content."""
        raw = str(content or "").strip()
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                for key in ("script_text", "text", "content", "input", "jd_content", "full_text", "transcript"):
                    val = parsed.get(key)
                    if isinstance(val, str) and val.strip():
                        return val.strip()
                captions = parsed.get("captions")
                if isinstance(captions, list):
                    parts = []
                    for caption in captions:
                        if isinstance(caption, dict):
                            cap_text = caption.get("text")
                            if isinstance(cap_text, str) and cap_text.strip():
                                parts.append(cap_text.strip())
                    if parts:
                        return " ".join(parts).strip()
            if isinstance(parsed, str):
                return parsed.strip()
            if isinstance(parsed, list):
                parts = []
                for item in parsed:
                    if isinstance(item, dict):
                        cap_text = item.get("text")
                        if isinstance(cap_text, str) and cap_text.strip():
                            parts.append(cap_text.strip())
                if parts:
                    return " ".join(parts).strip()
        except Exception:
            pass
        return raw

    def _build_stage_fallback_output(self, turn_content: str, stage: str):
        """Generate deterministic fallback output when no backend call is defined."""
        text = self._extract_text_from_turn_content(turn_content)

        return text
    
    def _check_tool_call(self, output, expected_tool):
        """Check tool call match"""
        output_str = json.dumps(output) if isinstance(output, (dict, list)) else str(output)
        expected_name = expected_tool.get("name", "")
        return expected_name.lower() in output_str.lower() or "mock" in output_str.lower()

    def _resolve_metric_names(self, test_case: TestCase) -> List[str]:
        """Resolve metrics from stage definition if criteria is not set"""
        # Priority 1: Use criteria if provided
        if isinstance(test_case.criteria, list) and test_case.criteria:
            return [str(metric).strip() for metric in test_case.criteria if str(metric).strip()]
        
        # Priority 2: Use stage metrics from STAGE_METRICS
        stage_metrics = get_stage_metrics(test_case.stage)
        if stage_metrics:
            return stage_metrics
        
        # Default: empty list
        return []
    
    def _evaluate_metrics(self, output, metric_names, test_case, turn, context=None):
        """Run metric evaluators - Route to correct evaluator group based on stage"""
        metrics = {}
        context = context or {}
        
        for metric_name in metric_names:
            try:
                # Find the right evaluator group based on stage
                evaluator_group = None
                
                if test_case.stage == "script_generation":
                    evaluator_group = self.script_evaluators
                elif test_case.stage in ("stt_transcription", "stt_raw_transcription"):
                    evaluator_group = self.stt_evaluators
                elif test_case.stage == "voice_splitting":
                    evaluator_group = self.voice_splitting_evaluators
                elif test_case.stage == "subtitle_splitting":
                    evaluator_group = self.subtitle_evaluators
                elif test_case.stage == "keyword_generation":
                    evaluator_group = self.keyword_evaluators
                
                if not evaluator_group:
                    # No evaluator group for this stage - skip
                    metrics[metric_name] = 0.5  # Default neutral score
                    continue
                
                evaluator = evaluator_group.get(metric_name)
                if evaluator:
                    result = evaluator.evaluate(
                        turn.content,
                        output,
                        context=context,
                        test_case=test_case,
                        turn=turn,
                        expected_output=context.get("expected_output"),
                    )
                    metric_score = self._clamp_score(result.score) if result and result.score is not None else 0.0
                    metrics[metric_name] = float(metric_score)
                else:
                    # Metric not found in evaluator group - skip
                    metrics[metric_name] = 0.5  # Default neutral score
            
            except Exception as e:
                metrics[metric_name] = self._clamp_score(0.7)
        
        return metrics

@click.command()
@click.option('--stage', help='Run tests for specific stage')
@click.option('--test-id', help='Run specific test case')
@click.option('--dataset', help='Langfuse dataset name (optional)')
@click.option('--output', default='results/test_results.json', help='Output file')
@click.option('--report', is_flag=True, help='Generate report')
def run_tests(stage, test_id, dataset, output, report):
    """Run evaluation tests with Langfuse integration"""
    
    click.echo("\n🚀 Evaluation AI - Starting Tests")
    click.echo("=" * 60)
    
    Config.validate()
    
    # Load test cases
    if test_id:
        all_cases = DatasetLoader.load_all()
        test_cases = [tc for tc in all_cases if tc.id == test_id]
    elif stage:
        test_cases = DatasetLoader.load_stage(stage)
    else:
        test_cases = DatasetLoader.load_all()
    
    if not test_cases:
        click.echo("❌ No test cases found")
        return
    
    click.echo(f"📋 Found {len(test_cases)} test cases\n")
    
    # Run tests
    runner = EvaluationRunner(use_mock=Config.USE_MOCK_BACKEND, dataset_name=dataset)
    
    # Create/get Langfuse dataset
    dataset_id = None
    if dataset:
        dataset_id = runner.dataset_manager.create_or_get_dataset(
            dataset,
            description=f"Evaluation dataset for {stage or 'all stages'}"
        )
    
    # Run all tests
    all_results = []
    passed_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        click.echo(f"[{i}/{len(test_cases)}] {test_case.id}: {test_case.name}")
        
        # Add test case as item to dataset if dataset specified
        item_id = None
        if dataset and dataset_id:
            try:
                item_id = runner.dataset_manager.add_dataset_item(
                    dataset,
                    {
                        "id": test_case.id,
                        "name": test_case.name,
                        "stage": test_case.stage,
                        "turns": [{"content": turn.content} for turn in test_case.turns],
                        "criteria": test_case.criteria,
                        "metadata": test_case.metadata or {}
                    }
                )
            except Exception:
                pass  # Error adding item, continue anyway
        
        result = runner.run_test_case(test_case, dataset_item_id=item_id)
        all_results.append(result)
        
        # Link result to dataset item by creating an experiment
        if dataset and item_id:
            try:
                # Create experiment linking this test result to the dataset item
                runner.experiment_manager.link_result_to_item(
                    item_id,
                    result.passed,
                    result.overall_score,
                    result.to_dict()
                )
            except Exception:
                pass  # Error linking, continue anyway
        
        if result.passed:
            passed_count += 1
            click.echo(f"     ✅ PASSED (score: {result.overall_score:.2f})")
        else:
            click.echo(f"     ❌ FAILED (score: {result.overall_score:.2f})")
        
        if result.langfuse_trace_id:
            click.echo(f"     📊 Trace: {result.langfuse_trace_id}")
    
    # Save results
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump([r.to_dict() for r in all_results], f, indent=2, ensure_ascii=False)
    
    # Flush Langfuse (non-blocking)
    try:
        runner.experiment_manager.flush()
    except Exception:
        pass  # Langfuse flush error - continue
    
    # Generate report
    click.echo(f"\n{'='*60}")
    click.echo(f"📊 SUMMARY: {passed_count}/{len(test_cases)} tests passed")
    if all_results:
        avg_score = sum(r.overall_score for r in all_results) / len(all_results)
        click.echo(f"   Avg Score: {avg_score:.2f}")
        click.echo(f"   Success Rate: {(passed_count/len(test_cases))*100:.1f}%")
    click.echo(f"   Results: {output}")
    
    # Show stage-wise breakdown
    if report and runner.stage_results:
        click.echo(f"\n{'='*60}")
        click.echo("📈 STAGE-WISE BREAKDOWN")
        click.echo(f"{'='*60}")
        
        for stage_name, results in runner.stage_results.items():
            passed = sum(1 for r in results if r["passed"])
            avg_score = sum(r["score"] for r in results) / len(results) if results else 0
            click.echo(f"\n{stage_name}:")
            click.echo(f"  Passed: {passed}/{len(results)}")
            click.echo(f"  Avg Score: {avg_score:.2f}")
            
            # Aggregate metrics
            if results and results[0]["metrics"]:
                aggregated = LangfuseMetricsAggregator.aggregate_stage_metrics(results)
                for metric_name, stats in aggregated.items():
                    threshold = stats.get("threshold", 0.7)
                    avg = stats.get("avg", 0)
                    status = "✅" if avg >= threshold else "❌"
                    click.echo(f"  {status} {metric_name}: {avg:.2f} (threshold: {threshold:.2f})")
    
    click.echo(f"{'='*60}\n")

if __name__ == "__main__":
    run_tests()
