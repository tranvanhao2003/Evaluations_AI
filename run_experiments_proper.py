#!/usr/bin/env python3
"""
Proper Langfuse Experiment Runner using run_experiment()
"""

import asyncio
import sys
import os
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Evaluation_AI.config import Config
from Evaluation_AI.runners.langfuse_experiment_runner import LangfuseExperimentRunner
from Evaluation_AI.runners.evaluation_runner import EvaluationRunner
from Evaluation_AI.runners.stage_metrics import get_stage_metrics


async def get_evaluation_task(stage: str):
    """Create task function for a specific stage"""
    
    async def eval_task(item) -> dict:
        """Execute evaluation for an item"""
        try:
            # Extract input
            if isinstance(item, dict):
                item_input = item.get('input', '')
                item_id = item.get('id', 'unknown')
                item_name = item.get('name', str(item_id))
            else:
                item_input = getattr(item, 'input', '')
                item_id = getattr(item, 'id', 'unknown')
                item_name = getattr(item, 'name', str(item_id))
            
            # Create test case from item
            from Evaluation_AI.core.dataset_utils import TestCase, TestTurn
            
            # Extract metadata and criteria
            item_metadata = {}
            item_criteria = []
            item_expected_output = None
            if hasattr(item, 'full_metadata'):
                raw_metadata = item.full_metadata
                item_metadata = raw_metadata if isinstance(raw_metadata, dict) else (json.loads(raw_metadata) if isinstance(raw_metadata, str) else {})
                item_criteria = getattr(item, 'criteria', None) or item_metadata.get('criteria', [])
                item_expected_output = getattr(item, 'expected_output', None)
            elif hasattr(item, 'metadata'):
                item_metadata = item.metadata if isinstance(item.metadata, dict) else (json.loads(item.metadata) if isinstance(item.metadata, str) else {})
                item_criteria = getattr(item, 'criteria', None) or item_metadata.get('criteria', [])
                item_expected_output = getattr(item, 'expected_output', None)
            elif isinstance(item, dict):
                item_metadata = item.get('metadata', {})
                item_criteria = item.get('criteria', []) or item_metadata.get('criteria', [])
                item_expected_output = item.get('expected_output') or item.get('actual_output')
            
            # If criteria is still empty, use stage metrics as fallback
            if not item_criteria:
                item_criteria = list(get_stage_metrics(stage))
            
            test_case = TestCase(
                id=item_id,
                name=item_name,
                stage=stage,
                criteria=item_criteria,
                turns=[TestTurn(
                    role='user',
                    content=(
                        str(item_input)
                        if not isinstance(item_input, dict)
                        else json.dumps(item_input, ensure_ascii=False)
                    )
                )],
                metadata=item_metadata,
                _expected_output=item_expected_output,
            )
            
            # Run evaluation
            runner = EvaluationRunner(use_mock=Config.USE_MOCK_BACKEND)
            result = runner.run_test_case(test_case)
            
            # Collect metrics from turn results
            metrics = {}
            if hasattr(result, 'turn_results') and result.turn_results:
                for turn in result.turn_results:
                    if hasattr(turn, 'metrics'):
                        metrics.update(turn.metrics or {})
            
            # Return result as dict (scores will be created by run_evaluator)
            return {
                "passed": result.passed,
                "score": float(result.overall_score or 0.0),
                "metrics": {
                    k: float(v) if isinstance(v, (int, float)) else 0.0 
                    for k, v in metrics.items()
                },
                "status": "PASSED" if result.passed else "FAILED",
                "test_id": item_id,
                "test_name": item_name
            }
        
        except Exception as e:
            print(f"❌ Error evaluating {getattr(item, 'id', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()
            return {
                "passed": False,
                "score": 0.0,
                "metrics": {},
                "error": str(e),
                "status": "ERROR",
                "test_id": getattr(item, 'id', 'unknown')
            }
    
    return eval_task


async def main():
    """Run all experiments"""
    
    runner = LangfuseExperimentRunner()
    
    # Define experiments - full active evaluation suite
    experiments = [
        ("jd_script_dataset", "script_generation"),
        ("jd_stt_dataset", "stt_transcription"),
        ("jd_stt_raw_dataset", "stt_raw_transcription"),
        ("jd_voice_splitting_dataset", "voice_splitting"),  # NEW: Voice splitting stage
        ("jd_subtitle_dataset", "subtitle_splitting"),
        ("jd_image_search_dataset", "image_search_generation"),
        ("jd_video_search_dataset", "video_search_generation"),
    ]
    
    print("\n🚀 Running Langfuse Experiments with run_experiment()\n")
    
    for dataset_name, stage in experiments:
        print(f"📊 Running experiment for dataset: {dataset_name}")
        
        # Get task function for this stage
        task_fn = await get_evaluation_task(stage)
        
        # Run experiment
        # Note: Evaluators are not needed when using score_current_span() in task_fn
        await runner.run_experiment_for_dataset(
            dataset_name=dataset_name,
            stage=stage,
            task_fn=task_fn
        )
        
        print(f"✅ Completed: {dataset_name}\n")
    
    # Flush all data
    runner.flush()
    
    # Wait a bit for Langfuse to process
    import time
    time.sleep(2)
    
    print("\n🎉 All experiments completed!")
    print("✅ Check Langfuse dashboard for results")
    print("   URL: https://cloud.langfuse.com")


if __name__ == "__main__":
    asyncio.run(main())
