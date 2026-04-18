"""
Dataset Loading & Langfuse Integration
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from Evaluation_AI.config import Config

@dataclass
class TestTurn:
    """1 turn trong test case"""
    role: str  # "user"
    content: str
    expected_output_contains: List[str] = None
    expected_tool_call: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.expected_output_contains is None:
            self.expected_output_contains = []

@dataclass
class TestCase:
    """1 test case hoàn chỉnh"""
    id: str
    name: str
    stage: str
    criteria: List[str]
    turns: List[TestTurn]
    metadata: Dict[str, Any] = None
    _expected_output: Any = None  # Optional reference output for assertions/judging
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class DatasetLoader:
    """Load test cases từ JSON datasets"""
    
    @staticmethod
    def load_json(file_path: str) -> List[TestCase]:
        """Load test cases từ JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_cases = []
        for item in data:
            # Support 2 formats:
            # Format A: input + expected/reference output (no turns)
            # Format B: turns array
            
            if 'turns' in item and item['turns']:
                # Format B
                turns = [
                    TestTurn(
                        role=turn.get('role', 'user'),
                        content=turn.get('content', ''),
                        expected_output_contains=turn.get('expected_output_contains', []),
                        expected_tool_call=turn.get('expected_tool_call')
                    )
                    for turn in item.get('turns', [])
                ]
            else:
                # Format A: Convert input/output to turns
                input_text = item.get('input', '')
                if isinstance(input_text, dict):
                    input_text = json.dumps(input_text)
                
                expected_output = item.get('expected_output', item.get('actual_output', ''))
                if isinstance(expected_output, dict):
                    expected_output = json.dumps(expected_output)
                elif isinstance(expected_output, list):
                    expected_output = json.dumps(expected_output)
                
                turns = [
                    TestTurn(
                        role='user',
                        content=str(input_text),
                        expected_output_contains=item.get('expected_output_contains', []),
                        expected_tool_call=item.get('expected_tool_call')
                    )
                ]
            
            test_case = TestCase(
                id=item['id'],
                name=item['name'],
                stage=item.get('stage') or item.get('metadata', {}).get('stage', ''),
                criteria=item.get('criteria', []),
                turns=turns,
                metadata=item.get('metadata', {}),
                _expected_output=item.get('expected_output', item.get('actual_output'))
            )
            test_cases.append(test_case)
        
        return test_cases
    
    @staticmethod
    def load_stage(stage: str) -> List[TestCase]:
        """Load test cases cho 1 stage"""
        stage_map = {
            "script_generation": ["all_templates_evaluation.json"],
            "stt_transcription": ["stt_transcription_test.json"],
            "stt_raw_transcription": ["stt_raw_transcription_test.json"],
            "transcription": ["stt_transcription_test.json"],
            "voice_splitting": ["voice_splitting_test.json"],  # Use dedicated voice splitting dataset
            "subtitle_splitting": ["subtitle_splitting_test.json"],
            "keyword_generation": ["keyword_generation_test.json"],
        }
        
        filenames = stage_map.get(stage, [])
        if not filenames:
            return []
        
        all_cases = []
        for filename in filenames:
            file_path = Config.DATASETS_DIR / filename
            if not file_path.exists():
                # Silently skip missing files instead of warning
                continue
            
            all_cases.extend(DatasetLoader.load_json(str(file_path)))
        
        return all_cases
    
    @staticmethod
    def load_all() -> List[TestCase]:
        """Load tất cả test cases"""
        all_cases = []
        for stage in Config.STAGES.values():
            all_cases.extend(DatasetLoader.load_stage(stage))
        return all_cases

class LangfuseManager:
    """Quản lý tích hợp Langfuse"""
    
    def __init__(self):
        self.enabled = Config.LANGFUSE_ENABLED
        self.client = None
        
        if self.enabled:
            try:
                from langfuse import Langfuse
                self.client = Langfuse(
                    public_key=Config.LANGFUSE_PUBLIC_KEY,
                    secret_key=Config.LANGFUSE_SECRET_KEY,
                    host=Config.LANGFUSE_HOST
                )
            except Exception as e:
                print(f"❌ Langfuse init error: {e}")
                self.enabled = False
    
    def create_trace(self, test_id: str, stage: str) -> Optional[str]:
        """Tạo trace mới trong Langfuse"""
        if not self.enabled or not self.client:
            return None
        
        try:
            trace = self.client.trace(
                name=f"test_{test_id}",
                tags=[stage, "test"],
                metadata={
                    "test_id": test_id,
                    "stage": stage,
                }
            )
            return trace.id
        except Exception as e:
            print(f"⚠️  Failed to create trace: {e}")
            return None
    
    def log_turn_result(self, trace_id: str, turn_result: Dict[str, Any]):
        """Log turn result như span"""
        if not self.enabled or not self.client or not trace_id:
            return
        
        try:
            self.client.span(
                trace_id=trace_id,
                name=f"turn_{turn_result['turn_index']}",
                input={"content": turn_result["content"]},
                output={
                    "passed": turn_result["passed"],
                    "score": turn_result["score"],
                    "metrics": turn_result.get("metrics", {})
                },
                metadata={
                    "turn_index": turn_result["turn_index"],
                    "score": turn_result["score"]
                }
            )
        except Exception as e:
            print(f"⚠️  Failed to log turn: {e}")
    
    def log_metrics(self, trace_id: str, metrics: Dict[str, float]):
        """Log metrics như observations"""
        if not self.enabled or not self.client or not trace_id:
            return
        
        try:
            for metric_name, score in metrics.items():
                self.client.observation(
                    trace_id=trace_id,
                    name=f"metric_{metric_name}",
                    type="metric",
                    value=float(score),
                    metadata={"metric": metric_name}
                )
        except Exception as e:
            print(f"⚠️  Failed to log metrics: {e}")
    
    def flush(self):
        """Flush pending data"""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except:
                pass
