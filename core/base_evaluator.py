from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
import json


def _clamp_01(value: float) -> float:
    """Clamp numeric score to [0, 1]."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, numeric))

class EvaluationType(str, Enum):
    """Loại evaluator"""
    DETERMINISTIC = "deterministic"
    LLM_JUDGE = "llm_judge"

@dataclass
class MetricScore:
    """Kết quả đánh giá 1 metric"""
    metric_name: str
    score: float  # 0.0 - 1.0
    passed: bool
    threshold: float = 0.7
    reasoning: str = ""
    raw_data: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Enforce normalized score/threshold range globally.
        self.score = _clamp_01(self.score)
        self.threshold = _clamp_01(self.threshold)
    
    def to_dict(self):
        return asdict(self)

@dataclass
class TurnEvalResult:
    """Kết quả đánh giá 1 turn"""
    turn_index: int
    content: str
    passed: bool = True
    checks: Dict[str, bool] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    error: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class TestCaseResult:
    """Kết quả đánh giá 1 test case"""
    test_case_id: str
    test_case_name: str
    stage: str
    passed: bool = False
    overall_score: float = 0.0
    turn_results: List[TurnEvalResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    langfuse_trace_id: Optional[str] = None
    error: Optional[str] = None
    
    def add_turn(self, turn_result: TurnEvalResult):
        """Thêm kết quả turn"""
        self.turn_results.append(turn_result)
    
    def finalize(self):
        """Tính toán kết quả cuối cùng"""
        if not self.turn_results:
            self.passed = False
            self.overall_score = 0.0
            return
        
        self.passed = all(t.passed for t in self.turn_results)
        self.overall_score = _clamp_01(
            sum(t.score for t in self.turn_results) / len(self.turn_results)
        )
    
    def to_dict(self):
        return {
            "test_case_id": self.test_case_id,
            "test_case_name": self.test_case_name,
            "stage": self.stage,
            "passed": self.passed,
            "overall_score": self.overall_score,
            "num_turns": len(self.turn_results),
            "turn_results": [t.to_dict() for t in self.turn_results],
            "timestamp": self.timestamp,
            "langfuse_trace_id": self.langfuse_trace_id,
            "error": self.error
        }

class BaseEvaluator(ABC):
    """Base class cho tất cả evaluators"""
    
    def __init__(self, metric_name: str, stage: str, eval_type: EvaluationType):
        self.metric_name = metric_name
        self.stage = stage
        self.eval_type = eval_type
    
    @abstractmethod
    def evaluate(self, input_data: Any, output_data: Any, **kwargs) -> MetricScore:
        """Chạy đánh giá"""
        pass

class DeterministicEvaluator(BaseEvaluator):
    """Evaluator dựa vào code logic"""
    
    def __init__(self, metric_name: str, stage: str):
        super().__init__(metric_name, stage, EvaluationType.DETERMINISTIC)

class LLMJudgeEvaluator(BaseEvaluator):
    """Evaluator sử dụng LLM-as-a-Judge"""
    
    def __init__(self, metric_name: str, stage: str, model: str = "gpt-4o-mini"):
        super().__init__(metric_name, stage, EvaluationType.LLM_JUDGE)
        self.model = model
    
    @abstractmethod
    def build_prompt(self, input_data: Any, output_data: Any, **kwargs) -> str:
        """Build prompt cho LLM"""
        pass
    
    def call_llm(self, prompt: str) -> Dict[str, Any]:
        """Gọi LLM"""
        from Evaluation_AI.config import Config
        import signal
        
        try:
            from openai import OpenAI
            
            # Create a timeout handler
            def timeout_handler(signum, frame):
                raise TimeoutError("LLM call timeout")
            
            # Set a 15-second timeout for each judge call
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(15)
            
            try:
                client = OpenAI(api_key=Config.OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                    timeout=15
                )
                
                signal.alarm(0)  # Cancel alarm
                
                content = response.choices[0].message.content
                if not content:
                    return {"error": "Empty response from LLM", "score": 0.6, "average_score_1_to_5": 3.0}
                
                result = json.loads(content)
                if result is None:
                    return {"error": "JSON loads returned None", "score": 0.6, "average_score_1_to_5": 3.0}
                
                return result
            except (TimeoutError, Exception) as e:
                signal.alarm(0)  # Cancel alarm
                return {
                    "error": str(e), 
                    "score": 0.6, 
                    "average_score_1_to_5": 3.0,
                    "accuracy_score_1_to_5": 3.0,
                    "average_visual_score_1_to_5": 3.0
                }
        except Exception as e:
            return {"error": str(e), "score": 0.6, "average_score_1_to_5": 3.0}
