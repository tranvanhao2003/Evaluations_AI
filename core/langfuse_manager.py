"""
Langfuse Integration Manager - Datasets, Experiments, Metrics
"""
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from Evaluation_AI.config import Config

class LangfuseDatasetManager:
    """Quản lý Langfuse Datasets"""
    
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
    
    def create_or_get_dataset(self, dataset_name: str, description: str = "") -> Optional[str]:
        """Tạo hoặc lấy dataset từ Langfuse"""
        if not self.enabled or not self.client:
            return None
        
        try:
            # Try to get existing dataset
            try:
                dataset = self.client.get_dataset(dataset_name)
                print(f"✅ Dataset '{dataset_name}' đã tồn tại")
                return dataset.id
            except:
                # Create new dataset
                dataset = self.client.create_dataset(
                    name=dataset_name,
                    description=description
                )
                print(f"✅ Created dataset '{dataset_name}'")
                return dataset.id
        except Exception as e:
            print(f"⚠️  Error: {e}")
            return None
    
    def add_dataset_item(self, dataset_name: str, test_case: Dict[str, Any]) -> Optional[str]:
        """Thêm item vào dataset"""
        if not self.enabled or not self.client:
            return None
        
        try:
            item = self.client.create_dataset_item(
                dataset_name=dataset_name,
                input={
                    "id": test_case.get("id"),
                    "name": test_case.get("name"),
                    "stage": test_case.get("stage"),
                    "content": test_case.get("turns", [{}])[0].get("content", "")
                },
                expected_output={
                    "criteria": test_case.get("criteria", [])
                },
                metadata={
                    "test_case_id": test_case.get("id"),
                    "category": test_case.get("metadata", {}).get("category", ""),
                    "difficulty": test_case.get("metadata", {}).get("difficulty", "")
                }
            )
            return item.id
        except Exception as e:
            print(f"⚠️  Error adding item: {e}")
            return None
    
    def get_dataset_items(self, dataset_name: str) -> List[Dict]:
        """Lấy tất cả items từ dataset"""
        if not self.enabled or not self.client:
            return []
        
        try:
            dataset = self.client.get_dataset(dataset_name)
            return dataset.items if hasattr(dataset, 'items') else []
        except Exception as e:
            print(f"⚠️  Error: {e}")
            return []

class LangfuseExperimentManager:
    """Quản lý Langfuse Experiments - Chạy và so sánh evaluations"""
    
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
    
    def create_trace(self, test_id: str, stage: str, dataset_item_id: Optional[str] = None) -> Optional[str]:
        """Tạo trace cho 1 test"""
        if not self.enabled or not self.client:
            return None
        
        try:
            # Create a simple event that will be used as trace identifier
            trace_id = f"{stage}_{test_id}_{int(datetime.now().timestamp() * 1000)}"
            
            # Create event as simple log entry
            self.client.create_event(
                name=f"test_{test_id}",
                input={"test_id": test_id, "stage": stage},
                metadata={
                    "test_id": test_id,
                    "stage": stage,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            
            return trace_id
        except Exception as e:
            # Langfuse API error - continue without it
            print(f"⚠️  Error creating trace for {test_id}: {e}")
            return None
    
    def log_metrics(self, trace_id: str, metrics: Dict[str, float], stage: str):
        """Log metrics vào Langfuse"""
        if not self.enabled or not self.client or not trace_id:
            return
        
        # Disabled - create_score causing bad request errors
        # Just skip metric logging for now
        pass
    
    def log_turn(self, trace_id: str, turn_index: int, turn_data: Dict[str, Any]):
        """Log 1 turn như event"""
        if not self.enabled or not self.client or not trace_id:
            return
        
        # Disabled - create_event causing issues
        pass
    
    def end_experiment(self, trace_id: str, test_passed: bool, overall_score: float):
        """Kết thúc experiment"""
        if not self.enabled or not self.client or not trace_id:
            return
        
        # Disabled - create_event causing issues  
        pass
    
    def link_result_to_item(self, dataset_item_id: str, test_passed: bool, overall_score: float, result_data: Dict[str, Any]):
        """Link test result to dataset item via HTTP API"""
        if not self.enabled or not self.client or not dataset_item_id:
            return
        
        try:
            # Use HTTP API to directly update dataset item observation
            # This will register the test as an "experiment" run
            response = self.client.api.post(
                f"/dataset-items/{dataset_item_id}/observations",
                {
                    "output": {
                        "passed": test_passed,
                        "overall_score": overall_score,
                        "status": "passed" if test_passed else "failed"
                    },
                    "metadata": {
                        "test_passed": test_passed,
                        "overall_score": overall_score
                    }
                }
            )
            return response
        except Exception as e:
            pass  # Silent fail
    
    def flush(self):
        """Flush all pending data"""
        if self.enabled and self.client:
            try:
                self.client.flush()
            except Exception as e:
                pass  # Silent fail on flush

class LangfuseMetricsAggregator:
    """Tính toán và aggregate metrics từ experiments"""
    
    @staticmethod
    def aggregate_stage_metrics(traces: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics cho 1 stage"""
        if not traces:
            return {}
        
        all_metrics = {}
        stage_results = {}
        
        for trace in traces:
            metrics = trace.get("metrics", {})
            for metric_name, value in metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Calculate statistics
        for metric_name, values in all_metrics.items():
            stage_results[metric_name] = {
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values),
                "threshold": Config.get_threshold(metric_name)
            }
        
        return stage_results
    
    @staticmethod
    def compare_experiments(exp1_results: Dict, exp2_results: Dict) -> Dict[str, Any]:
        """So sánh 2 experiments"""
        comparison = {}
        
        all_metrics = set(exp1_results.keys()) | set(exp2_results.keys())
        
        for metric in all_metrics:
            exp1_score = exp1_results.get(metric, {}).get("avg", 0)
            exp2_score = exp2_results.get(metric, {}).get("avg", 0)
            
            improvement = exp2_score - exp1_score
            improvement_pct = (improvement / exp1_score * 100) if exp1_score > 0 else 0
            
            comparison[metric] = {
                "exp1": exp1_score,
                "exp2": exp2_score,
                "improvement": improvement,
                "improvement_pct": improvement_pct,
                "winner": "exp2" if improvement > 0 else "exp1"
            }
        
        return comparison

class LangfuseReportGenerator:
    """Generate reports từ Langfuse data"""
    
    @staticmethod
    def generate_stage_report(stage: str, metrics: Dict[str, Any]) -> str:
        """Generate text report cho 1 stage"""
        report = f"\n{'='*60}\n"
        report += f"📊 STAGE: {stage.upper()}\n"
        report += f"{'='*60}\n\n"
        
        for metric_name, stats in metrics.items():
            threshold = stats.get("threshold", 0.7)
            avg_score = stats.get("avg", 0)
            status = "✅" if avg_score >= threshold else "❌"
            
            report += f"{status} {metric_name}\n"
            report += f"   Avg:  {avg_score:.2%}\n"
            report += f"   Min:  {stats.get('min', 0):.2%}\n"
            report += f"   Max:  {stats.get('max', 0):.2%}\n"
            report += f"   Threshold: {threshold:.2%}\n"
            report += f"   Tests: {stats.get('count', 0)}\n\n"
        
        return report
    
    @staticmethod
    def generate_comparison_report(comparison: Dict[str, Any]) -> str:
        """Generate comparison report"""
        report = f"\n{'='*60}\n"
        report += f"📊 EXPERIMENT COMPARISON\n"
        report += f"{'='*60}\n\n"
        
        for metric, comp in comparison.items():
            winner = "Exp2 ✅" if comp["winner"] == "exp2" else "Exp1 ✅"
            improvement = comp["improvement_pct"]
            
            report += f"{metric}\n"
            report += f"   Exp1: {comp['exp1']:.2%}\n"
            report += f"   Exp2: {comp['exp2']:.2%}\n"
            report += f"   Change: {improvement:+.1f}%\n"
            report += f"   Winner: {winner}\n\n"
        
        return report
