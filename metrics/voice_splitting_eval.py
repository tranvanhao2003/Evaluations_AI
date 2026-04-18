#!/usr/bin/env python3
"""
Voice Splitting Evaluators - Chia câu để tạo voice
"""
import json
from typing import Any
from Evaluation_AI.core.base_evaluator import (
    DeterministicEvaluator,
    LLMJudgeEvaluator,
    MetricScore,
)
from Evaluation_AI.config import Config


def _segment_lines(output_data: dict) -> str:
    segments = output_data.get("segments", []) if isinstance(output_data, dict) else []
    if not segments:
        return "(No segments provided)"
    return "\n".join(f"- {segment}" for segment in segments)


def _extract_source_script(input_data: Any) -> str:
    if isinstance(input_data, dict):
        return str(input_data.get("script_text", "")).strip()
    raw = str(input_data or "").strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return str(parsed.get("script_text", "")).strip()
    except Exception:
        pass
    return raw


def _extract_expected_segments(kwargs) -> str:
    expected_output = kwargs.get("expected_output")
    if isinstance(expected_output, dict):
        segments = expected_output.get("segments")
        if isinstance(segments, list) and segments:
            return "\n".join(f"- {str(segment).strip()}" for segment in segments if str(segment).strip())
    if isinstance(expected_output, list) and expected_output:
        return "\n".join(f"- {str(segment).strip()}" for segment in expected_output if str(segment).strip())
    return "(Không có segment kỳ vọng)"


class SemanticCompletenessEvaluator(LLMJudgeEvaluator):
    """Câu bị chia có đảm bảo đủ ý nghĩa không"""

    def __init__(self):
        super().__init__("semantic_completeness", "voice_splitting", Config.OPENAI_MODEL)

    def build_prompt(self, input_data: str, output_data: dict, **kwargs) -> str:
        source_script = _extract_source_script(input_data)
        expected_segments = _extract_expected_segments(kwargs)
        return f"""Bạn là reviewer chấm chất lượng chia đoạn để tạo voice từ script tiếng Việt.

                VĂN BẢN GỐC:
                {source_script}

                CÁC SEGMENT:
                {_segment_lines(output_data)}

                SEGMENT KỲ VỌNG/REFERENCE:
                {expected_segments}

                Chỉ chấm "semantic_completeness".
                Không chấm duration balance.
                Không chấm natural pause.

                Tiêu chí:
                1. Mỗi segment có giữ được ý nghĩa trọn vẹn ở mức cụm hoặc mệnh đề không.
                2. Phạt mạnh nếu bị cắt giữa cụm danh từ, cụm động từ, giới từ và bổ ngữ, số và đơn vị, hoặc tên riêng.
                3. Phạt vừa nếu segment vẫn hiểu được nhưng nghe cụt hoặc mất liên kết ngữ nghĩa.
                4. Ưu tiên segment mà khi đứng riêng vẫn đọc được tự nhiên.

                Rubric 1-5:
                1 = Nhiều segment bị cắt gãy ý nghiêm trọng.
                2 = Có nhiều segment khó hiểu hoặc chia sai cụm quan trọng.
                3 = Tạm ổn nhưng còn vài chỗ chia chưa trọn ý.
                4 = Đa số segment trọn ý, chỉ còn ít điểm chưa mượt.
                5 = Segment rất sạch, giữ tốt cấu trúc ngữ nghĩa.

                Trả về JSON:
                {{
                "score": 0,
                "reasoning": "...",
                "problem_segments": ["..."]
                }}"""

    def evaluate(self, input_data: str, output_data: dict, **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)

        if not output_data or not isinstance(output_data, dict):
            return MetricScore(
                self.metric_name, 0.0, False, threshold,
                "No valid output data", {}
            )

        try:
            prompt = self.build_prompt(input_data, output_data, **kwargs)
            result = self.call_llm(prompt)

            if not result or "error" in result:
                return MetricScore(
                    self.metric_name, 0.65, False, threshold,
                    f"LLM failed: {result.get('error')}", result
                )

            score_5 = float(result.get("score", 1.0))
            score_1 = (score_5 - 1) / 4 if score_5 > 0 else 0.0

            return MetricScore(
                self.metric_name, score_1, score_1 >= threshold, threshold,
                result.get("reasoning", ""), result
            )
        except Exception as e:
            return MetricScore(
                self.metric_name, 0.0, False, threshold,
                f"Evaluation error: {str(e)}", {}
            )


class DurationBalanceEvaluator(DeterministicEvaluator):
    """Các đoạn voice có độ dài sử dụng được và không quá lệch không"""

    def __init__(self):
        super().__init__("duration_balance", "voice_splitting")

    def evaluate(self, input_data: str, output_data: dict, **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)

        if not output_data or not isinstance(output_data, dict):
            return MetricScore(
                self.metric_name, 0.0, False, threshold,
                "No duration data available", {}
            )

        try:
            segments = output_data.get("segments", [])
            durations = output_data.get("durations", [])

            if not segments:
                return MetricScore(
                    self.metric_name, 0.0, False, threshold,
                    "No segments to analyze", {}
                )

            if durations and len(durations) == len(segments):
                values = [float(duration) for duration in durations if isinstance(duration, (int, float))]
                unit = "seconds"
                ideal_min = 1.2
                ideal_max = 6.0
            else:
                values = [len(str(segment).split()) for segment in segments]
                unit = "words"
                ideal_min = 4.0
                ideal_max = 16.0

            if not values:
                return MetricScore(
                    self.metric_name, 0.0, False, threshold,
                    "No usable duration values", {}
                )

            avg_value = sum(values) / len(values)
            variance = sum((value - avg_value) ** 2 for value in values) / len(values)
            std_dev = variance ** 0.5
            coefficient_variation = std_dev / max(avg_value, 0.1)

            usable_segments = 0
            for value in values:
                if ideal_min <= value <= ideal_max:
                    usable_segments += 1
            usable_ratio = usable_segments / len(values)

            balance_score = max(0.0, 1.0 - min(coefficient_variation, 1.0))
            score = (usable_ratio * 0.65) + (balance_score * 0.35)

            return MetricScore(
                self.metric_name, score, score >= threshold, threshold,
                (
                    f"Usable ratio={usable_ratio:.2f}, balance={balance_score:.2f}, "
                    f"avg={avg_value:.2f} {unit}"
                ),
                {
                    "avg_value": avg_value,
                    "std_dev": std_dev,
                    "usable_ratio": usable_ratio,
                    "unit": unit,
                }
            )
        except Exception as e:
            return MetricScore(
                self.metric_name, 0.0, False, threshold,
                f"Evaluation error: {str(e)}", {}
            )


class NaturalPauseEvaluator(LLMJudgeEvaluator):
    """Vị trí ngắt câu có tạo khoảng nghỉ tự nhiên"""

    def __init__(self):
        super().__init__("natural_pause", "voice_splitting", Config.OPENAI_MODEL)

    def build_prompt(self, input_data: str, output_data: dict, **kwargs) -> str:
        source_script = _extract_source_script(input_data)
        expected_segments = _extract_expected_segments(kwargs)
        return f"""Bạn là reviewer chấm natural pause cho hệ thống chia segment voice.

                VĂN BẢN GỐC:
                {source_script}

                CÁC SEGMENT:
                {_segment_lines(output_data)}

                SEGMENT KỲ VỌNG/REFERENCE:
                {expected_segments}

                Chỉ chấm "natural_pause".
                Không chấm semantic completeness.
                Không chấm duration balance.

                Tiêu chí:
                1. Điểm ngắt có rơi vào vị trí người nói tự nhiên có thể nghỉ hơi không.
                2. Ưu tiên ngắt sau dấu câu, sau mệnh đề, hoặc sau ý hoàn chỉnh.
                3. Phạt mạnh nếu ngắt ở vị trí làm nhịp đọc bị khựng, ví dụ giữa tên riêng, giữa con số và đơn vị, hoặc giữa từ nối và vế sau.
                4. Xem xét độ trôi chảy nếu đọc tuần tự các segment.

                Rubric 1-5:
                1 = Nhiều điểm ngắt rất gượng, nghe khó chịu.
                2 = Có nhiều điểm ngắt không tự nhiên.
                3 = Tạm chấp nhận, còn một số chỗ khựng.
                4 = Tự nhiên ở đa số vị trí.
                5 = Rất tự nhiên, gần với nhịp nghỉ của người đọc thật.

                Trả về JSON:
                {{
                "score": 0,
                "reasoning": "...",
                "awkward_breaks": ["..."]
                }}"""

    def evaluate(self, input_data: str, output_data: dict, **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)

        if not output_data or not isinstance(output_data, dict):
            return MetricScore(
                self.metric_name, 0.0, False, threshold,
                "No pause data available", {}
            )

        try:
            prompt = self.build_prompt(input_data, output_data, **kwargs)
            result = self.call_llm(prompt)

            if not result or "error" in result:
                return MetricScore(
                    self.metric_name, 0.65, False, threshold,
                    f"LLM failed: {result.get('error')}", result
                )

            score_5 = float(result.get("score", 1.0))
            score_1 = (score_5 - 1) / 4 if score_5 > 0 else 0.0

            return MetricScore(
                self.metric_name, score_1, score_1 >= threshold, threshold,
                result.get("reasoning", ""), result
            )
        except Exception as e:
            return MetricScore(
                self.metric_name, 0.0, False, threshold,
                f"Evaluation error: {str(e)}", {}
            )
