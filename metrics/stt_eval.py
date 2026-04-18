"""
STT (Speech-to-Text) Evaluators
"""
import re
from Evaluation_AI.core.base_evaluator import DeterministicEvaluator, LLMJudgeEvaluator, MetricScore
from Evaluation_AI.config import Config


def _extract_transcript(output_data):
    if isinstance(output_data, dict):
        for key in ("transcript", "text", "output", "content"):
            val = output_data.get(key)
            if isinstance(val, str):
                return val
        return ""
    return str(output_data or "")


def _resolve_reference_text(input_data, kwargs) -> str:
    expected_output = kwargs.get("expected_output")
    if isinstance(expected_output, dict):
        for key in ("transcript", "text", "output", "content", "full_text"):
            value = expected_output.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(expected_output, str) and expected_output.strip():
        return expected_output.strip()

    context = kwargs.get("context", {}) or {}
    generated_script = context.get("generated_script")
    if isinstance(generated_script, str) and generated_script.strip():
        return generated_script.strip()

    return str(input_data or "").strip()


def _extract_word_timestamps(output_data):
    if isinstance(output_data, dict):
        timestamps = output_data.get("word_timestamps", [])
        if isinstance(timestamps, list):
            return timestamps
    return []


def _normalize_for_wer(text: str) -> str:
    normalized = re.sub(r"[^\w\s]", " ", str(text or "").lower(), flags=re.UNICODE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _extract_timestamp_word(ts) -> str:
    if not isinstance(ts, dict):
        return ""
    for key in ("word", "text", "token", "content"):
        value = ts.get(key)
        if isinstance(value, str):
            return value.strip()
    return ""


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) using Levenshtein distance"""
    ref_words = _normalize_for_wer(reference).split()
    hyp_words = _normalize_for_wer(hypothesis).split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dist = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(1, rows):
        dist[i][0] = i
    for j in range(1, cols):
        dist[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dist[i][j] = min(
                dist[i - 1][j] + 1,
                dist[i][j - 1] + 1,
                dist[i - 1][j - 1] + cost,
            )

    return dist[len(ref_words)][len(hyp_words)] / len(ref_words)


class STTPunctuationEvaluator(LLMJudgeEvaluator):
    """Đánh giá punctuation/capitalization restoration dựa trên transcript và reference"""

    def __init__(self):
        super().__init__("punctuation_capitalization", "stt_transcription", Config.OPENAI_MODEL)

    def build_prompt(self, reference_text: str, transcript: str, **kwargs) -> str:
        return f"""Bạn là reviewer chấm chất lượng định dạng đầu ra của hệ thống STT.

                REFERENCE TEXT:
                {reference_text}

                STT OUTPUT:
                {transcript}

                Chỉ chấm khả năng khôi phục dấu câu, viết hoa và tách câu.
                Không chấm lỗi nội dung từ vựng nếu lỗi đó không ảnh hưởng đến punctuation/capitalization.

                Tiêu chí:
                1. Vị trí ngắt câu và dấu câu có hợp lý so với reference không.
                2. Viết hoa đầu câu, tên riêng, acronym hoặc thuật ngữ quan trọng có hợp lý không.
                3. Transcript có quá dồn cục hoặc thiếu dấu ngắt khiến khó đọc không.

                Rubric 1-5:
                1 = Gần như không có punctuation/capitalization đúng.
                2 = Có phục hồi một phần nhưng lỗi còn nhiều, khó đọc.
                3 = Đọc được nhưng còn thiếu khá nhiều dấu/ngắt hoặc viết hoa sai.
                4 = Tốt, chỉ còn vài lỗi nhỏ.
                5 = Rất tốt, gần như đầy đủ và tự nhiên.

                Trả về JSON:
                {{
                "score": 0,
                "reasoning": "...",
                "issues": ["..."]
                }}"""

    def evaluate(self, input_data: str, output_data: str, **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)
        transcript = _extract_transcript(output_data)
        if not transcript:
            return MetricScore(
                self.metric_name,
                0.0,
                False,
                threshold,
                "Missing transcript for punctuation evaluation"
            )

        ground_truth = _resolve_reference_text(input_data, kwargs)
        prompt = self.build_prompt(str(ground_truth), transcript, **kwargs)
        result = self.call_llm(prompt)

        if not result or "error" in result:
            return MetricScore(self.metric_name, 0.65, False, threshold, f"LLM failed: {result.get('error')}")

        score_5 = float(result.get("score", 1.0))
        score_1 = (score_5 - 1) / 4 if score_5 > 0 else 0.0

        return MetricScore(
            metric_name=self.metric_name,
            score=score_1,
            passed=score_1 >= threshold,
            threshold=threshold,
            reasoning=result.get("reasoning", ""),
            raw_data=result
        )


class STTAccuracyEvaluator(DeterministicEvaluator):
    """Kiểm tra độ chính xác STT (sử dụng WER chuẩn hóa)"""

    def __init__(self):
        super().__init__("word_error_rate", "stt_transcription")

    def evaluate(self, input_data: str, output_data: str, **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)

        ground_truth = _resolve_reference_text(input_data, kwargs)
        hypothesis = _extract_transcript(output_data)

        if not ground_truth or not hypothesis:
            return MetricScore(
                metric_name=self.metric_name,
                score=0.0,
                passed=False,
                threshold=threshold,
                reasoning="Missing reference or hypothesis for WER calculation"
            )

        wer = calculate_wer(str(ground_truth), hypothesis)
        score = max(0.0, 1.0 - wer)

        return MetricScore(
            metric_name=self.metric_name,
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            reasoning=f"Word Error Rate: {wer:.2%} (Score: {score:.2f})",
            raw_data={"wer": wer, "score": score}
        )


class STTTimestampEvaluator(DeterministicEvaluator):
    """Đánh giá độ nhất quán cấu trúc timestamp theo từ"""

    def __init__(self):
        super().__init__("timestamp_accuracy", "stt_transcription")

    def evaluate(self, input_data: str, output_data: str, **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)
        timestamps = _extract_word_timestamps(output_data)

        if not timestamps:
            return MetricScore(
                metric_name=self.metric_name,
                score=0.0,
                passed=False,
                threshold=threshold,
                reasoning="No word timestamps found in output",
                raw_data={"num_timestamps": 0}
            )

        transcript = _extract_transcript(output_data)
        transcript_words = _normalize_for_wer(transcript).split()

        valid_count = 0
        non_overlap_count = 0
        monotonic_count = 0
        duration_reasonable_count = 0
        labeled_word_count = 0
        matched_word_count = 0
        prev_end = -1.0

        for idx, ts in enumerate(timestamps):
            if not isinstance(ts, dict):
                continue
            start = ts.get("start")
            end = ts.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            duration = end - start
            if end > start and start >= 0:
                valid_count += 1
            if start >= prev_end:
                non_overlap_count += 1
            if end >= start:
                monotonic_count += 1
            if 0.05 <= duration <= 1.5:
                duration_reasonable_count += 1

            ts_word = _normalize_for_wer(_extract_timestamp_word(ts))
            if ts_word:
                labeled_word_count += 1
                if idx < len(transcript_words) and transcript_words[idx] == ts_word:
                    matched_word_count += 1
            prev_end = end

        total = max(1, len(timestamps))
        valid_ratio = valid_count / total
        non_overlap_ratio = non_overlap_count / total
        monotonic_ratio = monotonic_count / total
        duration_reasonable_ratio = duration_reasonable_count / total

        if labeled_word_count > 0:
            word_alignment_ratio = matched_word_count / labeled_word_count
        else:
            word_alignment_ratio = min(1.0, total / max(1, len(transcript_words))) if transcript_words else 0.5

        score = (
            (valid_ratio * 0.3) +
            (non_overlap_ratio * 0.2) +
            (monotonic_ratio * 0.15) +
            (duration_reasonable_ratio * 0.15) +
            (word_alignment_ratio * 0.2)
        )

        return MetricScore(
            metric_name=self.metric_name,
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            reasoning=(
                "Timestamp consistency "
                f"valid={valid_ratio:.2f}, non_overlap={non_overlap_ratio:.2f}, "
                f"monotonic={monotonic_ratio:.2f}, duration={duration_reasonable_ratio:.2f}, "
                f"word_alignment={word_alignment_ratio:.2f}"
            ),
            raw_data={
                "num_timestamps": len(timestamps),
                "valid_ratio": valid_ratio,
                "non_overlap_ratio": non_overlap_ratio,
                "monotonic_ratio": monotonic_ratio,
                "duration_reasonable_ratio": duration_reasonable_ratio,
                "word_alignment_ratio": word_alignment_ratio,
            }
        )
