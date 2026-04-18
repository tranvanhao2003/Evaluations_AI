"""
Subtitle Splitting Evaluators
"""
import json
from typing import List, Dict, Any
from Evaluation_AI.core.base_evaluator import (
    DeterministicEvaluator,
    MetricScore,
)
from Evaluation_AI.config import Config


BAD_LINE_STARTS = {",", ".", ";", ":", "!", "?", ")", "]"}
WEAK_ENDING_WORDS = {
    "va", "voi", "cua", "la", "de", "tai", "cho", "tu", "den",
    "nhung", "ma", "neu", "khi", "do", "thi", "se", "da",
}
WEAK_STARTING_WORDS = {
    "va", "voi", "cua", "la", "de", "tai", "cho", "tu", "den",
    "nhung", "ma", "neu", "khi", "thi", "hay", "hoac",
}


def _normalize_word(word: str) -> str:
    return "".join(ch.lower() for ch in str(word or "") if ch.isalnum())


def _extract_source_text(input_data: Any) -> str:
    if isinstance(input_data, dict):
        return str(input_data.get("text", "")).strip()
    raw = str(input_data or "").strip()
    if not raw:
        return ""
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return str(parsed.get("text", "")).strip()
    except Exception:
        pass
    return raw


def _extract_reference_subtitle_text(input_data: Any, kwargs) -> str:
    expected_output = kwargs.get("expected_output")
    if isinstance(expected_output, list):
        parts = []
        for item in expected_output:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(item, str) and item.strip():
                parts.append(item.strip())
        if parts:
            return " ".join(parts).strip()
    if isinstance(expected_output, dict):
        text = expected_output.get("text")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return _extract_source_text(input_data)


class SubtitleCPLEvaluator(DeterministicEvaluator):
    """Đánh giá độ dễ đọc: chars/line, cps, số dòng"""

    def __init__(self):
        super().__init__("readability", "subtitle_splitting")

    def evaluate(self, input_data: Any, output_data: List[Dict], **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)

        if not output_data:
            return MetricScore(self.metric_name, 0.0, False, threshold, "No subtitles")

        total_lines = 0
        line_length_scores = []
        cps_scores = []
        layout_scores = []
        for caption in output_data:
            if isinstance(caption, dict):
                text = str(caption.get("text", ""))
                start = caption.get("start")
                end = caption.get("end")
            else:
                text = str(caption)
                start = None
                end = None

            lines = [line.strip() for line in text.split("\n") if line.strip()] or [text.strip()]
            layout_scores.append(1.0 if len(lines) <= 2 else max(0.0, 1.0 - ((len(lines) - 2) * 0.4)))
            for line in lines:
                total_lines += 1
                char_count = len(line)
                if 12 <= char_count <= 42:
                    line_score = 1.0
                elif 8 <= char_count <= 48:
                    line_score = 0.8
                else:
                    distance = min(abs(char_count - 12), abs(char_count - 42))
                    line_score = max(0.0, 1.0 - (distance / 30.0))
                line_length_scores.append(line_score)

                if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end > start:
                    duration = end - start
                    cps = len(line) / max(duration, 0.1)
                    if cps <= 17:
                        cps_score = 1.0
                    elif cps <= 20:
                        cps_score = 0.8
                    else:
                        cps_score = max(0.0, 1.0 - ((cps - 20) / 12))
                else:
                    cps_score = 0.75
                cps_scores.append(cps_score)

        if total_lines == 0:
            return MetricScore(self.metric_name, 0.0, False, threshold, "No valid subtitle lines")

        score = (
            (sum(line_length_scores) / total_lines) * 0.45 +
            (sum(cps_scores) / total_lines) * 0.35 +
            (sum(layout_scores) / len(layout_scores)) * 0.2
        )
        return MetricScore(
            metric_name=self.metric_name,
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            reasoning=f"Subtitle readability from {total_lines} lines",
            raw_data={"num_lines": total_lines}
        )


class SubtitleOrphanWordsEvaluator(DeterministicEvaluator):
    """Đánh giá logic ngắt dòng: tránh orphan và ngắt xấu"""

    def __init__(self):
        super().__init__("line_break_logic", "subtitle_splitting")

    def evaluate(self, input_data: Any, output_data: List[Dict], **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)

        if not output_data:
            return MetricScore(self.metric_name, 0.0, False, threshold, "No subtitles")

        total_break_checks = 0
        penalties = 0
        for caption in output_data:
            text = caption.get("text", "") if isinstance(caption, dict) else str(caption)
            lines = [line.strip() for line in str(text).split("\n") if line.strip()] or [str(text).strip()]

            for line in lines:
                words = line.split()
                if len(words) <= 1:
                    penalties += 1
                if line and line[0] in BAD_LINE_STARTS:
                    penalties += 1

            for idx in range(len(lines) - 1):
                total_break_checks += 1
                left_words = lines[idx].split()
                right_words = lines[idx + 1].split()
                left_last = _normalize_word(left_words[-1]) if left_words else ""
                right_first = _normalize_word(right_words[0]) if right_words else ""

                if left_last in WEAK_ENDING_WORDS:
                    penalties += 1
                if right_first in WEAK_STARTING_WORDS:
                    penalties += 1
                if right_words and len(right_words) == 1:
                    penalties += 1

        total = max(1, total_break_checks + sum(1 for caption in output_data))
        score = max(0.0, 1.0 - (penalties / total))
        return MetricScore(
            metric_name=self.metric_name,
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            reasoning=f"Line break penalties: {penalties} over {total} checks",
            raw_data={"penalties": penalties, "checks": total}
        )


class SubtitleSyncEvaluator(DeterministicEvaluator):
    """Đánh giá đồng bộ subtitle theo timestamp và coverage"""

    def __init__(self):
        super().__init__("synchronization", "subtitle_splitting")

    def evaluate(self, input_data: List[Dict], output_data: List[Dict], **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)
        if not output_data:
            return MetricScore(self.metric_name, 0.0, False, threshold, "No subtitles")

        valid = 0
        non_overlap = 0
        monotonic = 0
        duration_reasonable = 0
        total = 0
        prev_end = -1.0

        source_text = _extract_reference_subtitle_text(input_data, kwargs)
        source_word_count = len(source_text.split())
        subtitle_word_count = 0

        for caption in output_data:
            if not isinstance(caption, dict):
                continue
            start = caption.get("start")
            end = caption.get("end")
            text = str(caption.get("text", "")).strip()
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            total += 1
            subtitle_word_count += len(text.replace("\n", " ").split())
            duration = end - start
            if end > start and start >= 0:
                valid += 1
            if start >= prev_end:
                non_overlap += 1
            if end >= start:
                monotonic += 1
            if 0.6 <= duration <= 7.0:
                duration_reasonable += 1
            prev_end = end

        if total == 0:
            return MetricScore(self.metric_name, 0.0, False, threshold, "No usable timestamps")

        coverage_ratio = 1.0
        if source_word_count > 0:
            coverage_ratio = min(subtitle_word_count, source_word_count) / source_word_count

        score = (
            ((valid / total) * 0.25) +
            ((non_overlap / total) * 0.2) +
            ((monotonic / total) * 0.15) +
            ((duration_reasonable / total) * 0.2) +
            (coverage_ratio * 0.2)
        )
        return MetricScore(
            self.metric_name,
            score,
            score >= threshold,
            threshold,
            (
                f"Sync structure valid={valid}/{total}, non_overlap={non_overlap}/{total}, "
                f"duration_reasonable={duration_reasonable}/{total}, coverage={coverage_ratio:.2f}"
            ),
            {"total": total, "coverage_ratio": coverage_ratio}
        )
