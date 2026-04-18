"""
Script Generation Evaluators
"""
import json
import re
from Evaluation_AI.core.base_evaluator import (
    DeterministicEvaluator,
    LLMJudgeEvaluator,
    MetricScore,
)
from Evaluation_AI.config import Config


def _stringify_script_output(output_data) -> str:
    if isinstance(output_data, dict):
        full_script = str(output_data.get("full_script", "")).strip()
        if full_script:
            return full_script
        parts = [
            str(output_data.get(key, "")).strip()
            for key in ("hook", "body", "cta")
            if str(output_data.get(key, "")).strip()
        ]
        if parts:
            return " ".join(parts).strip()
        return json.dumps(output_data, ensure_ascii=False)
    return str(output_data or "").strip()


def _build_script_context(kwargs) -> str:
    test_case = kwargs.get("test_case")
    metadata = getattr(test_case, "metadata", None) if test_case else None
    if not isinstance(metadata, dict):
        return "Không có metadata bổ sung."

    field_labels = {
        "industry": "Ngành",
        "target_audience": "Ứng viên mục tiêu",
        "job_level": "Cấp độ vị trí",
        "hook_type": "Kiểu hook mong muốn",
        "category": "Loại template",
        "difficulty": "Độ khó",
    }
    lines = []
    for key, label in field_labels.items():
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            lines.append(f"- {label}: {value.strip()}")
    return "\n".join(lines) if lines else "Không có metadata bổ sung."


def _build_expected_output_context(kwargs) -> str:
    expected_output = kwargs.get("expected_output")
    if expected_output is None:
        return "Không có script kỳ vọng."
    expected_text = _stringify_script_output(expected_output)
    if not expected_text:
        return "Không có script kỳ vọng."
    return expected_text


class ScriptStructureEvaluator(DeterministicEvaluator):
    """Kiểm tra JSON format & Hook-Body-CTA structure"""

    def __init__(self):
        super().__init__("structure", "script_generation")

    def evaluate(self, input_data: str, output_data: str, **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)

        try:
            if isinstance(output_data, str):
                try:
                    parsed = json.loads(output_data)
                except Exception:
                    parsed = None
            else:
                parsed = output_data

            cta_keywords = [
                "ung tuyen", "inbox", "nop ho so", "dang ky",
                "gui cv", "lien he", "apply", "ung tuyen ngay",
                "nop don", "tham gia", "join", "send cv",
            ]
            hook_markers = [
                "ban co", "ban da", "ban muon", "dung bo lo", "co hoi",
                "tim kiem", "san sang", "neu ban", "hay tuong tuong",
            ]

            def _score_text_structure(text: str):
                normalized_text = str(text or "").strip()
                sentences = [
                    segment.strip()
                    for segment in re.split(r"[.!?]+", normalized_text)
                    if segment.strip()
                ]
                words = normalized_text.split()
                first_sentence = sentences[0] if sentences else ""
                last_sentence = sentences[-1] if sentences else normalized_text
                text_norm = normalized_text.lower()
                has_hook_local = len(first_sentence.split()) >= 3
                has_body_local = len(words) >= 20 and len(sentences) >= 2
                has_cta_local = any(keyword in text_norm for keyword in cta_keywords)
                hook_attention_local = ("?" in normalized_text[:120]) or ("!" in normalized_text[:120]) or any(
                    marker in first_sentence.lower() for marker in hook_markers
                )
                cta_actionable_local = any(keyword in last_sentence.lower() for keyword in cta_keywords)
                return (
                    has_hook_local,
                    has_body_local,
                    has_cta_local,
                    hook_attention_local,
                    cta_actionable_local,
                    {"raw_text": normalized_text},
                )

            if isinstance(parsed, dict):
                hook_text = str(parsed.get("hook", "")).strip()
                body_text = str(parsed.get("body", "")).strip()
                cta_text = str(parsed.get("cta", "")).strip()
                if hook_text or body_text or cta_text:
                    has_hook = bool(hook_text)
                    has_body = bool(body_text)
                    has_cta = bool(cta_text)
                    hook_attention = ("?" in hook_text) or ("!" in hook_text) or any(
                        marker in hook_text.lower() for marker in hook_markers
                    )
                    cta_actionable = any(keyword in cta_text.lower() for keyword in cta_keywords)
                    evidence = parsed
                else:
                    full_script = _stringify_script_output(parsed)
                    (
                        has_hook,
                        has_body,
                        has_cta,
                        hook_attention,
                        cta_actionable,
                        evidence,
                    ) = _score_text_structure(full_script)
            else:
                (
                    has_hook,
                    has_body,
                    has_cta,
                    hook_attention,
                    cta_actionable,
                    evidence,
                ) = _score_text_structure(output_data)

            score = (
                (0.25 * float(has_hook)) +
                (0.35 * float(has_body)) +
                (0.2 * float(has_cta)) +
                (0.1 * float(hook_attention)) +
                (0.1 * float(cta_actionable))
            )

            return MetricScore(
                metric_name=self.metric_name,
                score=score,
                passed=score >= threshold,
                threshold=threshold,
                reasoning=(
                    "Structure check "
                    f"hook/body/cta={has_hook}/{has_body}/{has_cta}, "
                    f"hook_attention={hook_attention}, cta_actionable={cta_actionable}"
                ),
                raw_data=evidence
            )
        except Exception:
            return MetricScore(self.metric_name, 0.0, False, threshold, "Invalid format")


class ScriptWordCountEvaluator(DeterministicEvaluator):
    """Kiểm tra word count phù hợp"""

    def __init__(self):
        super().__init__("length_constraint", "script_generation")

    def evaluate(self, input_data: str, output_data: str, target_duration: int = 60, **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)

        if not output_data:
            return MetricScore(self.metric_name, 0.0, False, threshold, "No output")

        text = _stringify_script_output(output_data)
        word_count = len(text.split())

        test_case = kwargs.get("test_case")
        if test_case and isinstance(getattr(test_case, "metadata", None), dict):
            duration_meta = test_case.metadata.get("video_duration")
            if isinstance(duration_meta, (int, float)) and duration_meta > 0:
                target_duration = int(duration_meta)

        ideal_min = max(25, int(target_duration * 2.0))
        ideal_max = max(35, int(target_duration * 3.0))

        if ideal_min <= word_count <= ideal_max:
            score = 1.0
        else:
            dist = min(abs(word_count - ideal_min), abs(word_count - ideal_max))
            score = max(0.0, 1.0 - (dist / ideal_max))

        return MetricScore(
            metric_name=self.metric_name,
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            reasoning=f"Words: {word_count} (Ideal: {ideal_min}-{ideal_max})",
            raw_data={"count": word_count}
        )


class ScriptRelevanceEvaluator(LLMJudgeEvaluator):
    """Đánh giá độ liên quan: Kịch bản có bám sát JD không"""

    def __init__(self):
        super().__init__("relevance", "script_generation", Config.OPENAI_MODEL)

    def build_prompt(self, input_data: str, output_data: str, **kwargs) -> str:
        script_text = _stringify_script_output(output_data)
        metadata_context = _build_script_context(kwargs)
        expected_context = _build_expected_output_context(kwargs)
        return f"""Bạn là reviewer chấm chất lượng script tuyển dụng ngắn.

                Hãy đánh giá metric "relevance" giữa JD gốc và script sinh ra.

                JD GỐC:
                {input_data}

                SCRIPT SINH RA:
                {script_text}

                SCRIPT KỲ VỌNG/REFERENCE:
                {expected_context}

                CONTEXT:
                {metadata_context}

                Chỉ chấm mức độ bám JD, không chấm độ hay của văn phong.
                Nếu có script kỳ vọng, dùng nó như reference phụ để hiểu mức độ coverage mong muốn, nhưng vẫn ưu tiên đối chiếu trực tiếp với JD gốc.

                Quy trình đánh giá:
                1. Xác định các nhóm thông tin cốt lõi trong JD nếu có: vị trí, cấp độ/kinh nghiệm, địa điểm, quyền lợi, yêu cầu/chuyên môn.
                2. Kiểm tra script có nhắc đúng các thông tin cốt lõi đó hay không.
                3. Phạt mạnh nếu script thêm thông tin không có trong JD, suy diễn quá mức, hoặc làm sai nghĩa.
                4. Phạt vừa nếu script bỏ sót vài ý quan trọng nhưng phần còn lại vẫn đúng.
                5. Không phạt chỉ vì script diễn đạt ngắn gọn hơn JD, miễn là không sai lệch.

                Rubric 1-5:
                1 = Sai lệch rõ, nhiều hallucination hoặc bỏ sót phần lớn thông tin cốt lõi.
                2 = Có bám JD nhưng thiếu nhiều ý quan trọng hoặc có vài thông tin sai đáng kể.
                3 = Bám JD ở mức trung bình, đúng phần lớn nhưng còn thiếu/sơ sài.
                4 = Bám JD tốt, chỉ thiếu ít chi tiết nhỏ và không có sai lệch đáng kể.
                5 = Bám rất sát JD, đúng và đầy đủ các ý chính, không thêm sai thông tin.

                Trả về JSON:
                {{
                "score": 0,
                "reasoning": "...",
                "strengths": ["..."],
                "issues": ["..."]
                }}"""

    def evaluate(self, input_data: str, output_data: str, **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)
        prompt = self.build_prompt(input_data, output_data, **kwargs)
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


class ScriptToneEvaluator(LLMJudgeEvaluator):
    """Đánh giá giọng văn: phù hợp đối tượng ứng viên và bối cảnh"""

    def __init__(self):
        super().__init__("tone_of_voice", "script_generation", Config.OPENAI_MODEL)

    def build_prompt(self, input_data: str, output_data: str, **kwargs) -> str:
        script_text = _stringify_script_output(output_data)
        metadata_context = _build_script_context(kwargs)
        expected_context = _build_expected_output_context(kwargs)
        return f"""Bạn là reviewer chấm tone of voice cho script tuyển dụng video ngắn.

                JD GỐC:
                {input_data}

                SCRIPT:
                {script_text}

                SCRIPT KỲ VỌNG/REFERENCE:
                {expected_context}

                CONTEXT:
                {metadata_context}

                Chỉ chấm "tone_of_voice", không chấm độ đúng sai nội dung JD.
                Nếu có script kỳ vọng, dùng nó như reference phụ để hiểu tone mong muốn, nhưng không được copy điểm chỉ vì giống wording.

                Tiêu chí cần xem:
                1. Giọng điệu có phù hợp với đối tượng ứng viên và cấp độ vị trí không.
                2. Script có đủ tự nhiên, thuyết phục, không sáo rỗng hoặc quá salesy/clickbait không.
                3. Mức độ chuyên nghiệp có phù hợp ngành nghề và bối cảnh tuyển dụng không.
                4. CTA có giữ cùng tone với phần trước hay bị lệch giọng quá mạnh không.

                Rubric 1-5:
                1 = Lệch tone nặng, phản cảm, quá gượng hoặc quá clickbait.
                2 = Có vài phần phù hợp nhưng nhìn chung tone không hợp đối tượng mục tiêu.
                3 = Tone chấp nhận được nhưng còn generic hoặc chưa nhất quán.
                4 = Tone phù hợp, tự nhiên, có sức thuyết phục, chỉ còn vài điểm hơi gượng.
                5 = Tone rất phù hợp với ứng viên mục tiêu, nhất quán, tự nhiên, thuyết phục.

                Trả về JSON:
                {{
                "score": 0,
                "reasoning": "...",
                "strengths": ["..."],
                "issues": ["..."]
                }}"""

    def evaluate(self, input_data: str, output_data: str, **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)
        prompt = self.build_prompt(input_data, output_data, **kwargs)
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
