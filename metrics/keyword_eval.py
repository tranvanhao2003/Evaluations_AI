"""
Keyword Generation Evaluators
"""
import json
from Evaluation_AI.core.base_evaluator import DeterministicEvaluator, LLMJudgeEvaluator, MetricScore
from Evaluation_AI.config import Config


GENERIC_VISUAL_TERMS = {
    "image", "photo", "picture", "visual", "scene", "background",
    "nice", "beautiful", "good", "job", "work", "company",
}


def _extract_keywords(output_data):
    if isinstance(output_data, dict):
        keywords = output_data.get("keywords")
        if isinstance(keywords, list):
            return [str(k).strip() for k in keywords if str(k).strip()]
        video_segments = output_data.get("video_segments")
        if isinstance(video_segments, list):
            extracted = []
            for segment in video_segments:
                if not isinstance(segment, dict):
                    continue
                queries = segment.get("queries")
                if isinstance(queries, list):
                    extracted.extend(str(query).strip() for query in queries if str(query).strip())
            if extracted:
                return extracted
        image_queries = output_data.get("image_queries")
        if isinstance(image_queries, list):
            extracted = []
            for item in image_queries:
                if isinstance(item, dict):
                    query = item.get("query")
                    if isinstance(query, str) and query.strip():
                        extracted.append(query.strip())
            if extracted:
                return extracted
    if isinstance(output_data, list):
        extracted = []
        for item in output_data:
            if isinstance(item, dict):
                query = item.get("query")
                if isinstance(query, str) and query.strip():
                    extracted.append(query.strip())
            else:
                text = str(item).strip()
                if text:
                    extracted.append(text)
        return extracted
    return []


def _extract_keyword_context(input_data: str, kwargs) -> str:
    lines = []
    parsed_text = ""
    if isinstance(input_data, str) and input_data.strip():
        parsed_text = input_data.strip()
        try:
            parsed = json.loads(input_data)
            if isinstance(parsed, dict):
                captions = parsed.get("captions")
                if isinstance(captions, list):
                    parts = []
                    for caption in captions:
                        if isinstance(caption, dict):
                            text = caption.get("text")
                            if isinstance(text, str) and text.strip():
                                parts.append(text.strip())
                    if parts:
                        parsed_text = " ".join(parts)
        except Exception:
            pass
    if parsed_text:
        lines.append(f"Kịch bản/caption nguồn: {parsed_text}")
    expected_output = kwargs.get("expected_output")
    if isinstance(expected_output, list) and expected_output:
        lines.append("Keyword kỳ vọng/reference: " + ", ".join(str(item) for item in expected_output[:10]))
    elif isinstance(expected_output, dict):
        ref_keywords = expected_output.get("keywords")
        if isinstance(ref_keywords, list) and ref_keywords:
            lines.append("Keyword kỳ vọng/reference: " + ", ".join(str(item) for item in ref_keywords[:10]))
        ref_queries = expected_output.get("image_queries")
        if isinstance(ref_queries, list) and ref_queries:
            rendered_queries = []
            for item in ref_queries[:10]:
                if isinstance(item, dict):
                    query = item.get("query")
                    if isinstance(query, str) and query.strip():
                        rendered_queries.append(query.strip())
            if rendered_queries:
                lines.append("Image query kỳ vọng/reference: " + ", ".join(rendered_queries))
        ref_video_segments = expected_output.get("video_segments")
        if isinstance(ref_video_segments, list) and ref_video_segments:
            rendered_video_queries = []
            for segment in ref_video_segments[:10]:
                if not isinstance(segment, dict):
                    continue
                queries = segment.get("queries")
                if isinstance(queries, list):
                    rendered_video_queries.extend(str(query).strip() for query in queries[:3] if str(query).strip())
            if rendered_video_queries:
                lines.append("Video query kỳ vọng/reference: " + ", ".join(rendered_video_queries[:10]))
    test_case = kwargs.get("test_case")
    metadata = getattr(test_case, "metadata", None) if test_case else None
    if isinstance(metadata, dict):
        for key, label in (
            ("industry", "Ngành"),
            ("target_audience", "Đối tượng"),
            ("job_level", "Cấp độ"),
            ("category", "Loại template"),
        ):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                lines.append(f"{label}: {value.strip()}")
    return "\n".join(lines)


class KeywordRelevanceEvaluator(LLMJudgeEvaluator):
    """LLM judge: Kiểm tra độ liên quan visual của bộ query/search term"""

    def __init__(self, stage_name: str = "keyword_generation"):
        super().__init__("visual_relevance", stage_name, Config.OPENAI_MODEL)

    def build_prompt(self, input_data: str, output_data: list, **kwargs) -> str:
        keyword_lines = "\n".join(f"- {keyword}" for keyword in output_data)
        context = _extract_keyword_context(input_data, kwargs)
        return f"""Bạn là reviewer chấm chất lượng bộ query tìm stock visual cho video tuyển dụng.

                CONTEXT:
                {context}

                BỘ TỪ KHÓA:
                {keyword_lines}

                Chỉ chấm "visual_relevance".
                Không chấm diversity.
                Không chấm searchability/trending volume.

                Tiêu chí đánh giá:
                1. Query có gợi ra đúng chủ thể/hành động/bối cảnh visual cần tìm từ caption không.
                2. Query có đủ cụ thể để sinh ra hình ảnh hoặc footage đúng ngữ cảnh tuyển dụng, thay vì quá chung chung.
                3. Ưu tiên keyword mô tả được vai trò nghề nghiệp, môi trường làm việc, công nghệ hoặc bối cảnh liên quan.
                4. Phạt nếu query lệch ý, quá abstract, hoặc không giúp chọn được visual đúng.

                Rubric 1-5:
                1 = Phần lớn keyword lệch ngữ cảnh hoặc không thể đại diện đúng visual cần tìm.
                2 = Có vài keyword đúng nhưng đa số còn generic hoặc sai focus.
                3 = Khá liên quan nhưng vẫn còn vài keyword mơ hồ hoặc thiếu specificity.
                4 = Liên quan tốt, đa số keyword giúp tìm đúng visual, chỉ còn vài điểm chưa tối ưu.
                5 = Rất liên quan và cụ thể, bộ keyword bám sát visual intent của caption.

                Trả về JSON:
                {{
                "score": 0,
                "reasoning": "...",
                "good_keywords": ["..."],
                "bad_keywords": ["..."]
                }}"""

    def evaluate(self, input_data: str, output_data: list, **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)
        keywords = _extract_keywords(output_data)
        if not keywords:
            return MetricScore(
                metric_name=self.metric_name,
                score=0.0,
                passed=False,
                threshold=threshold,
                reasoning="No keywords provided",
                raw_data={"keywords": []}
            )

        prompt = self.build_prompt(input_data, keywords, **kwargs)
        result = self.call_llm(prompt)

        if not result or "error" in result:
            return MetricScore(
                metric_name=self.metric_name,
                score=0.65,
                passed=False,
                threshold=threshold,
                reasoning=f"LLM call failed: {result.get('error', 'Unknown error')}",
                raw_data=result
            )

        score_5 = float(result.get("score", result.get("average_visual_score_1_to_5", 1.0)))
        score_1 = (score_5 - 1) / 4 if score_5 > 0 else 0.0

        return MetricScore(
            metric_name=self.metric_name,
            score=score_1,
            passed=score_1 >= threshold,
            threshold=threshold,
            reasoning=result.get("reasoning", "No reasoning provided"),
            raw_data=result
        )


class SearchabilityEvaluator(DeterministicEvaluator):
    """Đánh giá khả năng tìm kiếm của keyword trên stock library"""

    def __init__(self, stage_name: str = "keyword_generation"):
        super().__init__("searchability", stage_name)

    def evaluate(self, input_data: str, output_data: dict, **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)
        keywords = _extract_keywords(output_data)
        if not keywords:
            return MetricScore(
                metric_name=self.metric_name,
                score=0.0,
                passed=False,
                threshold=threshold,
                reasoning="No keywords provided",
                raw_data={"result_count": 0}
            )

        result_count = output_data.get("pexels_results_count", 0) if isinstance(output_data, dict) else 0

        quality_scores = []
        for keyword in keywords:
            normalized = keyword.lower().strip()
            words = [word for word in normalized.split() if word]
            word_count_score = 1.0 if 1 <= len(words) <= 4 else 0.55
            generic_penalty = 0.75 if all(word in GENERIC_VISUAL_TERMS for word in words) else 1.0
            ascii_score = 1.0 if any("a" <= ch.lower() <= "z" for ch in normalized) else 0.8
            quality_scores.append(word_count_score * generic_penalty * ascii_score)

        keyword_quality = sum(quality_scores) / len(quality_scores)
        result_signal = min(1.0, result_count / 20.0) if result_count > 0 else 0.45
        score = (keyword_quality * 0.75) + (result_signal * 0.25)

        return MetricScore(
            metric_name=self.metric_name,
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            reasoning=f"Searchability quality={keyword_quality:.2f}, search results={result_count}",
            raw_data={"result_count": result_count, "num_keywords": len(keywords)}
        )


class KeywordDiversityEvaluator(DeterministicEvaluator):
    """Đánh giá độ đa dạng từ khóa giữa các cảnh"""

    def __init__(self, stage_name: str = "keyword_generation"):
        super().__init__("diversity", stage_name)

    def evaluate(self, input_data: str, output_data: dict, **kwargs) -> MetricScore:
        threshold = Config.get_threshold(self.metric_name)
        keywords = _extract_keywords(output_data)
        if not keywords:
            return MetricScore(
                metric_name=self.metric_name,
                score=0.0,
                passed=False,
                threshold=threshold,
                reasoning="No keywords provided",
                raw_data={"unique_ratio": 0.0}
            )

        normalized = [keyword.lower().strip() for keyword in keywords if keyword.strip()]
        unique_ratio = len(set(normalized)) / len(normalized)

        head_words = [keyword.split()[0] for keyword in normalized if keyword.split()]
        head_diversity = len(set(head_words)) / max(1, len(head_words))

        token_pool = []
        for keyword in normalized:
            token_pool.extend(word for word in keyword.split() if word not in GENERIC_VISUAL_TERMS)
        token_diversity = len(set(token_pool)) / max(1, len(token_pool))

        score = (unique_ratio * 0.5) + (head_diversity * 0.2) + (token_diversity * 0.3)

        return MetricScore(
            metric_name=self.metric_name,
            score=score,
            passed=score >= threshold,
            threshold=threshold,
            reasoning=(
                f"Keyword diversity unique={unique_ratio:.2f}, "
                f"head={head_diversity:.2f}, token={token_diversity:.2f}"
            ),
            raw_data={
                "unique_ratio": unique_ratio,
                "head_diversity": head_diversity,
                "token_diversity": token_diversity,
            }
        )
