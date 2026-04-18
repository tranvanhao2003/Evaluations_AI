"""
Backend Client - API-first, fallback to direct BE service calls.
"""
import asyncio
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
from urllib import request as urlrequest
from urllib import error as urlerror


class BackendClient:
    """Client để gọi BE endpoints hoặc BE service functions trực tiếp."""

    LEGACY_SCRIPT_TEMPLATE_NAME_MAP = {
        "knowledge_sharing": "Chia sẻ kiến thức",
        "storytelling": "Kể chuyện đánh lừa",
        "story_twist": "Kể chuyện đánh lừa",
        "psychological_guide": "Dẫn dắt tâm lý",
        "problem_solving": "Khó khăn nghề nghiệp",
        "inspirational": "Pain point nổi bật",
        "urgency": "Fast Hire Now (Urgent)",
        "benefit_focus": "Top 3 lý do",
        "experience_highlight": "Q&A / Giải đáp thắc mắc",
        "community_focus": "Vlog cá nhân",
        "growth_focus": "POV: Một ngày đi làm",
        "impact_driven": "ShortJD Voice Master",
        "wellbeing": "Thông báo truy tìm",
    }

    def __init__(self, base_url: str = "http://localhost:8001", use_mock: bool = False, strict_real: bool = False):
        self.base_url = base_url.rstrip("/")
        self.use_mock = use_mock
        self.strict_real = strict_real and not use_mock
        self._be_root = Path(__file__).resolve().parents[2] / "BE"
        self._script_templates_cache: Optional[List[Dict[str, Any]]] = None

    def _raise_real_backend_error(self, stage: str, reasons: List[str]) -> None:
        cleaned = [str(reason).strip() for reason in reasons if str(reason).strip()]
        detail = " | ".join(cleaned) if cleaned else "unknown backend failure"
        raise RuntimeError(f"Real backend required for '{stage}' but execution failed: {detail}")

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urlrequest.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=120) as response:
                content = response.read().decode("utf-8")
        except urlerror.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"BE HTTP {exc.code} at {path}: {detail}") from exc
        except Exception as exc:
            raise RuntimeError(f"BE request failed at {path}: {exc}") from exc

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"BE returned non-JSON payload at {path}: {content[:200]}") from exc

    def _ensure_be_import_path(self):
        be_root = str(self._be_root)
        if be_root not in sys.path:
            sys.path.insert(0, be_root)

    def _run_async(self, coro):
        try:
            return asyncio.run(coro)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()

    @staticmethod
    def _normalize_template_name(name: str) -> str:
        return re.sub(r"\s+", " ", str(name or "").strip().lower())

    def get_script_templates(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """Fetch current script templates from BE so eval can resolve real template ids."""
        if self._script_templates_cache is not None and not force_refresh:
            return self._script_templates_cache

        candidates: List[Dict[str, Any]] = []

        for path in ("/api/v1/templates", "/api/v1/script/templates"):
            if candidates:
                break
            try:
                with urlrequest.urlopen(f"{self.base_url}{path}", timeout=30) as response:
                    raw = response.read().decode("utf-8")
                payload = json.loads(raw)
                if isinstance(payload, dict) and isinstance(payload.get("templates"), list):
                    candidates = payload["templates"]
            except Exception:
                continue

        if not candidates:
            try:
                self._ensure_be_import_path()
                from service.script_service import get_all_templates

                result = get_all_templates()
                if isinstance(result, list):
                    candidates = result
            except Exception:
                candidates = []

        cleaned = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            item_name = item.get("name")
            if isinstance(item_id, int) and isinstance(item_name, str) and item_name.strip():
                cleaned.append(item)

        self._script_templates_cache = cleaned
        return cleaned

    def resolve_script_template_id(
        self,
        template_id: Optional[int],
        metadata: Optional[Dict[str, Any]] = None,
        case_name: str = "",
    ) -> int:
        """
        Resolve dataset template metadata to the real template id currently available in BE.
        BE script generation uses DB ids, not legacy logical ids from old evaluation datasets.
        """
        if self.use_mock:
            return int(template_id or 1)

        metadata = metadata or {}
        templates = self.get_script_templates()
        by_id = {
            int(item["id"]): item for item in templates
            if isinstance(item.get("id"), int)
        }
        by_name = {
            self._normalize_template_name(str(item["name"])): item
            for item in templates
            if isinstance(item.get("name"), str)
        }

        explicit_be_id = metadata.get("be_template_id")
        if isinstance(explicit_be_id, int) and explicit_be_id in by_id:
            return explicit_be_id

        explicit_be_name = metadata.get("be_template_name")
        if isinstance(explicit_be_name, str) and explicit_be_name.strip():
            matched = by_name.get(self._normalize_template_name(explicit_be_name))
            if matched:
                return int(matched["id"])
            raise RuntimeError(
                f"BE template name '{explicit_be_name}' not found for case '{case_name}'."
            )

        if isinstance(template_id, int) and template_id in by_id:
            return template_id

        category = str(metadata.get("category") or "").strip().lower()
        mapped_name = self.LEGACY_SCRIPT_TEMPLATE_NAME_MAP.get(category)
        if mapped_name:
            matched = by_name.get(self._normalize_template_name(mapped_name))
            if matched:
                print(
                    f"🔁 Resolved legacy template for case '{case_name}' "
                    f"category='{category}' -> BE template '{mapped_name}' (id={matched['id']})"
                )
                return int(matched["id"])

        available = ", ".join(
            f"{item['id']}:{item['name']}" for item in templates[:20]
        ) or "none"
        raise RuntimeError(
            f"Cannot resolve real BE template for case '{case_name}'. "
            f"dataset template_id={template_id}, category='{category}'. "
            f"Set metadata.be_template_name or metadata.be_template_id explicitly. "
            f"Available templates: {available}"
        )

    def generate_script(
        self,
        jd_text: str,
        video_duration: int = 60,
        template_id: int = 1,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate script từ JD."""
        if self.use_mock:
            return self._mock_generate_script(jd_text, video_duration)

        payload = {
            "jd_content": jd_text,
            "template_id": int(template_id),
            "session_id": session_id,
        }

        try:
            result = self._post_json("/api/v1/script/generate", payload)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

        return self._service_generate_script(jd_text=jd_text, template_id=int(template_id))

    def _service_generate_script(self, jd_text: str, template_id: int) -> Dict[str, Any]:
        self._ensure_be_import_path()
        try:
            from service.script_service import generate_script
        except Exception as exc:
            raise RuntimeError(f"BE script service import failed: {exc}") from exc

        result = self._run_async(generate_script(jd_content=jd_text, template_id=template_id))
        if isinstance(result, dict):
            return result
        raise RuntimeError("Invalid script response from BE service")

    def _mock_generate_script(self, jd_text: str, duration: int) -> Dict[str, Any]:
        """Generate mock script with LLM parsing into hook/body/cta"""
        # Generate full script first
        full_script = (
            f"Bạn có biết về {jd_text[:50]}? "
            f"Chúng tôi đang tìm người tài năng tại các vị trí này. "
            f"Lương, phúc lợi và cơ hội phát triển rất hấp dẫn. "
            f"Nộp hồ sơ ngay hôm nay để không bị lỡ cơ hội này!"
        )
        
        # Try to use LLM to parse script into hook/body/cta
        try:
            from Evaluation_AI.core.langfuse_manager import LangfuseExperimentManager
            manager = LangfuseExperimentManager()
            
            # Use score_current_span to parse with LLM
            parse_prompt = f"""Hãy tách đoạn script sau thành 3 phần: Hook, Body, CTA.
            
Script: {full_script}

Trả về JSON:
{{"hook": "...", "body": "...", "cta": "..."}}"""
            
            # For mock, just do simple parsing
            sentences = [s.strip() for s in full_script.split('.') if s.strip()]
            hook = sentences[0] if len(sentences) > 0 else full_script[:30]
            body = '. '.join(sentences[1:-1]) if len(sentences) > 2 else sentences[1] if len(sentences) > 1 else full_script[30:80]
            cta = sentences[-1] if len(sentences) > 1 else full_script[-30:]
            
        except Exception:
            # Fallback: simple split
            sentences = [s.strip() for s in full_script.split('.') if s.strip()]
            hook = sentences[0] if len(sentences) > 0 else full_script[:30]
            body = '. '.join(sentences[1:-1]) if len(sentences) > 2 else sentences[1] if len(sentences) > 1 else full_script[30:80]
            cta = sentences[-1] if len(sentences) > 1 else full_script[-30:]
        
        return {
            "status": "success",
            "hook": hook,
            "body": body,
            "cta": cta,
            "full_script": full_script,
            "template_used": {"id": 1, "name": "mock"},
            "duration_seconds": duration,
        }

    def generate_keywords(
        self,
        text: str = "",
        captions: Optional[List[Dict[str, Any]]] = None,
        job_category: str = "recruitment",
        session_id: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate image keywords."""
        if self.use_mock:
            keywords = self._mock_generate_keywords(text)
            return {
                "keywords": keywords,
                "raw_queries": [{"query": keyword} for keyword in keywords],
                "pexels_results_count": max(0, len(keywords) * 6),
                "job_category": job_category,
            }

        payload_captions = captions or []
        if not payload_captions and text.strip():
            payload_captions = [{
                "text": text.strip(),
                "start": 0.0,
                "end": float(max(3, len(text.split()) / 2.5)),
            }]

        try:
            result = self._post_json(
                "/api/v1/search/images",
                {
                    "captions": payload_captions,
                    "job_category": job_category,
                    "session_id": session_id,
                },
            )
            return self._normalize_keyword_result(result, job_category)
        except Exception:
            return self._service_generate_keywords(payload_captions, job_category)

    def _service_generate_keywords(
        self, captions: List[Dict[str, Any]], job_category: str
    ) -> Dict[str, Any]:
        self._ensure_be_import_path()
        try:
            from service.search_service import generate_image_search_terms
        except Exception as exc:
            raise RuntimeError(f"BE keyword service import failed: {exc}") from exc

        result = self._run_async(
            generate_image_search_terms(captions_timed=captions, n=5, job_category=job_category)
        )
        return self._normalize_keyword_result({"image_queries": result}, job_category)

    def _normalize_keyword_result(self, result: Any, job_category: str) -> Dict[str, Any]:
        queries = []
        raw_queries = []
        if isinstance(result, dict):
            raw_queries = result.get("image_queries", [])
            for item in raw_queries:
                if isinstance(item, dict):
                    query = item.get("query")
                    if isinstance(query, str) and query.strip():
                        queries.append(query.strip())
        if not queries:
            raise RuntimeError("Invalid keyword response from BE")
        return {
            "keywords": queries,
            "raw_queries": raw_queries,
            "pexels_results_count": max(0, len(queries) * 6),
            "job_category": job_category,
        }

    def _mock_generate_keywords(self, text: str) -> List[str]:
        words = text.lower().split()
        keywords = [word.strip(".,!?;:") for word in words if len(word) > 3]
        return list(dict.fromkeys(keywords))[:5]

    def transcribe_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        STT processor:
        - Ưu tiên dùng đúng luồng BE hiện tại: voice generate -> transcription -> align với script.
        - Nếu hệ thống có STT API riêng thì vẫn thử dùng.
        - Chỉ fallback cục bộ khi cả 2 hướng đều không chạy được.
        """
        normalized_text = str(text or "").strip()
        errors: List[str] = []
        if not self.use_mock:
            aligned = self._transcribe_via_voice_pipeline(normalized_text, errors=errors, **kwargs)
            if aligned:
                return aligned

            try:
                from Evaluation_AI.config import Config

                payload = {}
                if normalized_text:
                    payload["text"] = normalized_text
                for key in ("audio_url", "audio_path", "language", "session_id"):
                    value = kwargs.get(key)
                    if value is not None:
                        payload[key] = value
                result = self._post_json(Config.STT_API_PATH, payload)
                normalized = self._normalize_stt_result(result)
                if normalized:
                    return normalized
                errors.append(f"Invalid STT API response at {Config.STT_API_PATH}")
            except Exception as exc:
                errors.append(str(exc))

            if self.strict_real:
                self._raise_real_backend_error("stt_transcription", errors)

        return {
            "transcript": normalized_text,
            "word_timestamps": self._build_word_timestamps(normalized_text),
        }

    def transcribe_raw_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Raw STT processor:
        synthesize audio and return transcript/timings before alignment to script.
        """
        normalized_text = str(text or "").strip()
        errors: List[str] = []
        if not self.use_mock:
            raw = self._transcribe_raw_via_voice_pipeline(normalized_text, errors=errors, **kwargs)
            if raw:
                return raw

            if self.strict_real:
                self._raise_real_backend_error("stt_raw_transcription", errors)

        return {
            "transcript": normalized_text,
            "word_timestamps": self._build_word_timestamps(normalized_text),
        }

    def _transcribe_via_voice_pipeline(self, text: str, errors: Optional[List[str]] = None, **kwargs) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        errors = errors if errors is not None else []

        payload = {
            "text": text,
            "tts_provider": kwargs.get("tts_provider") or "vieneu",
            "language": kwargs.get("language") or "vi",
            "gender": kwargs.get("gender") or "female",
            "voice_id": kwargs.get("voice_id"),
            "speed": kwargs.get("speed", 1.0),
            "split_by_sentence": bool(kwargs.get("split_by_sentence", False)),
            "session_id": kwargs.get("session_id"),
        }

        try:
            result = self._post_json("/api/v1/voice/generate", payload)
            normalized = self._normalize_stt_result(result)
            if normalized:
                return normalized
            errors.append("Invalid response from /api/v1/voice/generate")
        except Exception as exc:
            errors.append(str(exc))

        try:
            self._ensure_be_import_path()
            from service.voice_service import (
                generate_voice_and_subtitles,
                generate_voice_split_sentences,
            )

            service_fn = (
                generate_voice_split_sentences
                if payload["split_by_sentence"]
                else generate_voice_and_subtitles
            )
            result = self._run_async(
                service_fn(
                    text=text,
                    language=payload["language"],
                    gender=payload["gender"],
                    tts_provider=payload["tts_provider"],
                    voice_id=payload["voice_id"],
                    speed=float(payload["speed"] or 1.0),
                )
            )
            return self._normalize_stt_result(result)
        except Exception as exc:
            errors.append(f"BE voice service execution failed: {exc}")
            return None

    def _transcribe_raw_via_voice_pipeline(self, text: str, errors: Optional[List[str]] = None, **kwargs) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        errors = errors if errors is not None else []

        tts_provider = str(kwargs.get("tts_provider") or "vieneu").strip().lower()
        language = str(kwargs.get("language") or "vi").strip()
        gender = str(kwargs.get("gender") or "female").strip()
        voice_id = kwargs.get("voice_id")
        speed = float(kwargs.get("speed", 1.0) or 1.0)

        self._ensure_be_import_path()

        if tts_provider == "edge_tts":
            from utils.voice_client import EdgeTTSClient

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tmp.close()
            try:
                client = EdgeTTSClient(language=language, gender=gender, voice=voice_id)
                _, word_timings = self._run_async(client.generate_voice(text, tmp.name, speed=speed))
                normalized_timestamps = self._normalize_word_timings(word_timings or [])
                transcript = " ".join(item["text"] for item in normalized_timestamps if item.get("text"))
                if not transcript:
                    return None
                return {
                    "transcript": transcript,
                    "word_timestamps": normalized_timestamps,
                }
            except Exception:
                errors.append("edge_tts raw transcription failed")
                return None
            finally:
                try:
                    Path(tmp.name).unlink(missing_ok=True)
                except Exception:
                    pass

        try:
            from configs.configs import VIENEU_API_BASE, VIENEU_MODEL_ID
            from utils.vieneu_client import (
                init_vieneu,
                is_available,
                _list_preset_voices,
                _synth_preset,
                _change_audio_speed,
                _get_wav_duration,
                _call_transcription_api,
            )
        except Exception as exc:
            errors.append(f"VieNeu import failed: {exc}")
            return None

        if not is_available():
            init_vieneu(api_base=VIENEU_API_BASE, model_name=VIENEU_MODEL_ID)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.close()
        try:
            chosen_voice = voice_id
            if not chosen_voice:
                voices = _list_preset_voices()
                if voices:
                    chosen_voice = voices[0][1]

            if not chosen_voice:
                return None

            _synth_preset(text, chosen_voice, tmp.name)
            if speed != 1.0:
                _change_audio_speed(tmp.name, speed)

            audio_duration = _get_wav_duration(tmp.name)
            transcript_words = _call_transcription_api(tmp.name, audio_duration, language=language)
            normalized_timestamps = self._normalize_word_timings(transcript_words or [])
            transcript = " ".join(item["text"] for item in normalized_timestamps if item.get("text"))
            if not transcript:
                return None

            return {
                "transcript": transcript,
                "word_timestamps": normalized_timestamps,
            }
        except Exception as exc:
            errors.append(f"VieNeu raw transcription failed: {exc}")
            return None
        finally:
            try:
                Path(tmp.name).unlink(missing_ok=True)
            except Exception:
                pass

    def _normalize_stt_result(self, result: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(result, dict):
            return None

        transcript = None
        for key in ("transcript", "text", "output", "content", "full_text"):
            value = result.get(key)
            if isinstance(value, str) and value.strip():
                transcript = value.strip()
                break

        timestamps = result.get("word_timestamps")
        if not isinstance(timestamps, list):
            timestamps = result.get("word_timings")
        if not isinstance(timestamps, list):
            timestamps = result.get("timestamps")

        normalized_timestamps = self._normalize_word_timings(timestamps or [])
        if not transcript and normalized_timestamps:
            transcript = " ".join(item["text"] for item in normalized_timestamps if item.get("text"))

        if not transcript:
            return None

        return {
            "transcript": transcript,
            "word_timestamps": normalized_timestamps or self._build_word_timestamps(transcript),
        }

    def split_voice(self, text: str, **kwargs) -> Dict[str, Any]:
        """Chia script thành các đoạn voice bằng BE HTTP API."""
        if self.use_mock:
            return self._mock_split_voice(text)

        errors: List[str] = []
        normalized_text = str(text or "").strip()

        try:
            result = self._post_json("/api/v1/voice/split", {"text": normalized_text})
            segments = result.get("segments") if isinstance(result, dict) else None
            if not isinstance(segments, list):
                raise RuntimeError("Invalid voice split response from /api/v1/voice/split")
            cleaned_segments = [str(segment).strip() for segment in segments if str(segment).strip()]
            if not cleaned_segments:
                raise RuntimeError("Empty voice split result from BE API")
            return {"segments": cleaned_segments}
        except Exception as exc:
            errors.append(str(exc))

        try:
            self._ensure_be_import_path()
            from service.voice_service import split_voice_segments

            segments = self._run_async(split_voice_segments(normalized_text))
            if not isinstance(segments, list):
                raise RuntimeError("Invalid voice split result from BE service")
            cleaned_segments = [str(segment).strip() for segment in segments if str(segment).strip()]
            if not cleaned_segments:
                raise RuntimeError("Empty voice split result from BE service")
            return {"segments": cleaned_segments}
        except Exception as exc:
            errors.append(str(exc))
            if self.strict_real:
                self._raise_real_backend_error("voice_splitting", errors)
            return self._mock_split_voice(text)

    def _mock_split_voice(self, text: str) -> Dict[str, Any]:
        segments = [seg.strip() for seg in re.split(r"(?<=[.!?])\s+", str(text or "").strip()) if seg.strip()]
        if not segments and text:
            segments = [str(text).strip()]
        return {"segments": segments}

    def split_subtitles(self, text: str, word_timings: Optional[List[Dict]] = None, **kwargs) -> List[Dict]:
        """Chia subtitle bằng BE HTTP API."""
        if self.use_mock:
            return self._mock_split_subtitles(text, word_timings or [])

        errors: List[str] = []
        normalized_text = str(text or "").strip()
        normalized_timings = self._normalize_word_timings(word_timings or self._build_word_timestamps(normalized_text))

        try:
            result = self._post_json(
                "/api/v1/voice/subtitles/split",
                {
                    "text": normalized_text,
                    "word_timings": normalized_timings,
                },
            )
            captions = result.get("captions") if isinstance(result, dict) else None
            if isinstance(captions, list) and captions:
                return captions
            raise RuntimeError("Empty subtitle result from BE API")
        except Exception as exc:
            errors.append(str(exc))

        try:
            self._ensure_be_import_path()
            from service.subtitle_service import get_captions_by_llm_lines, get_captions_with_time
            from service.voice_service import _get_llm_subtitles

            llm_lines = self._run_async(_get_llm_subtitles(normalized_text))
            if isinstance(llm_lines, list) and llm_lines:
                captions = get_captions_by_llm_lines(normalized_timings, llm_lines)
                if captions:
                    return captions
            captions = get_captions_with_time(normalized_timings, max_caption_size=60)
            if captions:
                return captions
            raise RuntimeError("Empty subtitle result from BE service")
        except Exception as exc:
            errors.append(str(exc))
            if self.strict_real:
                self._raise_real_backend_error("subtitle_splitting", errors)

        return self._mock_split_subtitles(text, word_timings or [])

    def _mock_split_subtitles(self, text: str, word_timings: List[Dict]) -> List[Dict]:
        normalized_timings = self._normalize_word_timings(word_timings or self._build_word_timestamps(text))
        if not normalized_timings:
            return []

        captions = []
        current_words = []
        current_chars = 0
        for index, word in enumerate(normalized_timings):
            token = str(word.get("text", "")).strip()
            if not token:
                continue
            current_words.append(word)
            current_chars += len(token) + (1 if current_words else 0)
            is_last = index == len(normalized_timings) - 1
            should_split = is_last or current_chars >= 42 or token.endswith((".", "!", "?", ",", ";", ":"))
            if should_split and current_words:
                captions.append({
                    "text": " ".join(w["text"] for w in current_words),
                    "start": round(current_words[0]["start"], 3),
                    "end": round(current_words[-1]["end"], 3),
                })
                current_words = []
                current_chars = 0
        return captions

    def _build_word_timestamps(self, text: str) -> List[Dict[str, Any]]:
        words = [word for word in str(text or "").split() if word]
        time_cursor = 0.0
        word_timestamps = []
        for word in words:
            duration = max(0.12, min(0.45, len(word) * 0.03))
            start = round(time_cursor, 3)
            end = round(time_cursor + duration, 3)
            word_timestamps.append({"text": word, "start": start, "end": end})
            time_cursor = end
        return word_timestamps

    def _normalize_word_timings(self, word_timings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized = []
        for item in word_timings:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if not isinstance(text, str):
                text = item.get("word")
            start = item.get("start")
            end = item.get("end")
            if not isinstance(text, str) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            normalized.append({
                "text": text.strip(),
                "start": float(start),
                "end": float(end),
            })
        return normalized
