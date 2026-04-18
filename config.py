import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), "..", "BE", ".env")
load_dotenv(dotenv_path)
load_dotenv()  # Also load from root .env

class Config:
    """Toàn bộ cấu hình hệ thống đánh giá"""
    
    # ===== Paths =====
    PROJECT_ROOT = Path(__file__).parent.parent
    EVAL_ROOT = Path(__file__).parent
    DATASETS_DIR = EVAL_ROOT / "datasets"
    RESULTS_DIR = EVAL_ROOT / "results"
    
    # Tạo thư mục
    RESULTS_DIR.mkdir(exist_ok=True)
    DATASETS_DIR.mkdir(exist_ok=True)
    
    # ===== API Keys =====
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
    LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)
    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")
    USE_MOCK_BACKEND = os.getenv("USE_MOCK_BACKEND", "false").lower() == "true"
    REQUIRE_REAL_BACKEND = os.getenv("REQUIRE_REAL_BACKEND", "true").lower() == "true"
    STT_API_PATH = os.getenv("STT_API_PATH", "/api/v1/stt/transcribe")
    
    # ===== Evaluation Thresholds =====
    METRIC_THRESHOLDS = {
        # Script Generation
        "relevance": 0.75,
        "structure": 0.8,
        "tone_of_voice": 0.75,
        "length_constraint": 0.8,
        # STT Transcription
        "word_error_rate": 0.85,  # score = 1 - WER
        "punctuation_capitalization": 0.75,
        "timestamp_accuracy": 0.75,
        # Voice Splitting
        "semantic_completeness": 0.8,
        "duration_balance": 0.75,
        "natural_pause": 0.75,
        # Subtitle Splitting
        "readability": 0.8,
        "synchronization": 0.8,
        "line_break_logic": 0.8,
        # Keyword Generation
        "visual_relevance": 0.75,
        "searchability": 0.75,
        "diversity": 0.75,
    }
    
    # ===== Stage Names =====
    STAGES = {
        "SCRIPT": "script_generation",
        "STT": "stt_transcription",
        "STT_RAW": "stt_raw_transcription",
        "SENTENCE_SPLIT": "sentence_splitting",
        "SUBTITLE_SPLIT": "subtitle_splitting",
        "KEYWORDS": "keyword_generation",
        "SPEED_ADJUST": "speed_adjustment",
        "ALIGNMENT": "alignment_process",
        "AUDIO_MERGE": "audio_merging",
        "CAPTIONING": "captioning_process",
        "PEXELS_SEARCH": "pexels_search",
        "VIDEO_EXPORT": "video_export",
    }
    
    # ===== Metric Weights =====
    STAGE_METRICS_WEIGHTS = {
        "script_generation": {
            "relevance": 0.35,
            "structure": 0.25,
            "tone_of_voice": 0.2,
            "length_constraint": 0.2,
        },
        "stt_transcription": {
            "word_error_rate": 0.5,
            "punctuation_capitalization": 0.2,
            "timestamp_accuracy": 0.3,
        },
        "stt_raw_transcription": {
            "word_error_rate": 0.5,
            "punctuation_capitalization": 0.2,
            "timestamp_accuracy": 0.3,
        },
        "voice_splitting": {
            "semantic_completeness": 0.4,
            "duration_balance": 0.3,
            "natural_pause": 0.3,
        },
        "subtitle_splitting": {
            "readability": 0.35,
            "synchronization": 0.4,
            "line_break_logic": 0.25,
        },
        "keyword_generation": {
            "visual_relevance": 0.45,
            "searchability": 0.3,
            "diversity": 0.25,
        },
    }
    
    @classmethod
    def get_threshold(cls, metric_name: str) -> float:
        """Lấy threshold cho metric"""
        return cls.METRIC_THRESHOLDS.get(metric_name, 0.7)
    
    @classmethod
    def get_metrics_for_stage(cls, stage: str) -> dict:
        """Lấy metrics cho stage"""
        return cls.STAGE_METRICS_WEIGHTS.get(stage, {})
    
    @classmethod
    def validate(cls):
        """Validate config"""
        print("\n⚙️  Configuration:")
        if cls.LANGFUSE_ENABLED:
            print("   ✅ Langfuse enabled")
        else:
            print("   ⚠️  Langfuse disabled (no API keys)")
        
        if cls.USE_MOCK_BACKEND:
            print("   ✅ Using mock backend")
        else:
            print(f"   ✅ Using backend: {cls.BACKEND_URL}")
        
        if cls.OPENAI_API_KEY:
            print(f"   ✅ OpenAI API configured ({cls.OPENAI_MODEL})")
        else:
            print("   ⚠️  OpenAI API key missing")
        print()
