#!/usr/bin/env python3
"""
Metrics by Stage - Define which metrics are used for each evaluation stage
Based on user's evaluation criteria for AI video generation system
"""

# Map each stage to its specific metrics
STAGE_METRICS = {
    "script_generation": [
        "relevance",           # JD alignment
        "structure",           # Hook + Body + CTA
        "tone_of_voice",       # Company culture fit
        "length_constraint"    # 30-60s video duration
    ],
    "stt_transcription": [
        "word_error_rate",    # WER score
        "punctuation_capitalization",  # Auto punctuation & caps
        "timestamp_accuracy"   # Timestamp structural consistency
    ],
    "stt_raw_transcription": [
        "word_error_rate",    # WER score on raw transcript before alignment
        "punctuation_capitalization",  # Auto punctuation & caps
        "timestamp_accuracy"   # Timestamp structural consistency
    ],
    "voice_splitting": [
        "semantic_completeness",  # Phrase integrity
        "duration_balance",       # Usable segment lengths
        "natural_pause"           # Natural break points
    ],
    "subtitle_splitting": [
        "readability",         # Chars/line + CPS + layout
        "synchronization",     # Timestamp and coverage consistency
        "line_break_logic"     # Visual formatting
    ],
    "keyword_generation": [
        "visual_relevance",    # Image search accuracy
        "searchability",       # Stock library results
        "diversity"            # Keyword variety
    ],
    "image_search_generation": [
        "visual_relevance",    # Image search accuracy
        "searchability",       # Stock library results
        "diversity"            # Keyword variety
    ],
    "video_search_generation": [
        "visual_relevance",    # Image search accuracy
        "searchability",       # Stock library results
        "diversity"            # Keyword variety
    ]
}

def get_stage_metrics(stage: str):
    """Get list of metrics for a stage"""
    return STAGE_METRICS.get(stage, [])

# Metric descriptions for documentation
METRIC_DESCRIPTIONS = {
    # Script Generation
    "relevance": "How well the script aligns with JD requirements (location, qualifications, benefits)",
    "structure": "Presence of Hook (attention), Body (info), and CTA (call-to-action)",
    "tone_of_voice": "Alignment with company culture (professional, youthful, formal)",
    "length_constraint": "Script length fit for 30-60s video duration",

    # STT Transcription
    "word_error_rate": "Error rate in transcription (missing, extra, or wrong words)",
    "punctuation_capitalization": "How well STT restores punctuation, sentence breaks, and capitalization",
    "timestamp_accuracy": "Structural quality of word timestamps (ordering, duration, and token alignment)",
    
    # Voice Splitting
    "semantic_completeness": "Phrases remain meaningful when split (not cut mid-phrase)",
    "duration_balance": "Segments stay within usable duration/length ranges without extreme imbalance",
    "natural_pause": "Sentence breaks create natural pause points when reassembled",
    
    # Subtitle Splitting
    "readability": "Subtitle readability based on chars per line, CPS, and line count",
    "synchronization": "Subtitle timing consistency and source-text coverage",
    "line_break_logic": "Line breaks avoid weak split points and orphan fragments",
    
    # Keyword Generation
    "visual_relevance": "Keywords reflect correct search image subject (e.g., 'Professional Developer')",
    "searchability": "Keywords are common enough for good stock library results",
    "diversity": "Keywords vary across scenes (not repetitive)"
}
