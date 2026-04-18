#!/usr/bin/env python3
"""
Register Langfuse scoring functions for metrics
Maps internal evaluation results to Langfuse dashboard scores
"""

def score_relevance(output, expected):
    """Script relevance to JD"""
    if isinstance(output, dict):
        return output.get("relevance", 0.0)
    return 0.0

def score_structure(output, expected):
    """Script structure (Hook + Body + CTA)"""
    if isinstance(output, dict):
        return output.get("structure", 0.0)
    return 0.0

def score_tone_of_voice(output, expected):
    """Tone alignment with company culture"""
    if isinstance(output, dict):
        return output.get("tone_of_voice", 0.0)
    return 0.0

def score_length_constraint(output, expected):
    """Script length fit for video duration"""
    if isinstance(output, dict):
        return output.get("length_constraint", 0.0)
    return 0.0

def score_word_error_rate(output, expected):
    """STT Word Error Rate"""
    if isinstance(output, dict):
        return output.get("word_error_rate", 0.0)
    return 0.0

def score_punctuation_capitalization(output, expected):
    """STT Punctuation & Capitalization"""
    if isinstance(output, dict):
        return output.get("punctuation_capitalization", 0.0)
    return 0.0

def score_timestamp_accuracy(output, expected):
    """STT Timestamp accuracy"""
    if isinstance(output, dict):
        return output.get("timestamp_accuracy", 0.0)
    return 0.0

def score_semantic_completeness(output, expected):
    """Voice splitting semantic completeness"""
    if isinstance(output, dict):
        return output.get("semantic_completeness", 0.0)
    return 0.0

def score_duration_balance(output, expected):
    """Voice splitting duration balance"""
    if isinstance(output, dict):
        return output.get("duration_balance", 0.0)
    return 0.0

def score_natural_pause(output, expected):
    """Voice splitting natural pause points"""
    if isinstance(output, dict):
        return output.get("natural_pause", 0.0)
    return 0.0

def score_readability(output, expected):
    """Subtitle readability (words per line)"""
    if isinstance(output, dict):
        return output.get("readability", 0.0)
    return 0.0

def score_synchronization(output, expected):
    """Subtitle synchronization with voice"""
    if isinstance(output, dict):
        return output.get("synchronization", 0.0)
    return 0.0

def score_line_break_logic(output, expected):
    """Subtitle line break logic"""
    if isinstance(output, dict):
        return output.get("line_break_logic", 0.0)
    return 0.0

def score_visual_relevance(output, expected):
    """Keyword visual relevance"""
    if isinstance(output, dict):
        return output.get("visual_relevance", 0.0)
    return 0.0

def score_searchability(output, expected):
    """Keyword searchability in stock libraries"""
    if isinstance(output, dict):
        return output.get("searchability", 0.0)
    return 0.0

def score_diversity(output, expected):
    """Keyword diversity across scenes"""
    if isinstance(output, dict):
        return output.get("diversity", 0.0)
    return 0.0

# This list defines which scores are ACTIVE on the dashboard
# Each stage will only create these specific charts
evaluators = [
    # Script Generation (4 metrics)
    {"name": "relevance", "score_func": score_relevance},
    {"name": "structure", "score_func": score_structure},
    {"name": "tone_of_voice", "score_func": score_tone_of_voice},
    {"name": "length_constraint", "score_func": score_length_constraint},
    
    # STT Transcription (3 metrics)
    {"name": "word_error_rate", "score_func": score_word_error_rate},
    {"name": "punctuation_capitalization", "score_func": score_punctuation_capitalization},
    {"name": "timestamp_accuracy", "score_func": score_timestamp_accuracy},
    
    # Voice Splitting (3 metrics)
    {"name": "semantic_completeness", "score_func": score_semantic_completeness},
    {"name": "duration_balance", "score_func": score_duration_balance},
    {"name": "natural_pause", "score_func": score_natural_pause},
    
    # Subtitle Splitting (3 metrics)
    {"name": "readability", "score_func": score_readability},
    {"name": "synchronization", "score_func": score_synchronization},
    {"name": "line_break_logic", "score_func": score_line_break_logic},
    
    # Keyword Generation (3 metrics)
    {"name": "visual_relevance", "score_func": score_visual_relevance},
    {"name": "searchability", "score_func": score_searchability},
    {"name": "diversity", "score_func": score_diversity},
]
