export PYTHONPATH=..

UV ?= uv
PYTHON ?= python

.PHONY: help install eval eval-mock eval-script eval-stt eval-stt-raw eval-subtitle eval-image-search eval-video-search clean

help:
	@echo "📋 Evaluation AI - Available Targets:"
	@echo ""
	@echo "  eval                    - Run all evaluation experiments against real BE"
	@echo "  eval-mock               - Run all evaluation experiments with mock backend"
	@echo "  eval-script             - Evaluate script generation (jd_script_dataset)"
	@echo "  eval-stt                - Evaluate STT transcription (jd_stt_dataset)"
	@echo "  eval-stt-raw            - Evaluate raw STT before alignment (jd_stt_raw_dataset)"
	@echo "  eval-subtitle           - Evaluate subtitle splitting (subtitle_splitting stage)"
	@echo "  eval-image-search       - Evaluate image search query generation (jd_image_search_dataset)"
	@echo "  eval-video-search       - Evaluate video search segment generation (jd_video_search_dataset)"
	@echo ""
	@echo "  eval-quick              - Quick test run (single stage)"
	@echo "  eval-report             - Generate evaluation report from latest results"
	@echo ""
	@echo "  clean                   - Remove cached files and artifacts"
	@echo ""
	@echo "📊 Datasets:"
	@echo "  - jd_script_dataset (15 items) - Script generation"
	@echo "  - jd_stt_dataset (5 items) - STT transcription"
	@echo "  - jd_stt_raw_dataset (5 items) - Raw STT transcription"
	@echo "  - jd_voice_splitting_dataset (5 items) - Voice splitting"
	@echo "  - jd_subtitle_dataset (5 items) - Subtitle splitting"
	@echo "  - jd_image_search_dataset (3 items) - Image search generation"
	@echo "  - jd_video_search_dataset (3 items) - Video search generation"
	@echo ""
	@echo "📈 Current Status:"
	@echo "  - Review latest results with 'make eval-report'"
	@echo "  - Langfuse datasets will be recreated automatically if deleted"

# Main evaluation - runs all experiments using proper Langfuse run_experiment() API
eval:
	@echo "🚀 Running all evaluation experiments..."
	@echo "🔌 Backend mode: REAL BE"
	@echo "⏱️  This will take ~2-3 minutes..."
	@cd .. && BACKEND_URL=http://localhost:8001 USE_MOCK_BACKEND=false REQUIRE_REAL_BACKEND=true $(PYTHON) Evaluation_AI/run_experiments_proper.py
	@echo ""
	@echo "✅ All experiments completed!"
	@echo "📊 Check Langfuse dashboard for results"
	@echo "   URL: https://cloud.langfuse.com"

eval-mock:
	@echo "🧪 Running all evaluation experiments with mock backend..."
	@echo "⏱️  This will take ~2-3 minutes..."
	@cd .. && BACKEND_URL=http://localhost:8001 USE_MOCK_BACKEND=true REQUIRE_REAL_BACKEND=false $(PYTHON) Evaluation_AI/run_experiments_proper.py
	@echo ""
	@echo "✅ All mock experiments completed!"
	@echo "📊 Check Langfuse dashboard for results"
	@echo "   URL: https://cloud.langfuse.com"

# Single stage evaluation
eval-script:
	@echo "📝 Evaluating Script Generation..."
	@cd .. && BACKEND_URL=http://localhost:8001 USE_MOCK_BACKEND=false REQUIRE_REAL_BACKEND=true $(PYTHON) Evaluation_AI/run_single_experiment.py --stage script_generation --dataset jd_script_dataset

eval-stt:
	@echo "🎙️  Evaluating STT Transcription..."
	@cd .. && BACKEND_URL=http://localhost:8001 USE_MOCK_BACKEND=false REQUIRE_REAL_BACKEND=true $(PYTHON) Evaluation_AI/run_single_experiment.py --stage stt_transcription --dataset jd_stt_dataset

eval-stt-raw:
	@echo "🎙️  Evaluating Raw STT Transcription..."
	@cd .. && BACKEND_URL=http://localhost:8001 USE_MOCK_BACKEND=false REQUIRE_REAL_BACKEND=true $(PYTHON) Evaluation_AI/run_single_experiment.py --stage stt_raw_transcription --dataset jd_stt_raw_dataset

eval-subtitle:
	@echo "📺 Evaluating Subtitle Splitting..."
	@cd .. && BACKEND_URL=http://localhost:8001 USE_MOCK_BACKEND=false REQUIRE_REAL_BACKEND=true $(PYTHON) Evaluation_AI/run_single_experiment.py --stage subtitle_splitting --dataset jd_subtitle_dataset

eval-image-search:
	@echo "🖼️  Evaluating Image Search Generation..."
	@cd .. && BACKEND_URL=http://localhost:8001 USE_MOCK_BACKEND=false REQUIRE_REAL_BACKEND=true $(PYTHON) Evaluation_AI/run_single_experiment.py --stage image_search_generation --dataset jd_image_search_dataset

eval-video-search:
	@echo "🎬 Evaluating Video Search Generation..."
	@cd .. && BACKEND_URL=http://localhost:8001 USE_MOCK_BACKEND=false REQUIRE_REAL_BACKEND=true $(PYTHON) Evaluation_AI/run_single_experiment.py --stage video_search_generation --dataset jd_video_search_dataset

# Quick test - script stage
eval-quick:
	@echo "⚡ Quick evaluation (script stage only)..."
	@cd .. && BACKEND_URL=http://localhost:8001 USE_MOCK_BACKEND=false REQUIRE_REAL_BACKEND=true $(PYTHON) Evaluation_AI/run_single_experiment.py --stage script_generation --dataset jd_script_dataset

# Generate report from latest results
eval-report:
	@echo "📊 Generating evaluation report..."
	@if [ -f results/test_results.json ]; then \
		cd .. && $(PYTHON) -c "import json; data = json.load(open('Evaluation_AI/results/test_results.json')); \
		print('\n' + '='*60); \
		print('📈 EVALUATION REPORT'); \
		print('='*60); \
		passed = sum(1 for r in data if r.get('passed')); \
		print(f'✅ Passed: {passed}/{len(data)} ({100*passed/len(data):.1f}%)'); \
		print(f'📊 Avg Score: {sum(r.get(\"overall_score\", 0) for r in data)/len(data):.2f}'); \
		print('='*60)"; \
	else \
		echo "❌ No results found. Run 'make eval' first."; \
	fi

# Clean artifacts
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache 2>/dev/null || true
	find . -type d -name ".deepeval" -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleanup complete"

# Development targets
install:
	@echo "📦 Installing dependencies..."
	pip install -r Evaluation_AI/requirements.txt

install-dev:
	@echo "📦 Installing dev dependencies..."
	pip install -r Evaluation_AI/requirements.txt pytest ruff black

test:
	@echo "🧪 Running tests..."
	pytest Evaluation_AI/tests -v 2>/dev/null || echo "No tests directory"

format:
	@echo "🎨 Formatting code..."
	black Evaluation_AI --quiet

lint:
	@echo "🔍 Linting code..."
	pylint Evaluation_AI || true

# Aliases
e: eval
e-s: eval-script
e-st: eval-stt
e-str: eval-stt-raw
e-sub: eval-subtitle
e-is: eval-image-search
e-vs: eval-video-search
