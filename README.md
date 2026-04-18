# Evaluation AI

`Evaluation_AI` là hệ thống đánh giá chất lượng cho các thành phần AI đang được sử dụng trong `BE` của dự án ShortJD. Mục tiêu của module này không phải tạo ra output mẫu, mà là lấy `input` từ dataset, gọi đúng processor hoặc API của `BE`, lấy `output` runtime thực tế, rồi chấm `output` đó theo các tiêu chí đã định nghĩa.

Hệ thống hiện tập trung vào 7 nhóm đánh giá:
- `script_generation`
- `stt_transcription`
- `stt_raw_transcription`
- `voice_splitting`
- `subtitle_splitting`
- `image_search_generation`
- `video_search_generation`

Điểm số của mọi metric đều được chuẩn hóa trong khoảng `0.0 -> 1.0` để thuận tiện cho Langfuse, dashboard so sánh thí nghiệm và các báo cáo tổng hợp.

## Mục tiêu

`Evaluation_AI` được thiết kế để giải quyết các nhu cầu sau:
- đánh giá chất lượng AI theo đúng artifact mà `BE` đang dùng trong pipeline thật
- so sánh kết quả giữa các lần thay đổi prompt, model, template hoặc logic xử lý
- đưa dữ liệu chấm điểm lên Langfuse để theo dõi theo dataset, theo experiment và theo metric
- tách riêng metric deterministic với metric LLM-as-a-judge để dễ kiểm soát độ ổn định

## Phạm vi hiện tại

Các stage đang active trong hệ thống:

| Stage | Mục tiêu | Artifact được chấm |
|---|---|---|
| `script_generation` | Chấm chất lượng kịch bản tuyển dụng | Script do `BE` sinh ra |
| `stt_transcription` | Chấm transcript sau bước align của `BE` | Transcript đã align về script |
| `stt_raw_transcription` | Chấm transcript thô trước align | Raw transcript |
| `voice_splitting` | Chấm cách chia script thành segment voice | Danh sách `segments` |
| `subtitle_splitting` | Chấm chất lượng chia subtitle | Danh sách caption `text/start/end` |
| `image_search_generation` | Chấm truy vấn tìm ảnh stock | Danh sách `image_queries` |
| `video_search_generation` | Chấm truy vấn tìm video theo phân cảnh | Danh sách `video_segments` |

`voice_generation` không còn nằm trong pipeline eval hiện tại vì repo này không đánh giá trực tiếp artifact audio waveform.

## Nguyên tắc đánh giá

Hệ thống đang bám theo nguyên tắc sau:
- dataset chỉ cung cấp `input`, `criteria`, `metadata` và khi cần thì có `expected_output`
- runner dùng `input` để gọi `BE`
- `expected_output` chỉ đóng vai trò reference, không thay thế output runtime
- evaluator chấm trên output thực tế của `BE`
- nếu một stage bị lỗi trước khi tạo output hợp lệ, các metric của stage đó sẽ không có dữ liệu và có thể bị ghi `0`

Điều này rất quan trọng: một biểu đồ toàn `0` chưa chắc là chất lượng AI kém; nhiều khi đó là dấu hiệu stage bị lỗi ở tầng gọi backend.

## Kiến trúc thư mục

```text
Evaluation_AI/
├── backend/
│   └── client.py
├── core/
│   ├── base_evaluator.py
│   ├── dataset_utils.py
│   └── langfuse_manager.py
├── datasets/
│   ├── all_templates_evaluation.json
│   ├── stt_transcription_test.json
│   ├── stt_raw_transcription_test.json
│   ├── voice_splitting_test.json
│   ├── subtitle_splitting_test.json
│   ├── image_search_generation_test.json
│   └── video_search_generation_test.json
├── metrics/
│   ├── script_eval.py
│   ├── stt_eval.py
│   ├── voice_splitting_eval.py
│   ├── subtitle_eval.py
│   └── keyword_eval.py
├── runners/
│   ├── evaluation_runner.py
│   ├── langfuse_experiment_runner.py
│   └── stage_metrics.py
├── results/
├── config.py
├── Makefile
├── run_experiments_proper.py
└── run_single_experiment.py
```

Ý nghĩa từng phần:
- `datasets/`: dữ liệu test đầu vào
- `backend/client.py`: lớp gọi `BE`, ưu tiên HTTP API
- `metrics/`: toàn bộ logic chấm điểm
- `runners/`: orchestration, mapping stage, Langfuse experiment
- `results/`: file output cục bộ khi chạy local

## Luồng chạy chuẩn

Luồng chạy đúng của hệ thống như sau:

1. Load test case từ `datasets/`
2. Xác định `stage` và `criteria`
3. Gọi processor tương ứng của `BE`
4. Nhận output runtime
5. Chạy các evaluator tương ứng theo stage
6. Chuẩn hóa score về `0..1`
7. Ghi kết quả ra file local hoặc đẩy lên Langfuse

Ví dụ với `script_generation`:
- dataset cung cấp JD và metadata template
- runner resolve template thật của `BE`
- gọi `POST /api/v1/script/generate`
- nhận script runtime
- chấm `relevance`, `structure`, `tone_of_voice`, `length_constraint`

Ví dụ với `voice_splitting`:
- dataset cung cấp `script_text`
- runner gọi `POST /api/v1/voice/split`
- nhận `{"segments": [...]}`
- chấm `semantic_completeness`, `duration_balance`, `natural_pause`

Ví dụ với `subtitle_splitting`:
- dataset cung cấp `text` và khi có thì có `word_timings`
- runner gọi `POST /api/v1/voice/subtitles/split`
- nhận danh sách captions
- chấm `readability`, `synchronization`, `line_break_logic`

## Cách chấm điểm hiện tại

Hệ thống dùng hai loại evaluator:

| Loại | Đặc điểm | Ví dụ |
|---|---|---|
| Deterministic | Tính bằng code, ổn định, dễ giải thích | `word_error_rate`, `duration_balance`, `readability` |
| LLM-as-a-Judge | Dùng LLM để chấm tiêu chí ngữ nghĩa hoặc văn phong | `relevance`, `tone_of_voice`, `semantic_completeness`, `visual_relevance` |

Một số metric judge dùng rubric nội bộ `1-5`, sau đó normalize về `0-1`. Một số metric deterministic tính trực tiếp ra score `0-1`.

## Metric theo từng stage

### 1. `script_generation`

| Metric | Ý nghĩa |
|---|---|
| `relevance` | Kịch bản có bám JD, vị trí, quyền lợi, yêu cầu, địa điểm, cấp độ không |
| `structure` | Kịch bản có đủ Hook, Body, CTA theo cấu trúc mong muốn không |
| `tone_of_voice` | Giọng văn có phù hợp với ngữ cảnh tuyển dụng và đối tượng ứng viên không |
| `length_constraint` | Độ dài có phù hợp với thời lượng video ngắn không |

### 2. `stt_transcription`

Đây là mode chấm artifact cuối cùng mà `BE` dùng sau bước align.

| Metric | Ý nghĩa |
|---|---|
| `word_error_rate` | Score = `1 - WER` |
| `punctuation_capitalization` | Dấu câu, viết hoa, ngắt câu |
| `timestamp_accuracy` | Chất lượng cấu trúc timestamp theo từ |

### 3. `stt_raw_transcription`

Đây là mode chấm transcript thô trước align.

| Metric | Ý nghĩa |
|---|---|
| `word_error_rate` | Score = `1 - WER` trên raw transcript |
| `punctuation_capitalization` | Dấu câu, viết hoa, ngắt câu |
| `timestamp_accuracy` | Chất lượng cấu trúc timestamp theo từ |

### 4. `voice_splitting`

| Metric | Ý nghĩa |
|---|---|
| `semantic_completeness` | Segment có giữ trọn ý không |
| `duration_balance` | Các segment có độ dài dùng được và không lệch quá mạnh không |
| `natural_pause` | Điểm ngắt có tự nhiên khi đọc không |

### 5. `subtitle_splitting`

| Metric | Ý nghĩa |
|---|---|
| `readability` | Dễ đọc theo chars/line, CPS, layout |
| `synchronization` | Tính nhất quán timing và coverage text |
| `line_break_logic` | Tránh orphan line, tránh ngắt ở điểm yếu |

### 6. `image_search_generation`

| Metric | Ý nghĩa |
|---|---|
| `visual_relevance` | Truy vấn ảnh có phản ánh đúng hình ảnh cần tìm không |
| `searchability` | Truy vấn ảnh có khả năng cho ra kết quả stock tốt không |
| `diversity` | Các truy vấn ảnh có đa dạng không |

### 7. `video_search_generation`

| Metric | Ý nghĩa |
|---|---|
| `visual_relevance` | Query video theo từng segment có phản ánh đúng cảnh cần tìm không |
| `searchability` | Query video có đủ cụ thể để tìm stock footage tốt không |
| `diversity` | Query giữa các segment có đa dạng không |

## Cấu hình chính

Các cấu hình quan trọng nằm ở [config.py](/home/tranhao/MBW/short_jd_ver2/Evaluation_AI/config.py):
- `BACKEND_URL`: mặc định `http://localhost:8001`
- `USE_MOCK_BACKEND`: bật/tắt mock backend
- `REQUIRE_REAL_BACKEND`: nếu `true`, lỗi backend sẽ không được che bằng output giả
- `OPENAI_MODEL`: model dùng cho LLM judge
- `METRIC_THRESHOLDS`: threshold pass/fail cho từng metric
- `STAGE_METRICS_WEIGHTS`: trọng số tính overall score

Mặc định hiện tại:
- `make eval` chạy với `USE_MOCK_BACKEND=false`
- `make eval` yêu cầu gọi `BE` thật ở `localhost:8001`

## Yêu cầu trước khi chạy

### 1. Chạy `BE`

`Evaluation_AI` được thiết kế để đánh giá trực tiếp `BE`, nên trước khi chạy cần đảm bảo `BE` đang lên.

Ví dụ:

```bash
cd /home/tranhao/MBW/short_jd_ver2/BE
python main.py
```

Mặc định `BE` đang chạy ở:

```text
http://localhost:8001
```

### 2. Cài dependency

```bash
cd /home/tranhao/MBW/short_jd_ver2
pip install -r Evaluation_AI/requirements.txt
```

### 3. Cấu hình biến môi trường

Tối thiểu nên có:

```bash
OPENAI_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://cloud.langfuse.com
```

`Config` sẽ load cả:
- `BE/.env`
- `.env` ở root repo

## Cách chạy

### Chạy toàn bộ evaluation với BE thật

```bash
cd /home/tranhao/MBW/short_jd_ver2/Evaluation_AI
make eval
```

Lệnh này:
- load toàn bộ dataset đang active
- gọi `BE` thật
- ghi trace và experiment lên Langfuse

### Chạy bằng mock backend

```bash
make eval-mock
```

Chế độ này chỉ dùng để kiểm tra framework eval, không dùng để kết luận chất lượng AI thật của `BE`.

### Chạy từng stage

```bash
make eval-script
make eval-stt
make eval-stt-raw
make eval-subtitle
make eval-image-search
make eval-video-search
```

`voice_splitting` hiện chưa có target `make` riêng. Có thể chạy bằng:

```bash
cd /home/tranhao/MBW/short_jd_ver2
python Evaluation_AI/run_single_experiment.py --stage voice_splitting
```

### Chạy một stage cục bộ, không cần Langfuse experiment

```bash
cd /home/tranhao/MBW/short_jd_ver2
python Evaluation_AI/run_single_experiment.py --stage subtitle_splitting
```

### Chạy một stage qua Langfuse experiment API

```bash
cd /home/tranhao/MBW/short_jd_ver2
python Evaluation_AI/run_single_experiment.py \
  --stage subtitle_splitting \
  --dataset jd_subtitle_dataset \
  --langfuse
```

## Dataset schema

Một test case chuẩn có dạng:

```json
{
  "id": "SCRIPT-001",
  "name": "Template 1 - Knowledge Sharing",
  "stage": "script_generation",
  "input": {
    "jd_content": "..."
  },
  "criteria": [
    "relevance",
    "structure",
    "tone_of_voice",
    "length_constraint"
  ],
  "metadata": {
    "be_template_name": "Chia sẻ kiến thức",
    "category": "knowledge_sharing"
  },
  "expected_output": {
    "full_script": "..."
  }
}
```

Ý nghĩa các field:
- `id`: mã test case ổn định
- `name`: tên dễ đọc trên Langfuse
- `stage`: stage cần đánh giá
- `input`: dữ liệu đưa vào processor
- `criteria`: metric cần chấm
- `metadata`: context bổ sung cho runner và evaluator
- `expected_output`: reference output, không phải output runtime

### Lưu ý theo từng stage

`script_generation`
- nên có `metadata.be_template_name` hoặc `metadata.be_template_id`
- không nên dựa vào `template_id` legacy nếu có thể khai báo template thật của `BE`

`voice_splitting`
- `expected_output` nên có dạng:

```json
{
  "segments": ["...", "..."]
}
```

`subtitle_splitting`
- `expected_output` nên là danh sách caption reference:

```json
[
  {"text": "..."},
  {"text": "..."}
]
```

`stt_transcription`
- nếu muốn chấm đúng artifact pipeline, nên để runner đi qua luồng voice/transcription/alignment của `BE`

`image_search_generation`
- `expected_output` nên có dạng:

```json
{
  "image_queries": [
    {"timestamp": 1.0, "query": "modern office"}
  ]
}
```

`video_search_generation`
- `expected_output` nên có dạng:

```json
{
  "video_segments": [
    {
      "time_range": [0.0, 4.5],
      "queries": ["busy street", "delivery truck", "urban traffic"]
    }
  ]
}
```

### Điều không nên làm

- không lưu `expected_metrics` kiểu số giả định trong dataset
- không dùng `expected_output` thay thế cho output runtime của backend
- không dùng mock backend để kết luận chất lượng thật của hệ thống production

## Dataset đang có

| Dataset file | Stage | Mục đích |
|---|---|---|
| `all_templates_evaluation.json` | `script_generation` | Chấm script trên nhiều template |
| `stt_transcription_test.json` | `stt_transcription` | Chấm transcript sau align |
| `stt_raw_transcription_test.json` | `stt_raw_transcription` | Chấm raw transcript |
| `voice_splitting_test.json` | `voice_splitting` | Chấm split voice |
| `subtitle_splitting_test.json` | `subtitle_splitting` | Chấm subtitle split |
| `image_search_generation_test.json` | `image_search_generation` | Chấm image search query generation |
| `video_search_generation_test.json` | `video_search_generation` | Chấm video search segment generation |

## Tích hợp Langfuse

Langfuse được dùng cho 3 việc:
- quản lý dataset item
- tạo experiment cho từng lần chạy
- ghi item-level metric và run-level average metric

Tên dataset Langfuse hiện tại:
- `jd_script_dataset`
- `jd_stt_dataset`
- `jd_stt_raw_dataset`
- `jd_voice_splitting_dataset`
- `jd_subtitle_dataset`
- `jd_image_search_dataset`
- `jd_video_search_dataset`

Khi chạy `make eval`, hệ thống sẽ:
1. tạo hoặc nạp dataset trên Langfuse
2. đồng bộ item từ dataset JSON
3. chạy `task_fn` cho từng item
4. ghi metric theo đúng stage
5. ghi thêm average metric ở run-level

## Kết quả đầu ra cục bộ

Khi chạy local, kết quả được ghi vào:

```text
Evaluation_AI/results/test_results.json
```

Mỗi phần tử có dạng tổng quát:

```json
{
  "test_case_id": "SUB-T1-001",
  "test_case_name": "Template 1 Subtitle - AI Engineer",
  "stage": "subtitle_splitting",
  "passed": true,
  "overall_score": 0.84,
  "turn_results": [
    {
      "turn_index": 0,
      "metrics": {
        "readability": 0.88,
        "synchronization": 0.81,
        "line_break_logic": 0.83
      }
    }
  ]
}
```

## Các vấn đề hay gặp

### 1. Biểu đồ Langfuse toàn `0`

Các nguyên nhân thường gặp:
- stage fail trước khi tạo output hợp lệ
- backend trả response sai shape
- evaluator không tìm thấy field cần thiết như `segments`, `captions`, `word_timestamps`

Ví dụ thực tế đã gặp:
- `voice_splitting` về `0` khi client import local service và chết vì thiếu `edge_tts`
- `subtitle_splitting` về `0` vì stage lỗi trước khi tạo captions
- `script_generation.structure` về `0` khi evaluator đọc sai shape `{full_script: ...}`

### 2. Langfuse warning metadata quá dài

Hệ thống đã tách:
- metadata đầy đủ để lưu trên dataset item
- metadata rút gọn để propagate trong experiment trace

Nếu vẫn thấy warning cũ, thường là dataset item trên Langfuse còn đang giữ schema cũ từ lần chạy trước.

### 3. Chạy mock cho ra kết quả đẹp nhưng real BE lại fail

Điều này là bình thường nếu môi trường `BE` thật thiếu dependency, endpoint chưa lên, hoặc response contract khác mock. Vì vậy cần phân biệt rõ:
- `make eval-mock`: kiểm tra framework eval
- `make eval`: đánh giá thật `BE`

## Lưu ý vận hành

- khi sửa contract response của `BE`, cần kiểm tra lại evaluator tương ứng
- khi thêm stage mới, cần cập nhật cả `DatasetLoader`, `stage_metrics`, `EvaluationRunner`, `LangfuseExperimentRunner`, dataset file và metric evaluators
- với script dataset, nên ưu tiên lưu `be_template_name` để map đúng template thật trong DB của `BE`
- với stage subtitle hoặc voice splitting, output phải bám đúng artifact pipeline, không nên chấm trên heuristic text rời rạc nếu mục tiêu là đánh giá production behavior

## Lệnh hữu ích

```bash
# Chạy toàn bộ
make eval

# Chạy mock
make eval-mock

# Chạy một stage local
python Evaluation_AI/run_single_experiment.py --stage script_generation

# Chạy một stage qua Langfuse
python Evaluation_AI/run_single_experiment.py \
  --stage script_generation \
  --dataset jd_script_dataset \
  --langfuse

# Xem report từ file local
make eval-report
```

## Kết luận

`Evaluation_AI` là lớp đánh giá trung gian giữa dataset test và AI runtime của `BE`. Giá trị chính của hệ thống nằm ở việc chấm đúng artifact thật mà pipeline đang sử dụng, thay vì chấm trên output giả hoặc reference output cố định. Khi dùng đúng cách, module này giúp bạn trả lời được ba câu hỏi quan trọng:

- thay đổi trong `BE` có làm chất lượng AI tốt hơn hay kém đi không
- stage nào đang hỏng do backend execution, stage nào đang kém thật về chất lượng
- metric nào đang fail theo từng dataset, từng template và từng lần chạy experiment
