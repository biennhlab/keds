# KEDs CC3M 500K · Composed Image Retrieval

Hệ thống nhận **reference image + modification text**, chạy CLIP ViT-L/14,
truy xuất knowledge CC3M 500K, tạo pseudo-words bằng Phi-M/Phi-A và trả về
Top-K ảnh thật từ **target gallery riêng** qua FAISS. Có Python API, CLI,
FastAPI và Gradio. 
**Advanced Multimodal RAG** đã được thêm dưới dạng profile opt-in: query-aware
hybrid/RRF knowledge retrieval, knowledge MMR/reranking, adaptive K/alpha, gallery
reranking và versioned semantic/exact caches.

```powershell
.venv/Scripts/python.exe scripts/search.py --image data/demo/dog.jpg --text "with two dogs instead of one" --gallery artifacts/gallery/demo --rag-profile combined_experimental --top-k 3
.venv/Scripts/python.exe scripts/profile_rag.py --repeats 2
# Gradio: chọn Research settings → RAG profile; REST: field rag_profile.
```

Đây là **author-style approximation v21** được chủ dự án huấn luyện trước,
không tự nhận exact paper reproduction. Không train từ epoch 0 hoặc encode lại
CC3M. Hai BEST exports được load strict; các notebook/artifact nguồn được giữ nguyên.

## Chạy nhanh trên Windows

Cần Python **3.11–3.12**, `uv`, khoảng 12 GB disk trống cho artifacts/indexes và
đủ RAM cho CLIP + DB metadata. Cấu hình CPU dùng shard FAISS 16.384 rows để tránh
allocate đồng thời hai index 1,5 GB. GPU không bắt buộc cho demo, nhưng có ích
cho latency và cần cho training/evaluation quy mô lớn.

```powershell
uv sync --frozen --extra dev
# Tạo .env từ .env.example nếu chưa có; điền HF_TOKEN bằng trình soạn thảo.
# Giữ nguyên .env đang có. Không đưa token lên Git.
uv run --no-sync python scripts/download_artifacts.py --clip --resume
uv run --no-sync python scripts/validate_artifacts.py --build-indexes
uv run --no-sync python scripts/download_demo_images.py
uv run --no-sync python scripts/build_gallery.py --input data/demo --output artifacts/gallery/demo --batch-size 2
uv run --no-sync python scripts/search.py --image data/demo/dog.jpg --text "with two dogs instead of one" --gallery artifacts/gallery/demo --top-k 3
uv run --no-sync python scripts/run_demo.py
```

Mở **http://127.0.0.1:8000/ui/** để upload ảnh, nhập modification tiếng Anh,
chọn gallery, số kết quả, so sánh CLIP fusion/KEDs hoặc xem neighbors/model.
Swagger: **http://127.0.0.1:8000/docs**. Liveness: `/health`; readiness:
`/ready` chỉ trả 200 khi model và một gallery hợp lệ đã sẵn sàng.

Trong workspace đã cài `.venv`, có thể thay `uv run --no-sync python` bằng
`.venv\Scripts\python.exe`. Các lệnh tương tự chạy được trên Linux.
Chạy download/validate trước serving; không thay file weights đang được mmap.

## Artifacts, kiến trúc và cấu hình

`scripts/download_artifacts.py` đọc **HF_TOKEN từ `.env`**, fallback sang env var
khi chạy container. Token không được ghi vào logs/reports. Kho private v21 cần
token có quyền đọc. File LFS được so SHA-256 từ remote và tái sử dụng nếu hợp lệ;
file khác checksum được giữ nguyên và công cụ dừng với lỗi rõ ràng.

| Artifact | Nguồn đã khóa |
|---|---|
| Knowledge DB, 500.000 cặp unit-normalized `[500000,768]` | `nightfurry/keds-500k-training-resume@f14b9e3568e8e57fe522c7cf3397dc973840d632` |
| Phi-M/Phi-A BEST + LAST | `nightfurry/keds-500k-resume-training-v21@42dcde4c9b437e1eac8978c7bed9de117178b12b` |
| CLIP/tokenizer | `openai/clip-vit-large-patch14@32bd64288804d66eefd0ccbe215aa642df71cc41` |

Đường dẫn, device, alpha, K và prompt ở `configs/model.yaml`; đường dẫn tương đối
được tính từ thư mục cha của `configs/`. `device: auto` chọn CUDA nếu có, CPU nếu
không. Lockfile dùng Windows PyPI CPU wheel và Linux CPU index; máy GPU cần thay torch bằng build CUDA
tương thích PyTorch 2.6 cho driver của máy, sau đó dùng `--no-sync` để giữ build đó.

`author_bkp_v1`: IM2TEXT 768→512→512→768 và hai CrossFormer 3 lớp, 8 heads × 64.
Thứ tự tokens theo checkpoint/official code: visual knowledge, text knowledge,
mapped reference. Không dùng SEARLE hoặc MHA/GELU/LayerNorm của repo cũ.
Placeholder được thay trực tiếp trong token embeddings, giữ 77-token context và
dịch EOT thêm 2. Câu quá dài bị từ chối; không silently truncate.

Knowledge dùng hai phép exact retrieval độc lập image→image và image→text.
Mỗi shard là `IndexFlatIP`; merge global Top-K giữ cosine ranking. Windows dùng
FAISS worker riêng vì wheel FAISS và torch có OpenMP runtimes khác nhau; Linux
dùng FAISS trực tiếp. Không cần `KMP_DUPLICATE_LIB_OK`.

Validator kiểm tra count/shape/dtype, NaN/Inf/zero vectors, strict state dict,
optimizer momentum shapes, hashes và toàn bộ thứ tự FAISS vectors. Kết quả tại
`reports/artifact_validation_report.json`. BEST v21 không tự chứa epoch; selection
metadata nằm trong remote `best_metrics.json`, không phải benchmark chạy local.

## Gallery và Python API

```powershell
uv run --no-sync python scripts/build_gallery.py --input D:/photos --output artifacts/gallery/my-gallery
uv run --no-sync python scripts/validate_gallery.py --gallery artifacts/gallery/my-gallery
```

Mỗi gallery có `images/`, các `generations/<version>/` chứa `gallery.faiss`,
`gallery_embeddings.npy`, `gallery_metadata.jsonl`, `gallery_manifest.json`, và
`CURRENT.json` là commit pointer atomic. Cập nhật giữ nguyên ảnh gốc và generations
cũ, đồng thời tránh lệch index/metadata. Append chỉ chấp nhận cùng CLIP revision.
Ảnh trùng được phát hiện bằng hash pixels; custom IDs mặc định lấy từ hash.
Benchmark phải truyền `--ids` là official split JSON để giữ IDs nguyên bản.

```python
from keds.inference import KEDsEngine

engine = KEDsEngine("configs/model.yaml")
engine.load_gallery("artifacts/gallery/demo")
query = engine.encode_query("data/demo/dog.jpg", "with two dogs instead of one")  # [768]
result = engine.retrieve("data/demo/dog.jpg", "with two dogs instead of one", top_k=3)
```

Model load một lần cho process. Inference độc lập FastAPI/Gradio, dùng
`torch.inference_mode()` và eval. Online loại reference bằng ID/hash khi phù hợp;
benchmark áp dụng riêng đúng protocol. CC3M neighbors không bị trả nhầm như gallery.

## REST API

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/search -F "image=@data/demo/dog.jpg" -F "modification=with two dogs instead of one" -F "gallery_id=demo" -F "top_k=3"
```

| Route | Chức năng |
|---|---|
| `GET /health`, `GET /ready` | Liveness/readiness |
| `GET /api/v1/model/info` | Versions, hashes, architecture, device, K, alpha |
| `GET /api/v1/gallery` | Danh sách gallery/version/count |
| `POST /api/v1/search` | Multipart image, modification, gallery_id, top_k; optional mode/alpha/knowledge_k/reference_id |
| `GET /api/v1/gallery/{gallery_id}/images/{image_id}` | Trả bytes ảnh thật |
| `POST /api/v1/gallery/images` | Multipart thêm ảnh hoặc tạo gallery mới |
| `DELETE /api/v1/gallery/{gallery_id}/images/{image_id}` | Bỏ ảnh khỏi active index; giữ storage/lịch sử |

Hai route quản trị yêu cầu `KEDS_ADMIN_TOKEN` trong `.env` và header
`X-Admin-Token`; bị vô hiệu nếu chưa cấu hình token. Client không được truyền
filesystem path. Upload kiểm tra bytes thật, định dạng, giới hạn 10 MB/25 MP,
text length và top_k. Model concurrency được giới hạn một request mỗi lần.
Log có versions/request ID/timestamp/timing, không lưu query text/ảnh mặc định.
Gradio có cache tạm dọn theo giờ. Chưa bật tính năng lưu lịch sử truy vấn.

## Kiểm thử

```powershell
uv run --no-sync pytest -q
$env:KEDS_TEST_ARTIFACTS="1"
uv run --no-sync pytest -q --junitxml=reports/verified_tests.xml
# Khi server demo đang chạy: exercise cả Gradio callback → REST → KEDs → ảnh
uv run --no-sync python scripts/smoke_live.py
```

Unit tests dùng tensor nhỏ để kiểm tra số học, inject/EOT, exact FAISS, ID exclusion,
checksums và optimizer/RNG roundtrip. Các fixtures số học này không được dùng làm
dữ liệu/prediction của ứng dụng. Integration opt-in chạy CLIP thật + DB500K + hai
BEST thật + sáu ảnh thật, kiểm tra determinism/regression/API và frozen CLIP backward
vào BKP. E0 oracle cố định ở `tests/fixtures/e0_regression.json`, không phụ thuộc
report có thể bị ghi lại bởi CLI. Official module/protocol parity
tests tự skip nếu chưa checkout các metadata/source trong `references/`.

## Evaluation và ablation

[Fashion-IQ](https://github.com/XiaoxiaoGuo/fashion-iq) annotations/splits nằm ở
`captions/` và `image_splits/`. [CIRR](https://www.zheyuanliu.me/CIRR/) cần nhánh
`cirr_dataset` cho annotations và ảnh NLVR2 gốc theo hướng dẫn chủ dữ liệu.
Repository không kèm full ảnh benchmark. Audit số lượng/split đã lưu ở
`reports/fashioniq_protocol_audit.json` và `reports/cirr_protocol_audit.json`.

```powershell
# Fashion-IQ: lặp cho dress, shirt, toptee với các split tương ứng
uv run --no-sync python scripts/build_gallery.py --input data/fashioniq/images --ids data/fashioniq/image_splits/split.dress.val.json --output artifacts/gallery/fashioniq/dress
uv run --no-sync python scripts/evaluate.py --dataset fashioniq --data-root data/fashioniq --gallery-root artifacts/gallery/fashioniq --output reports/fashioniq_evaluation.json
# CIRR val: --input trỏ vào img_raw/dev; --ids giữ nguyên ID của split dictionary
uv run --no-sync python scripts/build_gallery.py --input data/cirr/img_raw/dev --ids data/cirr/image_splits/split.rc2.val.json --output artifacts/gallery/cirr-val/cirr
uv run --no-sync python scripts/evaluate.py --dataset cirr --data-root data/cirr --gallery-root artifacts/gallery/cirr-val --ablation --output reports/cirr_ablation.json
```

Fashion-IQ: full gallery mỗi category, caption2 + `and` + caption1 theo official
KEDs, macro average ba categories. CIRR: full gallery, loại reference bằng ID,
global R@1/5/10/50, subset R@1/2/3, full-ranking MRR và latency P50/P95.
Test `--split test1` chỉ xuất submission JSON, không đưa ra metric thiếu labels.
`--ablation` chỉ chạy trên val: image-only/text-only/fusion/Phi-M/Phi-A/KEDs,
alpha 0/0.25/0.5/0.75/1 và K=4/8/16/32. Mỗi report ghi provenance, hashes và
diagnostics target rank/Top10; không cho phép sampled gallery trá hình full benchmark.
Chưa xác minh overlap CC3M/benchmark bằng raw image hashes do artifacts thiếu dữ liệu đó.

**Chưa có benchmark mới được công bố.** Sáu ảnh demo chỉ xác minh chức năng.

## Training / resume

`training/resume.py` kiểm tra architecture, optimizer states, epoch và RNG/scheduler
khi exact resume. `Trainer` dùng callback loss/validation rõ ràng, lưu LAST atomic,
chỉ lưu BEST khi có validation metric thật. Checkpoint mới lưu RNG, global step,
config/objective ID để lần tiếp theo có thể exact resume.

LAST v21 đã xác minh optimizer-compatible, nhưng thiếu original objective ID,
RNG, global step và scheduler. Chưa có notebook/pseudo-triplet recipe gốc để xác
nhận tiếp tục Phi-A như lần train v21; chưa chạy train tới epoch 30 trên CPU.

Lệnh sau là **Phi-M fine-tuning có định danh recipe riêng** `image_infonce_v1`
từ weights/optimizer cũ, không gọi là exact v21 resume và không ghi đè artifact cũ:

```powershell
uv run --no-sync python scripts/train.py --resume artifacts/checkpoints/phi_m_last.pt --output artifacts/runs/phi-m-continuation-01
```

Không chạy lệnh đó chỉ để serving. Cần GPU để chạy hiệu quả. `--exact` từ LAST v21
sẽ báo lỗi thay vì tự tuyên bố khôi phục RNG/scheduler không tồn tại. Muốn tiếp tục
Phi-A đúng recipe, cung cấp notebook/preprocessing gốc và gắn callback đã audit
vào `Trainer`; không thay bằng SEARLE hoặc nhãn ground truth benchmark.

## Docker

```powershell
docker build -t keds-cir .
docker run --rm -p 8000:8000 --env-file .env -v "${PWD}/artifacts:/app/artifacts" keds-cir
```

Image cài dependencies từ `uv.lock`, không bake artifacts/token vào image. Chuẩn bị
artifacts/index/gallery bằng các lệnh phía trên trước khi chạy. Container dùng CPU
torch; GPU deployment cần image/build torch phù hợp. Dockerfile đã viết nhưng
**chưa build/run ở lần triển khai này vì Docker daemon local chưa chạy**.
Chạy một worker để tránh nhân bản CLIP/knowledge metadata trong RAM.

## Cấu trúc

`src/keds/models` chứa encoder/BKP/composer/checkpoint loader;
`knowledge` chứa DB và shard retriever; `gallery` quản lý storage/FAISS;
`inference` điều phối; `api`, `training`, `evaluation` là các lớp sử dụng core.
`app/gradio_app.py` gọi REST API. `scripts/` là CLI; `tests/` và `reports/`
chứa kiểm chứng. Không cần state nằm trong notebook.

Tham khảo: [paper](https://arxiv.org/html/2403.16005v1),
[official implementation](https://github.com/suoych/KEDs),
[third-party notices](THIRD_PARTY_NOTICES.md).
