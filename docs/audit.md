# Audit và quyết định tương thích

Đối chiếu ngày 2026-11-17; giữ nguyên `plan.md` và toàn bộ file/tài nguyên ban đầu.

## Inventory

Repository ban đầu có README 6 bytes và kế hoạch. Không tìm thấy notebook/checkpoint
trong repository hoặc các thư mục dự án lân cận được kiểm tra. `../keds_production`
có CLIP wrapper, knowledge retrieval và một kiến trúc MHA/GELU/LayerNorm ba token,
nhưng Phi-A ở đó dùng pretrained SEARLE. Không nhập kiến trúc này vào family v21.
`../data` có contact sheets và bảng câu hỏi; không dùng chúng làm benchmark hay
coi contact sheet là ảnh gốc của gallery.

## Nguồn đã đối chiếu

- [KEDs paper](https://arxiv.org/html/2403.16005v1), §§3.2–3.4, 4.1–4.2.
- [Official code](https://github.com/suoych/KEDs/tree/dfe214015f68d731871a4db5e8f3b5d05680b268),
  `src/model/model.py`, `src/eval_utils.py`, `src/data.py`, `src/eval_retrieval.py`, `src/trainer.py`.
- [Source artifacts](https://huggingface.co/nightfurry/keds-500k-training-resume/tree/f14b9e3568e8e57fe522c7cf3397dc973840d632).
- [v21 artifacts](https://huggingface.co/nightfurry/keds-500k-resume-training-v21/tree/42dcde4c9b437e1eac8978c7bed9de117178b12b).
- [Fashion-IQ](https://github.com/XiaoxiaoGuo/fashion-iq/tree/f71b16c3b09ef03b02c0fed160fa072fb3b8aa16).
- [CIRR dataset branch](https://github.com/Cuberick-Orion/CIRR/tree/2276fba2d2c53862c8ae79ea18e1f1ae03719f56),
  [submission documentation](https://github.com/Cuberick-Orion/CIRR/blob/5cfd587f247c6d287900fcc4c6b3fdf81f7dab4d/Test-split_server.md).

Các revision/hash đầy đủ của file tải về nằm trong
`reports/artifact_download_manifest.json`; token không nằm trong report.

## Kiến trúc được chọn

| Thuộc tính | Quyết định và bằng chứng |
|---|---|
| CLIP | HF `openai/clip-vit-large-patch14`, revision `32bd64288804d66eefd0ccbe215aa642df71cc41`, 768 chiều |
| MLP | IM2TEXT 768→512→512→768, ReLU, dropout 0.1; keys/shapes strict-match v21 |
| Cross-attention | Hai CrossFormer độc lập, mỗi cái 3 lớp, 8 heads × 64; không thêm residual/LayerNorm |
| Token order | visual knowledge, text knowledge, mapped reference: theo official forward, không đổi theo thứ tự minh họa paper |
| Neighbor retrieval | Image→image và image→text Top-K độc lập, K=16; theo official code |
| Reference feature | L2-normalized: DB train features đã unit norm, dùng cấu hình này cho v21 |
| Composition | `a photo of * , {modification}`; thay marker bằng 3 tokens và dịch EOT thêm 2 |
| Fusion | Chuẩn hóa mỗi stream, alpha-weighted sum rồi L2-normalize |
| Matching | Exact cosine qua FAISS IndexFlatIP, knowledge và target gallery độc lập |

Các quy tắc normalization/prompt đã chạy được bằng checkpoint thật nhưng **chưa
được so output với notebook KEDs21_30ep.ipynb**, vì notebook không có trong các kho
artifact. Numerical parity của BKP đã so với ba class official. Chưa kết luận
toàn bộ inference parity với notebook hay OpenAI-native CLIP.

## Checkpoint và metric

BEST exports có ba module dict và meta branch/architecture; không tự chứa epoch.
LAST có `state_dict`, `optimizer`, `epoch`, `losses`, `phase`, `meta`.
Hai LAST có epoch=8, khớp báo cáo 9 completed epochs. Optimizer AdamW load được
và các momentum tensors đúng shape/hữu hạn. Không có scheduler/global-step/RNG.
Vì vậy không gọi lần tiếp tục từ LAST cũ là exact resume.

Theo remote `best_metrics.json`, BEST Phi-M là epoch 9, Phi-A epoch 7, được chọn
trên Fashion-IQ DEVELOPMENT subset. Không tái xuất các số này thành benchmark
đã chạy local. Source preflight tự xác định `exact_paper_reproduction=false`,
với các khác biệt HF CLIP, pseudo-triplets, training trên DB rows, scheduler
và self-exclusion. Tên mode giữ `author_style_approximation`.

## Sửa lỗi theo tài nguyên thật

1. FAISS monolithic build tăng capacity và gây `std::bad_alloc`. Giữ nguyên file
   partial 1.536 GB cho audit; dùng các shard IndexFlatIP 16.384 rows. Hợp nhất
   exact Top-K hai modalities độc lập; unit test so với brute-force cosine.
2. Windows PyTorch dùng Intel OpenMP còn FAISS wheel dùng LLVM OpenMP, gây
   `OMP Error #15` và abort khi search. FAISS chạy trong subprocess riêng trên
   Windows; Linux dùng native backend. Không dùng `KMP_DUPLICATE_LIB_OK`.
3. Một integration run bị native access violation khi safetensors đang load
   trong lúc downloader chạy lại trên cùng weights. Chưa chứng minh duy nhất
   nguyên nhân native crash; đã bỏ thao tác download trên file LFS có SHA đúng,
   chạy validation/download trước serving, và integration chạy lại thành công.
4. Gradio SDK coi key `path` trong model inspector là một file download. Đã bỏ
   filesystem path khỏi model-info public; vẫn giữ hashes/metadata trong report.

## Benchmark protocol

Fashion-IQ val có dress 2.017 queries/3.817 images, shirt 2.038/6.346,
toptee 1.961/5.373. Dùng caption2 `and` caption1 như official KEDs, giữ full
category gallery; macro average ba categories. CIRR val có 4.181 queries/2.297
images, test1 4.148/2.315. CIRR bỏ reference bằng ID, tính global R@1/5/10/50 và
subset R@1/2/3. Test1 chỉ xuất submission `recall`/`recall_subset`, không tính
metric khi không có labels. Các số lượng trên là audit annotations, không phải
kết quả model. Chưa có raw benchmark images local để chạy full evaluation.

