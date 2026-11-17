# Advanced Multimodal RAG cho KEDs 500K

Các module dùng **cùng frozen CLIP ViT-L/14 và Phi-M/Phi-A AuthorBKP v21**.
Không thay CrossFormer, số pseudo-tokens, checkpoint hoặc knowledge DB. E0 giữ
fixed K=16, alpha=0.5 và independent image→image/image→caption cosine retrieval.
Mọi kỹ thuật mới mặc định tắt, chưa được chọn làm mặc định bằng kết quả demo.

## Audit và baseline

Code gốc của lần nâng cấp là commit `24ce17f` có end-to-end engine/API/Gradio.
`plan.md` đã được giữ nguyên; yêu cầu Advanced RAG bổ sung thứ tự P1–P5.
Không tìm thấy notebooks 17/19/21 trong repository, thư mục dự án lân cận hoặc
danh sách file hai kho HF đã cung cấp. Vì vậy chưa gán công thức/nhãn E1–E5 của
Notebook 17 cho các heuristic mới. E4/E5 ở đây chưa được tuyên bố tái lập.

`tests/fixtures/e0_regression.json` chứa kết quả thật đã quan sát ở CLI trước
nâng cấp. Report cũ không còn trong workspace sau lần runtime/workspace reset;
oracle này được khôi phục từ tool output đã ghi, có provenance rõ ràng. Nó chỉ
được đọc trong kiểm thử, không nằm trên đường prediction của ứng dụng.
Kiểm tra E0 đối chiếu IDs, scores, determinism và frozen CLIP backward tới BKP.

## P1 · Query-aware hybrid knowledge retrieval

`rag/query_aware.py` nhận reference image feature và modification text feature;
không có target/ground-truth trong interface. Các truy vấn FAISS exact gồm:

1. Image → image bank; image → caption bank.
2. Modification text → image bank; modification text → caption bank.
3. Hợp candidate pool riêng cho hai modality, loại duplicate row IDs.
4. Tính bốn cross-modal cosine channels cho từng cặp image-caption ứng viên.
5. Min-max hoặc sigmoid(z-score) normalization; weighted score fusion, RRF hoặc
   kết hợp hai cách. RRF mặc định dùng constant 60, route weights 0.7/0.3.

Ví dụ cấu hình trọng số bốn channels theo thứ tự image→image, image→caption,
text→image, text→caption:

```yaml
rag:
  query_aware:
    enabled: true
    candidate_pool: 64
    method: hybrid       # weighted | rrf | hybrid
    normalization: minmax
    image_weights: [0.55, 0.15, 0.20, 0.10]
    text_weights: [0.15, 0.55, 0.10, 0.20]
    route_weights: [0.7, 0.3]
    rrf_constant: 60
    rrf_weight: 0.5
```

Các trọng số trên là **research defaults chưa tuning**, không phải lựa chọn đã
chứng minh tốt hơn E0 trên Fashion-IQ.

## P2 · Knowledge reranking và MMR

`knowledge_reranking.py` kết hợp initial relevance, image-caption consistency,
modification alignment và visual-reference preservation. Mỗi channel được
normalize trước fusion. MMR tùy chọn giảm redundancy theo image/text embeddings
tương ứng; `diversity_lambda=1` trả relevance ranking. Trả K rows đúng modality
cho BKP, không dùng text decoder, LLM hoặc ground truth.

```yaml
rag:
  knowledge_reranking:
    enabled: true
    normalization: minmax
    weights: {initial: 0.5, consistency: 0.15, modification: 0.25, visual: 0.1}
    mmr: true
    diversity_lambda: 0.8
```

## P3 · Adaptive K và alpha

`adaptive.py` tính confidence từ normalized softmax entropy và score gap.
Thresholds mặc định 0.75/0.5/0.25 chọn K trong 4/8/16/32. Query rõ nét dùng ít
neighbors hơn; phân phối mơ hồ dùng nhiều hơn. Batch CLIP được encode cùng nhau;
sample có K khác nhau được đưa riêng qua BKP, không thêm padding không có mask.

Adaptive alpha đo confidence của mỗi composed stream trên cùng knowledge context,
kết hợp prior alpha bằng logistic, giới hạn mặc định [0.1,0.9]. Đây là heuristic
chưa calibration; không thay đổi bất kỳ tham số đã train nào. Details trả
`knowledge_k_used`, `alpha_used`, confidence/entropy/gap để audit từng truy vấn.

## P4 · Gallery reranking

`gallery_reranking.py` chỉ đổi thứ tự trong Top-N (mặc định 100), kết hợp cosine
composed-query, visual consistency và modification-text alignment; MMR là tùy
chọn. Full evaluation vẫn giữ mọi image trong gallery: rerank head, nối nguyên
tail để tính MRR đầy đủ. Candidate-1500 restrict pool **trước** ranking.

API trả `score_type` và `cosine_score` để phân biệt relevance score với cosine.
MMR có thể ưu tiên diversity thay cho relevance thuần, nên score không được hiểu
là xác suất đúng. Không có target ID trong thuật toán rerank.

## P5 · Cache, batching và memory

`cache.py` có LRU/TTL và byte budget, chỉ giữ CPU tensors đã detach. Cache gồm
CLIP image/text features, knowledge neighbors và composed embeddings/context.
Key/namespace chứa CLIP revision/preprocessing, checkpoint hashes, DB/index
fingerprint, prompt/normalization, mode, alpha, K, toàn bộ RAG config và image/
text content hashes. Gallery cache kiểm tra `CURRENT` generation trước khi reuse.
Không cache final results qua một gallery generation khác.

Exact cache có thể giữ nguyên E0; semantic approximate reuse là profile riêng,
mặc định tắt, threshold 0.9995 và chỉ lookup cùng namespace. Không tuyên bố
semantic cache không ảnh hưởng Recall. `encode_queries()` batch CLIP và giữ
variable-length knowledge đúng cách; online dùng `torch.inference_mode()`.

`knowledge/storage.py` đọc trực tiếp contiguous tensor storage trong torch ZIP
bằng NumPy **read-only mmap**. File `.pt` và values không thay đổi. Điều này
tránh commit reservation khoảng 3 GB của Windows torch copy-on-write mapping.
Reader dùng whitelist storage types/rebuild functions nội bộ, không thực thi
global tùy ý của pickle. Tests kiểm tra bitwise equality và reject unknown globals.

## Chạy profile và API/UI

```powershell
.venv/Scripts/python.exe scripts/search.py --image data/demo/dog.jpg --text "with two dogs instead of one" --gallery artifacts/gallery/demo --rag-profile hybrid_combined --top-k 3
.venv/Scripts/python.exe scripts/profile_rag.py --repeats 2
.venv/Scripts/python.exe scripts/render_rag_report.py --input reports/rag_functional_profile.json
.venv/Scripts/python.exe scripts/run_demo.py
```

`configs/rag_profiles.yaml` định nghĩa E0, weighted/RRF/hybrid, rerank/MMR,
adaptive K/fusion, gallery rerank, exact/semantic cache và combined experimental.
REST nhận field `rag_profile`; Gradio có selector research profile. Có thể dùng
`--config configs/rag_experimental.yaml` để opt-in combined cho một process riêng.
Không đổi cấu hình E0 mặc định chỉ vì smoke run thành công.

## Full Gallery / Candidate-1500 / bootstrap

```powershell
# Chỉ partition IDs, không dùng target để chọn split.
.venv/Scripts/python.exe scripts/partition_queries.py --data-root references/fashion-iq
.venv/Scripts/python.exe scripts/evaluate.py --dataset fashioniq --data-root data/fashioniq --gallery-root artifacts/gallery/fashioniq --partitions configs/query_partitions.json --phase development --rag-profiles E0,hybrid_weighted,hybrid_rrf,knowledge_mmr,adaptive_k,adaptive_fusion,gallery_rerank,combined_experimental --output reports/rag_development.json
.venv/Scripts/python.exe scripts/render_rag_report.py --input reports/rag_development.json
# Sau khi khóa hyperparameters trên development, chạy riêng E0 và variant trên holdout.
.venv/Scripts/python.exe scripts/evaluate.py --dataset fashioniq --data-root data/fashioniq --gallery-root artifacts/gallery/fashioniq --partitions configs/query_partitions.json --phase holdout --rag-profile E0 --output reports/e0_holdout.json
.venv/Scripts/python.exe scripts/compare_rag_reports.py --baseline reports/e0_holdout.json --variant reports/variant_holdout.json
```

Giữ Fashion-IQ full category gallery, caption2 + `and` + caption1, reference
policy và preprocessing như baseline. Macro average theo category. Report có
Recall@1/5/10/50, MRR, per-query ranks, P50/P95, peak VRAM (null trên CPU).
Paired bootstrap resample cùng query IDs, stratified theo categories để giữ
macro averaging, seed=999, 2.000 repetitions; CI chứa zero là chưa kết luận.

Candidate-1500 dùng `--protocol candidate1500 --candidates frozen.jsonl`.
JSONL có một row/query, fields `query_id`, `candidate_ids`; validator bắt buộc
đúng 6.016 queries và 1.500 unique gallery IDs/query. Không tự tạo/repair pool
bằng target label; thiếu target thì dừng protocol với lỗi. Candidate file được
đọc theo byte offsets, không giữ 9 triệu IDs trong RAM. Không so trực tiếp
Candidate-1500 với Full Gallery paper.

Partition mới chỉ là **nominal holdout**: chưa có original v21 dev manifest để
chứng minh không overlap với checkpoint selection trước đây. Report giữ flag
historical audit và không tự coi là confirmatory evidence.

## Trạng thái kết quả

`reports/rag_functional_profile.json` và bảng `reports/rag_comparison.md` là
functional/latency smoke trên sáu ảnh thật, **không có accuracy labels**. Sample
latency ít và phụ thuộc thứ tự/OS page cache; không suy ra speedup giữa kỹ thuật.
Full Fashion-IQ images, Candidate-1500 pools và notebooks 17/19/21 chưa có;
chưa chạy Recall/MRR/accuracy bootstrap và chưa có kỹ thuật mới được promote.

