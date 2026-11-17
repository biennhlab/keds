# Kết quả xác minh Advanced RAG — 2026-11-17

Đây là báo cáo xác minh chức năng. **Chưa có kết luận tăng Recall/MRR Fashion-IQ.**

- Full suite: **35 passed, 0 failures, 0 errors, 0 skipped**; 69,4 giây trên Windows CPU.
- Artifact validation sau nâng cấp: **23 checks PASS**, gồm đủ 500.000 cặp, finite
  vectors/weights, strict checkpoint/optimizer, full FAISS ordering và SHA-256 artifacts.
- Unit-only: 32 passed, 3 artifact tests được skip theo cơ chế opt-in.
- CLIP ViT-L/14 frozen, CC3M 500K thật và hai BEST AuthorBKP v21 thật được sử dụng.
- E0 vẫn khớp image/text neighbor IDs, Top-3 IDs và cosine scores đã ghi trước nâng cấp.
- Kiểm thử variable-length knowledge cho batch trả [1,4,D] và [1,32,D], không padding.
- Đã xác minh frozen CLIP backward về BKP, optimizer/RNG continuation và các API lỗi.
- Knowledge reader read-only cho giá trị tensor bitwise-equal với torch-save fixture;
  reject các pickle global ngoài whitelist, không rewrite database.

Chạy thật 9 profiles × 2 lần truy vấn trên gallery 6 ảnh: E0, hybrid weighted,
hybrid RRF, knowledge MMR, adaptive K, adaptive fusion, gallery reranking,
combined experimental và exact cache. Không có label accuracy cho gallery này.

| Kiểm chứng thực tế | Quan sát |
|---|---|
| E0 | K=16, alpha=0.5, output regression giữ nguyên |
| Adaptive K | K=32 cho query demo |
| Adaptive fusion | Alpha ~0.25002 cho query demo |
| Combined | K=32, alpha ~0.54164, trả 3 ảnh thật |
| Exact cache | Lần đầu ~6,6 s; lặp lại ~65 ms, IDs/scores giữ nguyên |
| Live HTTP + Gradio | health/ready/docs/UI/openapi đều 200; combined search và CLIP fusion comparison nhận ảnh thật |

Những latency này có ít samples và khác thứ tự/OS page-cache; không phải bằng
chứng speedup tổng quát giữa các phương pháp. Raw profiles và bảng P50/P95 được
sinh tại `reports/rag_functional_profile.json`, `reports/rag_comparison.md`.
Live smoke ở `reports/rag_live_smoke.json`; full test XML ở
`reports/rag_verified_tests.xml`. Các file reports/progress đang bị ignore theo
cấu hình workspace hiện tại; không tự sửa quy tắc ignore của người dùng.

Evaluation code đã có full-gallery protocol, strict Candidate-1500 (6.016 queries,
1.500 unique IDs/query), deterministic development/nominal-holdout partition và
paired bootstrap 95% CI, stratified theo category. Partition seed 999 có 1.203
development / 4.813 nominal holdout query IDs. Đây là audit annotations, không phải
số query đã chạy evaluation model.

Chưa chạy Recall@1/5/10/50, MRR hay paired **accuracy** CI: thiếu full benchmark
images và Candidate-1500 manifest. Không có notebooks 17/19/21 hoặc original v21
development manifest, nên chưa đối chiếu E1–E5 và chưa chứng minh historical
holdout sạch. Default giữ E0, mọi kỹ thuật RAG mới là opt-in.

