from __future__ import annotations

import io
from pathlib import Path

import gradio as gr
import httpx
from PIL import Image


def _check(response):
    if response.is_success:
        return
    try:
        detail = response.json().get("detail")
    except (TypeError, ValueError, AttributeError):
        detail = None
    message = {
        404: "Ảnh hoặc gallery không còn khả dụng. Hãy làm mới danh sách.",
        413: "Ảnh vượt giới hạn dung lượng hoặc kích thước.",
        422: "Ảnh hoặc mô tả không hợp lệ. Hãy kiểm tra định dạng và độ dài.",
        503: "Model hoặc gallery chưa sẵn sàng. Hãy thử lại hoặc kiểm tra /ready.",
    }.get(response.status_code, "Máy chủ chưa thể xử lý yêu cầu. Hãy thử lại.")
    if response.status_code == 422 and isinstance(detail, str):
        message += f" ({detail})"
    request_id = response.headers.get("x-request-id")
    raise gr.Error(f"{message} Mã yêu cầu: {request_id}" if request_id else message)


def _photo(response):
    _check(response)
    with Image.open(io.BytesIO(response.content)) as decoded:
        return decoded.convert("RGB")


def _gallery_choices(galleries):
    return [(f"{row['gallery_id']} · {row['count']:,} ảnh", row["gallery_id"]) for row in galleries]


def create_demo(api_url="http://127.0.0.1:8000", *, upload_mb=10, defaults=None, examples=(),
                experimental=False, initial_galleries=()):
    api_url = api_url.rstrip("/")
    defaults = defaults or {}
    examples_by_label = {row["label"]: row for row in examples}

    def get_galleries(client):
        response = client.get("/api/v1/gallery")
        _check(response)
        return response.json()

    def refresh(current=None):
        try:
            with httpx.Client(base_url=api_url, timeout=20, trust_env=False) as client:
                galleries = get_galleries(client)
                model = client.get("/api/v1/model/info")
                info = model.json() if model.is_success else {}
            ids = [row["gallery_id"] for row in galleries]
            choices = _gallery_choices(galleries)
            first = next((name for name in ids if name == "dress" or name.startswith("fashion-demo-dress-")),
                         ids[0] if ids else None)
            status = ("Sẵn sàng tìm kiếm." if ids else "Chưa có gallery. Hãy nạp ảnh và xây index trước.")
            if not info:
                status = "Model chưa sẵn sàng. Hãy kiểm tra `/ready`."
            profiles = info.get("available_rag_profiles", ["E0"]) if experimental else ["E0"]
            return (gr.update(choices=choices, value=current if current in ids else first), info,
                    gr.update(choices=profiles, value="E0"), status)
        except (gr.Error, httpx.HTTPError, ValueError):
            return (gr.skip(), gr.skip(), gr.skip(),
                    "**Chưa tải được gallery.** Bấm Làm mới gallery để thử lại.")

    def load_example(label):
        if label is None:
            return (gr.skip(),) * 7
        example = examples_by_label.get(label) if isinstance(label, str) else None
        try:
            if example is None:
                raise gr.Error("Ví dụ đã thay đổi. Hãy chọn lại từ menu hoặc tải lại trang.")
            gallery_id = example["gallery_id"]
            with httpx.Client(base_url=api_url, timeout=20, trust_env=False) as client:
                galleries = get_galleries(client)
                if gallery_id not in {row["gallery_id"] for row in galleries}:
                    raise gr.Error("Gallery của ví dụ này chưa có. Hãy thử ví dụ khác.")
                photo = _photo(client.get(f"/api/v1/gallery/{gallery_id}/images/{example['reference_id']}"))
        except (gr.Error, httpx.HTTPError, OSError, ValueError) as exc:
            message = exc.message if isinstance(exc, gr.Error) else "Hãy thử lại hoặc chọn ví dụ khác."
            gr.Warning(message)
            return gr.skip(), gr.skip(), gr.skip(), f"**Chưa tải được ví dụ.** {message}", gr.skip(), gr.skip(), gr.skip()
        return (photo, example["modification"], gr.update(choices=_gallery_choices(galleries), value=gallery_id),
                "Đã tải ví dụ. Bấm **Tìm ảnh** để xem kết quả.", [], [], None)

    def search_api(image, text, gallery_id, top_k, mode, alpha, knowledge_k, rag_profile="E0"):
        if not image or not (text or "").strip() or not gallery_id:
            raise gr.Error("Cần ảnh tham chiếu, mô tả thay đổi và gallery.")
        try:
            with httpx.Client(base_url=api_url, timeout=300, trust_env=False) as client:
                with open(image, "rb") as original:
                    response = client.post("/api/v1/search",
                        data={"modification": text.strip(), "gallery_id": gallery_id, "top_k": int(top_k),
                              "mode": mode, "alpha": float(alpha), "knowledge_k": int(knowledge_k),
                              "rag_profile": rag_profile},
                        files={"image": (Path(image).name, original)})
                _check(response)
                output = response.json()
                items = []
                for row in output["results"]:
                    photo = _photo(client.get(row["image_url"]))
                    label = "cosine" if row.get("score_type", "cosine").startswith("cosine") else "relevance"
                    items.append((photo, f"#{row['rank']} · {row['image_id']} · {label} {row['score']:.4f}"))
        except httpx.TimeoutException as exc:
            raise gr.Error("Tìm kiếm quá thời gian chờ. Hãy thử lại với Top-K nhỏ hơn.") from exc
        except FileNotFoundError as exc:
            raise gr.Error("Ảnh tham chiếu hết hiệu lực. Hãy tải lại ảnh hoặc chọn lại ví dụ.") from exc
        except (httpx.HTTPError, OSError, ValueError) as exc:
            raise gr.Error("Không thể đọc ảnh hoặc kết nối backend. Hãy thử lại.") from exc
        if items:
            status = (f"**{len(items)} / {int(top_k)} ảnh** · API **{output['latency_ms']:,.0f} ms** · "
                      f"`{gallery_id}` · `{mode} / {output['details']['rag_profile']}`. "
                      "Điểm dùng để xếp hạng, không phải xác suất đúng.")
        else:
            status = "Không còn ảnh để trả về sau khi loại ảnh tham chiếu. Hãy thử gallery có thêm ảnh."
        output["modification"] = text.strip()
        return items, output, status

    def run_search(image, text, gallery_id, top_k, mode, alpha, knowledge_k, rag_profile="E0"):
        try:
            return search_api(image, text, gallery_id, top_k, mode, alpha, knowledge_k, rag_profile)
        except gr.Error as exc:
            gr.Warning(exc.message)
            return [], None, "**Tìm kiếm chưa hoàn tất.** Kiểm tra ảnh, mô tả hoặc trạng thái backend và thử lại."

    def compare(image, text, gallery_id, top_k, alpha, knowledge_k, rag_profile="E0"):
        try:
            baseline, _, _ = search_api(image, text, gallery_id, top_k, "fusion", alpha, knowledge_k, "E0")
            composed, details, status = search_api(image, text, gallery_id, top_k, "keds", alpha, knowledge_k, rag_profile)
            return baseline, composed, details, status
        except gr.Error as exc:
            gr.Warning(exc.message)
            return [], [], None, "**So sánh chưa hoàn tất.** Kiểm tra đầu vào và thử lại."

    def clear():
        return [], [], None, "Đang tìm ảnh trong gallery…"

    def recover_search():
        return ([], [], None,
                "**Truy vấn chưa hoàn tất.** Nếu trang đã mở lâu, chọn lại ví dụ hoặc tải lại ảnh rồi bấm Tìm ảnh.")

    theme = gr.themes.Soft(primary_hue="teal", secondary_hue="blue")
    with gr.Blocks(title="KEDs · Tìm ảnh theo mô tả", theme=theme, delete_cache=(3600, 3600),
                   analytics_enabled=False) as demo:
        gr.Markdown("# KEDs · Tìm ảnh theo mô tả\n"
                    "Chọn ảnh và mô tả phần cần đổi. Kết quả là ảnh có sẵn trong gallery đã chọn.")
        with gr.Row(equal_height=False):
            with gr.Column(min_width=290):
                reference = gr.Image(type="filepath", image_mode=None, format="png", sources=["upload", "clipboard"],
                                     label="1. Ảnh tham chiếu", height=320)
                gr.Markdown(f"JPEG, PNG, WebP hoặc BMP · tối đa {upload_mb} MiB.")
            with gr.Column(min_width=290):
                example = gr.Dropdown(list(examples_by_label),
                                      label="Chọn ví dụ có ảnh thật", visible=bool(examples_by_label))
                modification = gr.Textbox(label="2. Mô tả thay đổi (tiếng Anh)", lines=3,
                                          placeholder="make it blue with longer sleeves")
                gallery = gr.Dropdown(choices=_gallery_choices(initial_galleries), label="3. Gallery đích")
                with gr.Row():
                    top_k = gr.Slider(1, 50, value=5, step=1, label="Số kết quả")
                    refresh_button = gr.Button("Làm mới gallery", size="sm")
                search = gr.Button("Tìm ảnh", variant="primary")
        status = gr.Markdown("Đang kết nối backend…")
        results = gr.Gallery(label="Kết quả theo thứ hạng", columns=[2, 3, 4], object_fit="contain", height=440,
                             format="png")
        with gr.Accordion("So sánh và thiết lập nghiên cứu", open=False):
            mode = gr.Dropdown(choices=[("KEDs BEST", "keds"), ("Chỉ ảnh", "image"), ("Chỉ văn bản", "text"),
                                        ("CLIP fusion", "fusion"), ("Phi-M", "phi_m"), ("Phi-A", "phi_a")],
                               value="keds", label="Phương pháp")
            rag_profile = gr.Dropdown(["E0"], value="E0", visible=experimental,
                                      label="Profile · E0 là baseline; các profile khác là thử nghiệm")
            alpha = gr.Slider(0, 1, value=defaults.get("alpha", 0.5), label="Alpha · trọng số Phi-M")
            knowledge_k = gr.Dropdown([4, 8, 16, 32], value=defaults.get("knowledge_k", 16), label="CC3M neighbors K")
            comparison = gr.Button("So sánh CLIP fusion và KEDs")
            baseline_results = gr.Gallery(label="CLIP fusion", columns=[2, 3, 4], object_fit="contain")
        with gr.Accordion("Chi tiết truy vấn và model", open=False):
            details = gr.JSON(label="Truy vấn")
            inspector = gr.JSON(label="Model")
        refresh_outputs = [gallery, inspector, rag_profile, status]
        clear_outputs = [baseline_results, results, details, status]
        refresh_button.click(refresh, [gallery], refresh_outputs, preprocess=False, queue=False)
        demo.load(refresh, [gallery], refresh_outputs, preprocess=False, queue=False, api_name=False)
        example.input(load_example, [example], [reference, modification, gallery, status, *clear_outputs[:3]],
                      preprocess=False, queue=False, trigger_mode="always_last", show_progress="minimal")
        search_event = search.click(clear, outputs=clear_outputs, queue=False, api_name=False).then(
            run_search, [reference, modification, gallery, top_k, mode, alpha, knowledge_k, rag_profile],
            [results, details, status], concurrency_limit=1, concurrency_id="inference", api_name="run_search")
        compare_event = comparison.click(clear, outputs=clear_outputs, queue=False, api_name=False).then(
            compare, [reference, modification, gallery, top_k, alpha, knowledge_k, rag_profile],
            clear_outputs, concurrency_limit=1, concurrency_id="inference", api_name="compare")
        for event in (search_event, compare_event):
            event.failure(recover_search, outputs=clear_outputs, queue=False, api_name=False)
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    create_demo(args.api_url).launch(server_name="127.0.0.1", server_port=args.port, max_file_size=10 * 1024**2)
