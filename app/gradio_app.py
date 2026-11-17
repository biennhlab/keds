from __future__ import annotations

import io

import gradio as gr
import httpx
from PIL import Image


def create_demo(api_url="http://127.0.0.1:8000"):
    api_url = api_url.rstrip("/")

    def api_get(path):
        with httpx.Client(timeout=300, trust_env=False) as client:
            response = client.get(api_url + path)
            response.raise_for_status()
            return response.json()

    def refresh():
        try:
            galleries = api_get("/api/v1/gallery")
            ids = [x["gallery_id"] for x in galleries]
            info = api_get("/api/v1/model/info")
            profiles = info.get("available_rag_profiles", ["E0"])
            return (gr.update(choices=ids, value=ids[0] if ids else None), info,
                    gr.update(choices=profiles, value=info.get("rag", {}).get("profile", "E0")))
        except httpx.HTTPError as e:
            raise gr.Error("API chưa sẵn sàng. Kiểm tra /ready và artifact validation report.") from e

    def run_search(image, text, gallery_id, top_k, mode, alpha, knowledge_k, rag_profile="E0"):
        if image is None or not text.strip() or not gallery_id:
            raise gr.Error("Cần ảnh tham chiếu, modification text và gallery.")
        content = io.BytesIO()
        image.convert("RGB").save(content, format="PNG")
        with httpx.Client(timeout=300, trust_env=False) as client:
            response = client.post(api_url + "/api/v1/search",
                data={"modification": text, "gallery_id": gallery_id, "top_k": int(top_k),
                      "mode": mode, "alpha": float(alpha), "knowledge_k": int(knowledge_k),
                      "rag_profile": rag_profile},
                files={"image": ("reference.png", content.getvalue(), "image/png")})
            if response.is_error:
                raise gr.Error(f"Search failed ({response.status_code}): {response.json().get('detail')}")
            output = response.json()
            items = []
            for row in output["results"]:
                photo = client.get(api_url + row["image_url"])
                photo.raise_for_status()
                with Image.open(io.BytesIO(photo.content)) as decoded:
                    label = "cosine" if row.get("score_type", "cosine") == "cosine" else "relevance"
                    items.append((decoded.convert("RGB"),
                                  f"#{row['rank']} · {row['image_id']} · {label} {row['score']:.4f}"))
        return items, output

    def compare(image, text, gallery_id, top_k, alpha, knowledge_k, rag_profile="E0"):
        baseline, _ = run_search(image, text, gallery_id, top_k, "fusion", alpha, knowledge_k, "E0")
        composed, details = run_search(image, text, gallery_id, top_k, "keds", alpha, knowledge_k, rag_profile)
        return baseline, composed, details

    with gr.Blocks(title="KEDs Composed Image Retrieval", delete_cache=(3600, 3600),
                   analytics_enabled=False) as demo:
        gr.Markdown("# KEDs · Tìm ảnh bằng ảnh + mô tả thay đổi\n"
                    "Kết quả là ảnh có sẵn trong gallery. CC3M cung cấp knowledge cho model.")
        with gr.Row():
            reference = gr.Image(type="pil", label="Ảnh tham chiếu")
            with gr.Column():
                modification = gr.Textbox(label="Modification text (English)", lines=3,
                    placeholder="with two dogs instead of one")
                gallery = gr.Dropdown(choices=[], label="Target gallery")
                refresh_button = gr.Button("Refresh galleries / model")
                top_k = gr.Slider(1, 50, value=10, step=1, label="Top-K results")
                mode = gr.Dropdown(choices=["keds", "phi_m", "phi_a", "image", "text", "fusion"],
                                   value="keds", label="Model / baseline")
                with gr.Accordion("Research settings", open=False):
                    rag_profile = gr.Dropdown(["E0"], value="E0", label="RAG profile · experimental, chưa xác minh Recall")
                    alpha = gr.Slider(0, 1, value=0.5, label="Alpha · Phi-M weight")
                    knowledge_k = gr.Dropdown([4, 8, 16, 32], value=16, label="CC3M neighbors K")
                search = gr.Button("Search", variant="primary")
                comparison = gr.Button("Compare CLIP fusion vs KEDs")
        results = gr.Gallery(label="Ranked results", columns=3, object_fit="contain")
        baseline_results = gr.Gallery(label="CLIP fusion baseline (comparison)", columns=3)
        with gr.Accordion("Query details / knowledge neighbors", open=False):
            details = gr.JSON()
        with gr.Accordion("Model inspector", open=False):
            inspector = gr.JSON()
        refresh_button.click(refresh, outputs=[gallery, inspector, rag_profile])
        demo.load(refresh, outputs=[gallery, inspector, rag_profile])
        search.click(run_search, [reference, modification, gallery, top_k, mode, alpha, knowledge_k, rag_profile],
                     [results, details], concurrency_limit=1)
        comparison.click(compare, [reference, modification, gallery, top_k, alpha, knowledge_k, rag_profile],
                         [baseline_results, results, details], concurrency_limit=1)
    return demo


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--api-url", default="http://127.0.0.1:8000")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()
    create_demo(args.api_url).launch(server_name="127.0.0.1", server_port=args.port)

