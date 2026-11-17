import importlib.util
from pathlib import Path


def test_gradio_interface_builds_without_loading_a_second_model():
    path = Path("app/gradio_app.py")
    spec = importlib.util.spec_from_file_location("keds_gradio", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    demo = module.create_demo("http://127.0.0.1:8000")
    config = demo.get_config_file()
    assert sum(c["type"] == "gallery" for c in config["components"]) == 2
    assert any(c["type"] == "image" for c in config["components"])
    demo.close()

