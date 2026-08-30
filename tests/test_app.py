from pathlib import Path

from streamlit.testing.v1 import AppTest

from toxiscan_core.classifier import load_model_bundle

APP_PATH = Path(__file__).parents[1] / "toxiscan.py"


def test_initial_web_page_mode_renders_without_loading_model():
    load_model_bundle.cache_clear()

    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)

    assert not app.exception
    assert app.title[0].value == "🛡️ ToxiScan"
    assert app.radio[0].value == "Web page"
    assert app.button[0].label == "Scan page"
    assert load_model_bundle.cache_info().currsize == 0


def test_document_mode_uses_generic_user_facing_description():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=20)

    app.radio[0].set_value("Document").run(timeout=20)

    assert not app.exception
    assert app.file_uploader[0].label == "Upload a document"
    visible_copy = " ".join(element.value for element in app.markdown)
    assert "wide range of documents" in visible_copy
    assert ".txt" not in visible_copy
    assert ".pdf" not in visible_copy
    assert ".docx" not in visible_copy
