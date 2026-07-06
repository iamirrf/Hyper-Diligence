from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root_serves_workbench() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "Hyper-Diligence" in response.text
    assert "/ui/app.js" in response.text


def test_ui_assets_are_served() -> None:
    css_response = client.get("/ui/styles.css")
    js_response = client.get("/ui/app.js")

    assert css_response.status_code == 200
    assert "text/css" in css_response.headers["content-type"]
    assert js_response.status_code == 200
    assert "javascript" in js_response.headers["content-type"]
