"""Run: PYTHONPATH=. uvicorn murphy.app.main:app --port 8765"""
from murphy.app.server import create_app

app = create_app()
