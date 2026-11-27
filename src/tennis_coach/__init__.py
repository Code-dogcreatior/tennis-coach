"""Tennis swing training assistant: video -> pose -> swing analysis -> reference matching."""

__version__ = "0.1.0"


def main() -> None:
    """Console entry point: start the web app (``uv run tennis-coach``)."""
    import uvicorn

    uvicorn.run("tennis_coach.api:app", host="127.0.0.1", port=8765, reload=False)
