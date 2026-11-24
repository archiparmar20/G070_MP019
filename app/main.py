"""FastAPI service for MP019 EV charging energy prediction."""
from fastapi import FastAPI

app = FastAPI(title="MP019 EV charging energy")


@app.get("/health")
def health():
    """Liveness check: needs no model or data files."""
    return {"status": "ok"}
