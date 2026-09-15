from fastapi import FastAPI

app = FastAPI(
    title="Mining Dispatch API",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Mining Dispatch API funcionando"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }