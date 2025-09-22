from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="NeMo-PD Scoring API")

class Loan(BaseModel):
    # TODO: add origination fields (FICO, LTV, DTI, etc.)
    fico: int
    ltv: float
    dti: float

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/score")
def score(loan: Loan):
    # TODO: load model, preprocess, return pd and risk band
    return {"pd": 0.0123, "risk_band": "C", "explain": ["FICO", "LTV", "DTI"]}
