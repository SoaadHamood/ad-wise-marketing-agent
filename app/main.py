import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any, Dict

from app.ctr_analytics import run_smart_diagnosis
from app.ad_recommender import generate_strategy_and_ads

app = FastAPI(title="Ad-Wise Agency Intelligence")
os.makedirs("data", exist_ok=True)


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.post("/api/upload")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed.")
    with open(f"data/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "success"}


class ExecuteRequest(BaseModel):
    prompt: str


@app.post("/api/execute")
def execute(request: ExecuteRequest) -> Dict[str, Any]:
    steps = []

    # 1. Analyst Diagnosis
    analysis = run_smart_diagnosis(request.prompt)
    steps.append({"module": "Analyst", "response": analysis})

    # 2. Strategist Recommendations
    creative = generate_strategy_and_ads(request.prompt, analysis["report"])
    steps.append({"module": "Strategist", "response": creative})

    # 3. Merged Report with strict separator for the UI
    final_report = f"### 📊 DIAGNOSIS\n{analysis['report']}\n\n---\n\n### 💡 STRATEGY\n{creative['strategy_and_ads']}"

    return {
        "status": "ok",
        "response": final_report,
        "dashboard_data": analysis.get("kpis_used", {}),
        "steps": steps
    }