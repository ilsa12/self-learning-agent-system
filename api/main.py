from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os

# Allow importing from core/ folder
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from core.agent import run_agent

app = FastAPI(title="Self-Learning Agent API")

class TaskRequest(BaseModel):
    task: str

@app.post("/run-agent")
def run_agent_endpoint(request: TaskRequest):
    result = run_agent(request.task)
    return {"task": request.task, "output": result}

@app.get("/")
def health_check():
    return {"status": "Agent API is running"}