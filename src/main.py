from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
import sys
import threading
import pandas as pd
from typing import Dict, Any, List
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.generator import NetworkGenerator, save_network_to_json, load_network_from_json
from src.utils.trainer import train_agent
from src.evaluation.benchmark import run_benchmarks

app = FastAPI(
    title="AIR-NDF API",
    description="API for the AI-Reinforced Adaptive Network Discovery Framework",
    version="1.0.0"
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory training status
training_state = {
    "is_running": False,
    "agent_type": None,
    "network_scale": None,
    "current_episode": 0,
    "total_episodes": 0,
    "last_reward": 0.0,
    "last_coverage": 0.0,
    "last_loss": 0.0,
    "rewards": [],
    "coverages": [],
    "losses": [],
    "status_message": "Idle"
}

def training_progress_callback(episode, total_episodes, reward, coverage, loss):
    training_state["current_episode"] = episode
    training_state["total_episodes"] = total_episodes
    training_state["last_reward"] = reward
    training_state["last_coverage"] = coverage
    training_state["last_loss"] = loss
    training_state["rewards"].append(reward)
    training_state["coverages"].append(coverage)
    training_state["losses"].append(loss)
    
    if episode % 10 == 0 or episode == total_episodes:
        training_state["status_message"] = f"Training episode {episode}/{total_episodes} - Cov: {coverage*100:.1f}% - Loss: {loss:.4f}"

def run_training_async(agent_type: str, network_scale: str, episodes: int):
    global training_state
    try:
        training_state["is_running"] = True
        training_state["agent_type"] = agent_type
        training_state["network_scale"] = network_scale
        training_state["current_episode"] = 0
        training_state["total_episodes"] = episodes
        training_state["rewards"] = []
        training_state["coverages"] = []
        training_state["losses"] = []
        training_state["status_message"] = "Initializing environment..."
        
        train_agent(
            agent_type=agent_type,
            network_scale=network_scale,
            episodes=episodes,
            progress_callback=training_progress_callback
        )
        training_state["status_message"] = "Training Completed Successfully!"
    except Exception as e:
        training_state["status_message"] = f"Error during training: {str(e)}"
    finally:
        training_state["is_running"] = False

# Endpoints

@app.get("/")
def read_root():
    """Serves the main dashboard application HTML."""
    html_path = "src/static/index.html"
    if os.path.exists(html_path):
        return FileResponse(html_path)
    raise HTTPException(status_code=404, detail="Dashboard UI index.html not found.")

@app.get("/generate-network")
def get_generated_network(scale: str = "medium"):
    """Generates (or loads if pre-existing) a network topology and returns node/edge data."""
    try:
        filepath = f"src/datasets/network_{scale}.json"
        if os.path.exists(filepath):
            G = load_network_from_json(filepath)
        else:
            gen = NetworkGenerator(seed=42)
            G = gen.generate(scale)
            save_network_to_json(G, filepath)
            
        nodes = []
        for n, attr in G.nodes(data=True):
            node_data = attr.copy()
            node_data["label"] = f"{attr['host_type']} ({attr['ip']})"
            nodes.append(node_data)
            
        edges = [{"from": u, "to": v} for u, v in G.edges()]
        
        return JSONResponse(content={"nodes": nodes, "edges": edges})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
def trigger_training(agent: str, scale: str = "small", episodes: int = 150, background_tasks: BackgroundTasks = None):
    """Triggers agent training in a background process."""
    if training_state["is_running"]:
        return {"status": "error", "message": f"An agent ({training_state['agent_type']}) is already training."}
        
    background_tasks.add_task(run_training_async, agent.upper(), scale, episodes)
    return {"status": "success", "message": f"Started training {agent.upper()} in background."}

@app.get("/train-status")
def get_training_status():
    """Retrieves current in-progress training stats."""
    return JSONResponse(content=training_state)

@app.post("/evaluate")
def trigger_evaluation(scale: str = "small", episodes: int = 20):
    """Triggers baseline and RL agent benchmarking on the specified network."""
    try:
        # Load or generate
        filepath = f"src/datasets/network_{scale}.json"
        if not os.path.exists(filepath):
            gen = NetworkGenerator(seed=42)
            G = gen.generate(scale)
            save_network_to_json(G, filepath)
            
        df = run_benchmarks(network_scale=scale, num_episodes=episodes)
        return JSONResponse(content=df.to_dict(orient="records"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results")
def get_available_results():
    """Lists available charts, tables, and reports in the results directory."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        return {"images": [], "csvs": []}
        
    files = os.listdir(results_dir)
    images = [f for f in files if f.endswith(".png")]
    csvs = [f for f in files if f.endswith(".csv")]
    
    return {"images": images, "csvs": csvs}

@app.get("/results/image/{image_name}")
def get_result_image(image_name: str):
    """Serves a generated plot file."""
    image_path = os.path.join("results", image_name)
    if os.path.exists(image_path):
        return FileResponse(image_path)
    # Also check results/logs directory for training curve plots
    log_image_path = os.path.join("results/logs", image_name)
    if os.path.exists(log_image_path):
        return FileResponse(log_image_path)
    raise HTTPException(status_code=404, detail="Result plot not found.")

@app.get("/metrics")
def get_metrics_table(scale: str = "medium"):
    """Reads and returns the benchmarking CSV results table as JSON."""
    csv_path = f"results/benchmark_{scale}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return JSONResponse(content=df.to_dict(orient="records"))
    else:
        # Return fallback mock empty list
        return JSONResponse(content=[])

# Mount static files directory (if stylesheet or extra js files are used)
os.makedirs("src/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="src/static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
