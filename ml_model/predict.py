"""
predict.py – FastAPI inference server for STGCN traffic speed prediction.

Endpoints:
  GET  /health              → liveness check
  GET  /links               → list all link IDs
  POST /predict             → predict next time-step speed for all links
  POST /predict/{link_id}   → predict for a specific link

Usage:
  pip install fastapi uvicorn
  cd ml_model
  python predict.py          (runs on http://localhost:8000)

Frontend can call:
  POST http://localhost:8000/predict
  Body: { "speeds": [[spd_t-2, spd_t-1, spd_t], ...] }   (100 nodes × 3 timesteps)
"""

import os
import re
import csv
import sys
import json
import torch
import numpy as np

# 加入 data_collection 到 sys.path 以便引入 fetch_traffic 的功能
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
data_collection_path = os.path.join(parent_dir, "data_collection")
if data_collection_path not in sys.path:
    sys.path.append(data_collection_path)

from fetch_traffic import get_live_snapshot

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
# ── Load the saved model ──────────────────────────────────────────────────────
model_path  = os.path.join(current_dir, 'stgcn_model.pth')

# Import STGCN architecture (must be in same folder)
from stgcn import STGCN_Prototype

def load_model():
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Run train.py first to generate stgcn_model.pth"
        )
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    hp = checkpoint['hyperparams']
    model = STGCN_Prototype(
        num_nodes       = hp['num_nodes'],
        in_channels     = hp['in_channels'],
        hidden_channels = hp['hidden_channels'],
        out_channels    = hp['out_channels'],
        time_steps      = hp['time_steps'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint

try:
    model, checkpoint = load_model()
    hp       = checkpoint['hyperparams']
    norm     = checkpoint['normalization']
    link_ids = checkpoint['link_ids']
    adj      = checkpoint['adj']            # FloatTensor [N, N]
    MEAN     = norm['mean']
    STD      = norm['std']
    NUM_NODES    = hp['num_nodes']
    WINDOW_SIZE  = hp['window_size']
    MODEL_LOADED = True
    print(f"✓ Model loaded: {NUM_NODES} nodes, window={WINDOW_SIZE}")
except FileNotFoundError as e:
    MODEL_LOADED = False
    print(f"⚠ {e}")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="GeoAI Traffic Prediction API",
    description="STGCN-based traffic speed forecasting for Taipei",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """
    speeds: 2-D list of shape [num_nodes, window_size].
    Each row is one road segment's speed history (km/h) for the last W timesteps.
    If omitted, the server uses zeros (demo mode).
    """
    speeds: Optional[List[List[float]]] = None


class LinkPrediction(BaseModel):
    link_id         : str
    predicted_speed : float          # km/h
    congestion_level: int            # 1=free, 2=moderate, 3=congested


class PredictResponse(BaseModel):
    predictions     : List[LinkPrediction]
    window_size     : int
    model_version   : str = "stgcn-v1"


class MessageContext(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[MessageContext] = []


class ChatResponse(BaseModel):
    reply: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def speed_to_congestion(speed_kmh: float) -> int:
    if speed_kmh >= 40:
        return 1   # free flow
    elif speed_kmh >= 20:
        return 2   # moderate
    return 3        # congested


def load_live_speeds_matrix():
    """
    Load current speeds from taipei_live_traffic.csv and arrange them into a
    [NUM_NODES, WINDOW_SIZE] matrix in the order of the model's link_ids.
    The single snapshot is replicated across the time window. Missing links
    or -99 sentinels fall back to MEAN.
    Returns (speeds, matched_count).
    """
    live = {}
    if os.path.exists(LIVE_CSV_PATH):
        with open(LIVE_CSV_PATH, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    spd = float(row["TravelSpeed"])
                except (ValueError, KeyError):
                    continue
                if spd < 0:  # -99 = no data
                    continue
                live[row["LinkID"]] = spd

    speeds = np.full((NUM_NODES, WINDOW_SIZE), MEAN, dtype=np.float32)
    matched = 0
    for i, lid in enumerate(link_ids):
        if lid in live:
            speeds[i, :] = live[lid]
            matched += 1
    return speeds, matched


def run_inference(speeds_kmh: np.ndarray) -> np.ndarray:
    """
    speeds_kmh: [num_nodes, window_size] in km/h
    Returns: [num_nodes] predicted speeds in km/h
    """
    # Normalize
    x = (speeds_kmh - MEAN) / STD                         # [N, W]
    x = x[:, :, np.newaxis]                               # [N, W, 1]
    x_tensor = torch.FloatTensor(x)                       # [N, W, 1]
    x_tensor = x_tensor.permute(2, 0, 1).unsqueeze(0)     # [1, 1, N, W]
    
    with torch.no_grad():
        out = model(x_tensor, adj)                        # [1, N, 1]
    
    pred_norm = out.squeeze().numpy()                      # [N]
    pred_kmh  = pred_norm * STD + MEAN                    # denormalize
    pred_kmh  = np.clip(pred_kmh, 0, 120)                 # physical bounds
    return pred_kmh


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"      : "ok" if MODEL_LOADED else "model_not_loaded",
        "model_loaded": MODEL_LOADED,
        "num_nodes"   : NUM_NODES if MODEL_LOADED else 0,
    }


@app.get("/links")
def get_links():
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded")
    return {"link_ids": link_ids, "count": len(link_ids)}


LINESTRING_RE = re.compile(r"LINESTRING\s*\((.*)\)", re.IGNORECASE)
LIVE_CSV_PATH  = os.path.join(parent_dir, "data_collection", "taipei_live_traffic.csv")
LINKS_CSV_PATH = os.path.join(parent_dir, "data_collection", "taipei_traffic_links.csv")


def _parse_linestring(wkt: str):
    m = LINESTRING_RE.match(wkt.strip())
    if not m:
        return None
    coords = []
    for pt in m.group(1).split(","):
        parts = pt.strip().split()
        if len(parts) < 2:
            continue
        coords.append([float(parts[0]), float(parts[1])])
    return coords or None


def _load_link_geometry():
    geom = {}
    with open(LINKS_CSV_PATH, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            coords = _parse_linestring(row.get("Geometry", ""))
            if not coords:
                continue
            geom[row["LinkID"]] = {
                "road_name"   : row.get("RoadName", ""),
                "section_name": row.get("SectionName", ""),
                "direction"   : row.get("Direction", ""),
                "coords"      : coords,
            }
    return geom


@app.get("/live-traffic")
def live_traffic():
    """
    Return a GeoJSON FeatureCollection joining taipei_live_traffic.csv
    (LinkID, TravelSpeed, CongestionLevel, UpdateTime) with the link
    geometry in taipei_traffic_links.csv. Links with CongestionLevel=-99
    (no data) are excluded.
    """
    if not os.path.exists(LIVE_CSV_PATH) or not os.path.exists(LINKS_CSV_PATH):
        raise HTTPException(503, "Traffic CSV files not found")

    geom = _load_link_geometry()
    features = []
    latest_update = None

    with open(LIVE_CSV_PATH, "r", encoding="utf-8", newline="") as f:
      for row in csv.DictReader(f):
        link = geom.get(row["LinkID"])
        if not link:
            continue
        try:
            level = int(row["CongestionLevel"])
            speed = float(row["TravelSpeed"])
        except (ValueError, KeyError):
            continue
        if level == -99:
            continue
        update_time = row.get("UpdateTime", "")
        if update_time and (latest_update is None or update_time > latest_update):
            latest_update = update_time
        features.append({
            "type": "Feature",
            "properties": {
                "link_id"         : row["LinkID"],
                "road_name"       : link["road_name"],
                "section_name"    : link["section_name"],
                "direction"       : link["direction"],
                "speed"           : speed,
                "congestion_level": level,
                "update_time"     : update_time,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": link["coords"],
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "updated_at": latest_update,
    }


@app.post("/predict-geojson")
def predict_geojson():
    """
    Run STGCN inference seeded by the current taipei_live_traffic.csv snapshot
    and return predictions joined with the links' geometry as GeoJSON.
    """
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded. Run train.py first.")

    speeds, matched = load_live_speeds_matrix()
    pred_kmh = run_inference(speeds)

    geom = _load_link_geometry()
    features = []
    for i, lid in enumerate(link_ids):
        link = geom.get(lid)
        if not link:
            continue
        spd = float(pred_kmh[i])
        features.append({
            "type": "Feature",
            "properties": {
                "link_id"         : lid,
                "road_name"       : link["road_name"],
                "section_name"    : link["section_name"],
                "predicted_speed" : round(spd, 2),
                "congestion_level": speed_to_congestion(spd),
            },
            "geometry": {"type": "LineString", "coordinates": link["coords"]},
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "matched": matched,
        "num_nodes": NUM_NODES,
    }


@app.post("/predict", response_model=PredictResponse)
def predict_all(req: PredictRequest):
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded. Run train.py first.")
    
    if req.speeds is not None:
        speeds = np.array(req.speeds, dtype=np.float32)
        if speeds.shape != (NUM_NODES, WINDOW_SIZE):
            raise HTTPException(
                400,
                f"speeds must be shape [{NUM_NODES}, {WINDOW_SIZE}], "
                f"got {list(speeds.shape)}"
            )
    else:
        # Default: seed the model with the latest live CSV snapshot,
        # replicated across the time window. Missing links use MEAN.
        speeds, matched = load_live_speeds_matrix()
        print(f"[/predict] using live snapshot for {matched}/{NUM_NODES} nodes")

    pred_kmh = run_inference(speeds)

    predictions = [
        LinkPrediction(
            link_id          = link_ids[i],
            predicted_speed  = round(float(pred_kmh[i]), 2),
            congestion_level = speed_to_congestion(float(pred_kmh[i])),
        )
        for i in range(NUM_NODES)
    ]

    return PredictResponse(predictions=predictions, window_size=WINDOW_SIZE)


@app.post("/predict/{link_id}", response_model=LinkPrediction)
def predict_single(link_id: str, req: PredictRequest):
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded. Run train.py first.")
    
    if link_id not in link_ids:
        raise HTTPException(404, f"Link '{link_id}' not found")
    
    # Run full inference, return only this node
    if req.speeds is not None:
        speeds = np.array(req.speeds, dtype=np.float32)
        if speeds.shape != (NUM_NODES, WINDOW_SIZE):
            raise HTTPException(400, f"speeds must be [{NUM_NODES}, {WINDOW_SIZE}]")
    else:
        speeds = np.full((NUM_NODES, WINDOW_SIZE), MEAN, dtype=np.float32)
    
    pred_kmh = run_inference(speeds)
    idx      = link_ids.index(link_id)
    spd      = float(pred_kmh[idx])
    
    return LinkPrediction(
        link_id          = link_id,
        predicted_speed  = round(spd, 2),
        congestion_level = speed_to_congestion(spd),
    )


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not gemini_client:
        raise HTTPException(500, "GEMINI_API_KEY environment variable is not set. Cannot use Gemini API.")
    context = ""
    stgcn_info = ""
    if MODEL_LOADED:
        speeds = np.full((NUM_NODES, WINDOW_SIZE), MEAN, dtype=np.float32)
        pred_kmh = run_inference(speeds)
        avg_speed = float(np.mean(pred_kmh))
        free_links = sum(1 for s in pred_kmh if speed_to_congestion(float(s)) == 1)
        mod_links = sum(1 for s in pred_kmh if speed_to_congestion(float(s)) == 2)
        cong_links = sum(1 for s in pred_kmh if speed_to_congestion(float(s)) == 3)
        stgcn_info = (
            f"STGCN Model Predictions:\n"
            f"- Average speed: {avg_speed:.1f} km/h across {NUM_NODES} segments.\n"
            f"- Segment status: {free_links} free-flowing, {mod_links} moderate, {cong_links} congested.\n"
        )
    else:
        stgcn_info = "STGCN Model is currently offline.\n"

    # 取得 TDX 真實即時快照
    tdx_live_data = get_live_snapshot(top_n=10)
    tdx_info = ""
    if tdx_live_data:
        tdx_info = "Live TDX Traffic Snapshot (Top 10 Congested):\n"
        for i, item in enumerate(tdx_live_data, 1):
            road = item.get("RoadName", item.get("LinkID"))
            spd = item.get("TravelSpeed")
            tdx_info += f"{i}. {road}: {spd} km/h\n"
    else:
        tdx_info = "Live TDX Data is currently unavailable (using fallback logic or API failed).\n"

    # 組合 System Context
    context = (
        "You are a GeoAI Traffic Assistant helping users understand Taipei traffic conditions. "
        "You have access to both AI predictions (STGCN) and real-time TDX sensor data. "
        "Provide localized, helpful, and concise answers based on the following real-time context:\n\n"
        f"{stgcn_info}\n"
        f"{tdx_info}"
    )
    
    try:
        history_gemini = [
            types.Content(
                role="model" if msg.role == "assistant" else "user",
                parts=[types.Part.from_text(text=msg.content)],
            )
            for msg in req.history
        ]

        chat = gemini_client.chats.create(
            model="gemini-2.5-pro",
            config=types.GenerateContentConfig(system_instruction=context) if context else None,
            history=history_gemini,
        )
        response = chat.send_message(req.message)
        return ChatResponse(reply=response.text)
    except Exception as e:
        print(f"Chat Error: {e}")
        raise HTTPException(500, f"Error generating response: {str(e)}")


# ── Run standalone ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000, reload=False)
