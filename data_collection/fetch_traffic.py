"""
fetch_traffic.py – 台北市即時交通資料擷取腳本

功能：
  1. fetch_and_save_taipei_traffic()
     → 從 TDX API 拉取台北即時路況與路段幾何，存成 CSV
     → 若 API 失敗（無金鑰或網路異常），自動 fallback 產生 mock 資料

  2. get_live_snapshot(top_n=20)
     → 即時從 TDX 拉取最新路況快照，回傳 list[dict]
     → 供 /chat 端點注入 LLM context 使用

  3. generate_mock_data()
     → 產生模擬資料供離線開發測試

資料來源：
  TDX v2 Road Traffic Live API
  https://tdx.transportdata.tw/api-service/swagger
"""

import os
import time
import random
import pandas as pd
from tdx_client import TDXClient
from dotenv import load_dotenv

# 從 .env 檔讀取環境變數（若已存在環境變數，dotenv 不會覆蓋）
load_dotenv()


# ── 即時快照函式（供 /chat 端點使用）─────────────────────────────────────────

def get_live_snapshot(top_n: int = 20) -> list[dict]:
    """
    從 TDX 即時交通 API 拉取台北市最新路況快照。

    回傳 list[dict]，每筆包含：
      - LinkID:          路段編號
      - RoadName:        道路名稱（若 API 有提供）
      - TravelSpeed:     旅行速度 (km/h)
      - CongestionLevel: 壅塞等級 (1=暢通, 2=中等, 3=壅塞)
      - DataCollectTime: 資料收集時間

    若 API 失敗或無金鑰，回傳空 list（由呼叫端決定 fallback）。
    """
    try:
        client = TDXClient()
        # 拉取即時路況，限制筆數以控制 context 大小
        resp = client.get(
            "/v2/Road/Traffic/Live/City/Taipei",
            params={"$top": top_n, "$orderby": "TravelSpeed asc"},
        )
        raw = resp.get("LiveTraffics", []) if isinstance(resp, dict) else resp
        if not raw:
            return []

        snapshot = []
        for seg in raw:
            link_id = seg.get("SectionID", "N/A")
            snapshot.append({
                "LinkID":          link_id,
                "RoadName":        seg.get("SectionName", link_id),
                "TravelSpeed":     round(seg.get("TravelSpeed", 0), 1),
                "CongestionLevel": seg.get("CongestionLevel", seg.get("CongestionLevelID", 0)),
                "DataCollectTime": seg.get("DataCollectTime", "N/A"),
            })
        return snapshot

    except Exception as e:
        print(f"[get_live_snapshot] TDX API 查詢失敗: {e}")
        return []


# ── Mock 資料產生（離線開發用）────────────────────────────────────────────────

def generate_mock_data():
    """當 TDX API 不可用時，產生模擬的台北路況資料供開發使用。"""
    print("API Key not found or Unauthorized. Generating realistic mock data for Taipei...")
    
    # ── 產生模擬路段（Nodes）──────────────────────────────────────────────
    links = []
    for i in range(100):
        link_id = f"TPE-{1000+i}"
        length = random.uniform(50.0, 800.0)
        links.append({"LinkID": link_id, "Length": length})
    
    df_links = pd.DataFrame(links)
    links_output = os.path.join(os.path.dirname(__file__), "taipei_traffic_links.csv")
    df_links.to_csv(links_output, index=False)
    print(f"Saved {len(df_links)} mock link records to {links_output}")

    # ── 產生模擬車速資料（Features）───────────────────────────────────────
    records = []
    # 模擬 200 個時間步（每 5 分鐘一筆）
    import datetime
    base_time = datetime.datetime.strptime("2026-04-19T00:00:00", "%Y-%m-%dT%H:%M:%S")
    time_steps = [(base_time + datetime.timedelta(minutes=5*i)).strftime("%Y-%m-%dT%H:%M:%S") for i in range(200)]
    for t in time_steps:
        for link in links:
            speed = random.uniform(10.0, 60.0) # speed in km/h
            congestion = 1 if speed > 40 else (2 if speed > 20 else 3)
            records.append({
                "LinkID": link["LinkID"],
                "TravelSpeed": speed,
                "CongestionLevel": congestion,
                "UpdateTime": t
            })
            
    df_speed = pd.DataFrame(records)
    output_path = os.path.join(os.path.dirname(__file__), "taipei_live_traffic.csv")
    df_speed.to_csv(output_path, index=False)
    print(f"Saved {len(df_speed)} mock traffic speed records to {output_path}")

# ── 主要資料擷取函式 ─────────────────────────────────────────────────────────

def fetch_and_save_taipei_traffic():
    """
    從 TDX 拉取台北市即時路況與路段幾何，存成 CSV。

    產出檔案：
      - taipei_live_traffic.csv  → 即時車速 & 壅塞等級
      - taipei_traffic_links.csv → 路段基本資訊（LinkID, Length）

    若 API 拋出例外（無金鑰、超過配額等），自動 fallback 至 generate_mock_data()。
    """
    client = TDXClient()
    print("Fetching live traffic speed for Taipei...")
    
    try:
        # TDX: Road/Traffic/Live → LiveTraffics list (speed + congestion)
        live_resp = client.get("/v2/Road/Traffic/Live/City/Taipei", params={"$top": 2000})
        live_list = live_resp.get("LiveTraffics", []) if isinstance(live_resp, dict) else live_resp
        update_time = live_resp.get("UpdateTime") if isinstance(live_resp, dict) else None

        if not live_list:
            print("No live traffic data received.")
            return

        records = []
        for seg in live_list:
            records.append({
                "LinkID":          seg.get("SectionID"),
                "TravelSpeed":     seg.get("TravelSpeed", 0),
                "CongestionLevel": seg.get("CongestionLevel", seg.get("CongestionLevelID", 0)),
                "UpdateTime":      seg.get("DataCollectTime") or update_time,
            })

        df_speed = pd.DataFrame(records)
        output_path = os.path.join(os.path.dirname(__file__), "taipei_live_traffic.csv")
        df_speed.to_csv(output_path, index=False)
        print(f"Saved {len(df_speed)} live traffic records to {output_path}")

        # TDX: Road/Traffic/Section → static metadata (SectionLength, SectionName)
        # TDX: Road/Traffic/SectionShape → geometry per section
        print("Fetching section metadata & shapes for Taipei...")
        sec_resp   = client.get("/v2/Road/Traffic/Section/City/Taipei",      params={"$top": 2000})
        shape_resp = client.get("/v2/Road/Traffic/SectionShape/City/Taipei", params={"$top": 2000})

        sec_list   = sec_resp.get("Sections", [])     if isinstance(sec_resp, dict)   else sec_resp
        shape_list = shape_resp.get("SectionShapes", []) if isinstance(shape_resp, dict) else shape_resp

        # build lookup: SectionID → geometry WKT
        geo_map = {s["SectionID"]: s.get("Geometry", "") for s in shape_list if "SectionID" in s}

        links = []
        for item in sec_list:
            sid = item.get("SectionID")
            links.append({
                "LinkID":      sid,
                "SectionName": item.get("SectionName", ""),
                "RoadName":    item.get("RoadName", ""),
                "Direction":   item.get("RoadDirection", ""),
                "Length":      item.get("SectionLength"),
                "Geometry":    geo_map.get(sid, ""),
            })

        if links:
            df_links = pd.DataFrame(links)
            links_output = os.path.join(os.path.dirname(__file__), "taipei_traffic_links.csv")
            df_links.to_csv(links_output, index=False)
            print(f"Saved {len(df_links)} link records to {links_output}")
            
    except Exception as e:
        # API 失敗時自動產生 mock 資料，確保下游流程不中斷
        print(f"Error fetching data: {e}")
        generate_mock_data()

if __name__ == "__main__":
    fetch_and_save_taipei_traffic()
