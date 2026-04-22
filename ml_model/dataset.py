import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    """
    Loads Taipei traffic data (real TDX snapshots) and builds a geographically-correct
    adjacency matrix from WKT LINESTRING endpoints in taipei_traffic_links.csv.

    When only one real timestamp is available (fresh fetch), a 200-step AR(1) time
    series is synthesised so STGCN has enough windows to train.  Once you have
    collected multiple snapshots via repeated fetch_traffic.py runs the synthetic
    path is skipped automatically.
    """

    def __init__(self, data_path, window_size=6, predict_size=1,
                 synthetic_steps=200, seed=42):
        self.data_path       = data_path
        self.window_size     = window_size
        self.predict_size    = predict_size
        self.synthetic_steps = synthetic_steps
        self.seed            = seed

        self.features, self.adj, self.link_ids, self.mean, self.std = self._load_data()
        self.x, self.y = self._generate_samples()

    # ── Geometry helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _wkt_endpoints(wkt: str):
        """Return (start_xy, end_xy) float tuples from a WKT LINESTRING, else (None, None)."""
        coords = re.findall(r'([\d.]+)\s+([\d.]+)', str(wkt))
        if len(coords) < 2:
            return None, None
        return (float(coords[0][0]), float(coords[0][1])), \
               (float(coords[-1][0]), float(coords[-1][1]))

    def _build_adjacency(self, df_links: pd.DataFrame, num_nodes: int) -> np.ndarray:
        """
        Connect road sections whose endpoints are within COORD_THRESH degrees (~55 m
        near Taipei lat 25°).  All four endpoint pair combinations are checked so
        head-to-tail, tail-to-head, and merging/diverging roads are all captured.
        Falls back to chain edges when geometry is missing.
        """
        COORD_THRESH = 5e-4

        # float64 preserves ~10 decimal digits for coordinates like 121.57226
        starts = np.full((num_nodes, 2), np.nan, dtype=np.float64)
        ends   = np.full((num_nodes, 2), np.nan, dtype=np.float64)
        for i, row in enumerate(df_links.itertuples(index=False)):
            s, e = self._wkt_endpoints(getattr(row, 'Geometry', ''))
            if s: starts[i] = s
            if e: ends[i]   = e

        adj      = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        has_geom = not np.all(np.isnan(starts))

        if has_geom:
            for pts_i in (starts, ends):
                for pts_j in (starts, ends):
                    vi    = ~np.isnan(pts_i[:, 0])
                    vj    = ~np.isnan(pts_j[:, 0])
                    diff  = pts_i[:, None, :] - pts_j[None, :, :]          # (N, N, 2)
                    close = (np.sqrt((diff ** 2).sum(-1)) < COORD_THRESH) \
                            & vi[:, None] & vj[None, :]
                    adj  += close.astype(np.float32)
            adj = (adj > 0).astype(np.float32)
        else:
            for i in range(num_nodes - 1):
                adj[i, i + 1] = adj[i + 1, i] = 1.0

        np.fill_diagonal(adj, 1.0)

        degree     = adj.sum(axis=1)
        d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        D          = np.diag(d_inv_sqrt)
        return (D @ adj @ D).astype(np.float32)

    # ── Temporal synthesis ────────────────────────────────────────────────────

    def _synthesize_temporal(self, snapshot: np.ndarray) -> np.ndarray:
        """
        Simulate a plausible multi-step time series via stationary AR(1) noise
        (ρ=0.85, σ=5 km/h) centred on the real snapshot.
        Returns array of shape (num_nodes, synthetic_steps).
        """
        valid    = snapshot >= 0
        mean_spd = float(snapshot[valid].mean()) if valid.any() else 40.0
        base     = np.where(valid, snapshot, mean_spd).astype(np.float32)

        rho, sigma = 0.85, 5.0
        N, T       = len(base), self.synthetic_steps
        white_std  = sigma * np.sqrt(1 - rho ** 2)

        np.random.seed(self.seed)
        noise = np.zeros((N, T), dtype=np.float32)
        for t in range(1, T):
            noise[:, t] = rho * noise[:, t - 1] + \
                          np.random.randn(N).astype(np.float32) * white_std

        return np.clip(base[:, None] + noise, 0.0, 120.0)   # (N, T)

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self):
        df_links   = pd.read_csv(os.path.join(self.data_path, 'taipei_traffic_links.csv'))
        df_traffic = pd.read_csv(os.path.join(self.data_path, 'taipei_live_traffic.csv'))

        link_ids  = df_links['LinkID'].tolist()
        num_nodes = len(link_ids)

        # Efficient pivot: rows=time, cols=LinkID, values=TravelSpeed
        pivot = (
            df_traffic
            .pivot_table(index='UpdateTime', columns='LinkID',
                         values='TravelSpeed', aggfunc='first')
            .reindex(columns=link_ids, fill_value=-99.0)
            .fillna(-99.0)
        )
        feature_mat = pivot.values.T.astype(np.float32)   # (num_nodes, num_times)
        num_times   = feature_mat.shape[1]

        # Single-snapshot fallback: synthesise temporal dimension
        if num_times <= 1:
            snapshot    = feature_mat[:, 0] if num_times == 1 else np.full(num_nodes, 40.0)
            feature_mat = self._synthesize_temporal(snapshot)
            print(f"[dataset] 1 筆真實時間步 → 合成 {self.synthetic_steps} 步 AR(1) 時序")
        else:
            print(f"[dataset] 載入 {num_times} 筆真實時間步")

        # Impute remaining -99 (offline sensors) with each node's valid mean
        for i in range(num_nodes):
            row  = feature_mat[i]
            good = row >= 0
            feature_mat[i, ~good] = float(row[good].mean()) if good.any() else 40.0

        # Z-score normalise
        mean = float(feature_mat.mean())
        std  = float(feature_mat.std()) if feature_mat.std() > 0 else 1.0
        feature_mat = (feature_mat - mean) / std

        features = torch.FloatTensor(feature_mat[:, :, None])   # (N, T, 1)
        adj      = self._build_adjacency(df_links, num_nodes)

        return features, torch.FloatTensor(adj), link_ids, mean, std

    # ── Window generation ─────────────────────────────────────────────────────

    def _generate_samples(self):
        _, num_times, _ = self.features.shape
        x_list, y_list  = [], []

        for i in range(num_times - self.window_size - self.predict_size + 1):
            x_list.append(self.features[:, i : i + self.window_size, :])
            y_list.append(self.features[:,
                          i + self.window_size :
                          i + self.window_size + self.predict_size, 0])

        return torch.stack(x_list), torch.stack(y_list)

    # ── PyTorch Dataset interface ──────────────────────────────────────────────

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # [in_channels, num_nodes, window_size]  for STGCN Conv2d
        return self.x[idx].permute(2, 0, 1), self.y[idx]


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(current_dir, '..', 'data_collection')

    ds = TrafficDataset(data_dir)
    print(f"Nodes         : {ds.features.shape[0]}")
    print(f"Time steps    : {ds.features.shape[1]}")
    print(f"Samples       : {len(ds)}")
    print(f"Adj matrix    : {ds.adj.shape}  (non-zero: {(ds.adj > 0).sum().item()})")
    print(f"Speed mean    : {ds.mean:.2f} km/h,  std: {ds.std:.2f}")
    x, y = ds[0]
    print(f"Sample X      : {x.shape}  (channels, nodes, time)")
    print(f"Sample Y      : {y.shape}  (nodes, predict_size)")
