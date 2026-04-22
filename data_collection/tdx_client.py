import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

_TOKEN_URL = "https://tdx.transportdata.tw/auth/realms/TDXConnect/protocol/openid-connect/token"
_BASE_URL  = "https://tdx.transportdata.tw/api/basic"


class TDXClient:
    def __init__(self):
        self._client_id     = os.getenv("TDX_CLIENT_ID")
        self._client_secret = os.getenv("TDX_CLIENT_SECRET")
        self._token         = None
        self._token_expiry  = 0

    def _refresh_token(self):
        res = requests.post(
            _TOKEN_URL,
            headers={"content-type": "application/x-www-form-urlencoded"},
            data={
                "grant_type":    "client_credentials",
                "client_id":     self._client_id,
                "client_secret": self._client_secret,
            },
            timeout=10,
        )
        res.raise_for_status()
        payload = res.json()
        self._token        = payload["access_token"]
        self._token_expiry = time.time() + payload.get("expires_in", 1800) - 30

    def _ensure_token(self):
        if not self._token or time.time() >= self._token_expiry:
            self._refresh_token()

    def get(self, path: str, params: dict | None = None) -> list | dict:
        self._ensure_token()
        url = _BASE_URL + path
        res = requests.get(
            url,
            headers={"Authorization": f"Bearer {self._token}"},
            params={**(params or {}), "$format": "JSON"},
            timeout=30,
        )
        res.raise_for_status()
        return res.json()
