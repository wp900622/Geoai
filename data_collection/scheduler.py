"""
scheduler.py – 每 INTERVAL_SEC 秒呼叫一次 fetch_and_save_taipei_traffic()，
更新 taipei_live_traffic.csv / taipei_traffic_links.csv。

使用方式：
  cd data_collection
  python scheduler.py              # 預設 60 秒一次
  python scheduler.py --interval 300   # 改 5 分鐘一次

Ctrl+C 可優雅停止。單次抓取失敗不會終止排程，錯誤會記錄到 log。
"""
import argparse
import logging
import time

from fetch_traffic import fetch_and_save_taipei_traffic

DEFAULT_INTERVAL_SEC = 300

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scheduler")


def run(interval_sec: int) -> None:
    log.info(f"Scheduler starting (interval={interval_sec}s). Press Ctrl+C to stop.")
    tick = 0
    while True:
        tick += 1
        start = time.time()
        log.info(f"--- Tick {tick}: fetching traffic data ---")
        try:
            fetch_and_save_taipei_traffic()
        except Exception as exc:  # noqa: BLE001
            # 單次抓取失敗就記錄並繼續,避免排程因網路/API 暫時性錯誤掛掉
            log.exception(f"Fetch failed: {exc}")

        elapsed = time.time() - start
        sleep_for = max(0.0, interval_sec - elapsed)
        log.info(f"Tick {tick} done in {elapsed:.1f}s. Next fetch in {sleep_for:.1f}s.")
        try:
            time.sleep(sleep_for)
        except KeyboardInterrupt:
            log.info("Scheduler stopped by user.")
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="Taipei traffic data scheduler")
    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_SEC,
        help=f"Fetch interval in seconds (default: {DEFAULT_INTERVAL_SEC})",
    )
    args = parser.parse_args()

    try:
        run(args.interval)
    except KeyboardInterrupt:
        log.info("Scheduler stopped by user.")


if __name__ == "__main__":
    main()
