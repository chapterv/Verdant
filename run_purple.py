"""Launch Verdant Purple assessee service."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn
import yaml

from src.purple_service import create_purple_app


def load_service_config(config_path: Path) -> tuple[str, int]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    service = config.get("service", {}) or {}
    host = service.get("host", "0.0.0.0")
    port = int(service.get("port", 8001))
    return host, port


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Verdant Purple service")
    parser.add_argument("--config", default="config/purple_config.yaml", help="Path to config yaml")
    parser.add_argument("--host", default=None, help="Override host")
    parser.add_argument("--port", type=int, default=None, help="Override port")
    args = parser.parse_args()

    config_path = Path(args.config)
    default_host, default_port = load_service_config(config_path)
    env_host = os.getenv("HOST")
    env_port_raw = os.getenv("PORT")
    env_port = int(env_port_raw) if env_port_raw and env_port_raw.isdigit() else None

    host = args.host or env_host or default_host
    port = args.port or env_port or default_port

    app = create_purple_app(config_path=str(config_path))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
