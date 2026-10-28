"""Local entry point. Training runs only when explicitly selected."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("INVESTMENT_AI_HOME", str(ROOT / "runtime"))
command = sys.argv[1] if len(sys.argv) > 1 else "api"
arguments = sys.argv[2:]
if command == "api":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000)
elif command == "ui":
    sys.argv = ["streamlit", "run", "frontend/app.py", "--server.port", "8501", *arguments]
    from streamlit.web.cli import main
    main()
elif command == "test":
    import pytest
    raise SystemExit(pytest.main(["tests", "-q", *arguments]))
elif command in {"train", "evaluate", "infer", "preflight", "news_search", "prepare_market"}:
    if command in {"train", "preflight"} and "--config" not in arguments:
        arguments = ["--config", str(ROOT / "config/investment_ai.json"), *arguments]
    sys.argv = [command, *arguments]
    runpy.run_module("backend.investment_ai." + command, run_name="__main__")
else:
    raise SystemExit("Choose api, ui, test, train, preflight, evaluate or infer")
