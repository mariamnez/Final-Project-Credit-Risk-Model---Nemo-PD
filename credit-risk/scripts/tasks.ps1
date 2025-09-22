param([Parameter()] [string] $Task = "help")

switch ($Task) {
  "setup"     { python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -U pip; pip install -r requirements.txt; pre-commit install }
  "lint"      { ruff check src; black --check src }
  "format"    { black src; ruff check --fix src }
  "abt"       { python -m src.data.build_abt }
  "features"  { python -m src.features.build_features }
  "train"     { python -m src.models.train }
  "evaluate"  { python -m src.evaluation.evaluate }
  "policy"    { python -m src.policy.thresholding }
  "dashboard" { streamlit run src/app/streamlit_app.py }
  "serve"     { uvicorn src.api.main:app --reload --port 8080 }
  default     { "Usage: .\scripts\tasks.ps1 [setup|lint|format|abt|features|train|evaluate|policy|dashboard|serve]" }
}
