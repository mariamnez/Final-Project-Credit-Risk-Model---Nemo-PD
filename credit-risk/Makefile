PY=python
ACTIVATE=. ./.venv/bin/activate

setup:
	python -m venv .venv && $(ACTIVATE) && pip install -U pip && pip install -r requirements.txt && pre-commit install

lint:
	ruff check src && black --check src

format:
	black src && ruff check --fix src

abt:
	$(PY) -m src.data.build_abt

features:
	$(PY) -m src.features.build_features

train:
	$(PY) -m src.models.train

evaluate:
	$(PY) -m src.evaluation.evaluate

policy:
	$(PY) -m src.policy.thresholding

dashboard:
	streamlit run src/app/streamlit_app.py

serve:
	uvicorn src.api.main:app --reload --port 8080

clean:
	rm -rf mlruns models/*.pkl data/processed/*.parquet
