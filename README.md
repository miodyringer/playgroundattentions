# Setup Instructions

Clone the repository from GitHub.

```
git clone https://github.com/miodyringer/playgroundattentions.git
cd playgroundattentions
```

Create a virtual environment.

```
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Install all dependencies.

```
pip install -r requirements.txt
```

Configure environment in `.env` (example in `.env.example`). `HF_TOKEN` only necessary when using a mistral model. Compatible models are listed on https://transformerlensorg.github.io/TransformerLens/generated/model_properties_table.html

```
(optional)HF_TOKEN="..."
(optional)COMPARE_MODEL="..."
MODEL_NAME="..."
```

Run the application and open index.html in your browser.
```
python api.py
```
