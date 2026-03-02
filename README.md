# VivaGene

VivaGene is a Streamlit-based educational genomics app that converts uploaded genotype files into trait summaries across Neurobehavior, Nutrition, Fitness, and optional Liver panels. Reports are evidence-aware, citation-linked where available, and designed for education only (not diagnosis or treatment guidance).

## Project Structure

- App entrypoint: `app.py`
- Core interpretation engine: `genomics_interpreter.py`
- Trait data: `data/traits.json`, `trait_database.csv`, `trait_study_packs/`
- Evidence corpus/cache: `evidence_corpus/`
- Scripts: `scripts/`

## Local Run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set secrets as environment variables (or local Streamlit secrets file):

```bash
export OPENROUTER_API_KEY="your_key_here"
# optional
export OPENROUTER_MODEL="mistralai/mistral-7b-instruct"
export EUROPE_PMC_EMAIL="you@example.com"
```

4. Run:

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to Streamlit Community Cloud and sign in.
3. Click **Create app**.
4. Select your GitHub repository and branch.
5. Set **Main file path** to `app.py`.
6. In app settings, open **Secrets** and paste TOML keys (example below).
7. Click **Deploy**.

Streamlit Community Cloud automatically redeploys on new pushes to the selected branch.

## Secrets Setup (Community Cloud)

Set secrets in **App settings → Secrets** (do not commit keys to git):

```toml
OPENROUTER_API_KEY = "your_openrouter_key"
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"  # optional override
EUROPE_PMC_EMAIL = "you@example.com"                 # optional
DEV_MODE = false                                      # optional
SHOW_VALIDATION = 0                                   # optional
```

VivaGene reads secrets from `st.secrets` first and falls back to environment variables for local runs.

## GitHub Push Commands

If repository is not initialized:

```bash
git init
git add .
git commit -m "Deploy VivaGene"
git branch -M main
git remote add origin <PLACEHOLDER_GITHUB_URL>
git push -u origin main
```

If already initialized, use:

```bash
git add .
git commit -m "Deploy VivaGene"
git push
```

## Notes on Privacy and Use

- Educational genomics insights only.
- Not medical advice.
- No diagnosis or treatment recommendations.
- Do not upload sensitive files to public repositories.

## Quick Troubleshooting (Cloud)

If you see a blank page or startup failure:

1. Check app logs in Streamlit Community Cloud.
2. Confirm `requirements.txt` installed cleanly.
3. Confirm main file path is `app.py`.
4. Confirm required secrets (especially `OPENROUTER_API_KEY`) are set.
5. Confirm required data files are present in the repo (`data/traits.json`, `trait_study_packs/`, etc.).
