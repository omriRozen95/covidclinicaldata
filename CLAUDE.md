# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

COVID-19 clinical data analysis repository combining data from **Carbon Health** (clinical characteristics, lab findings) and **Braid Health** (chest X-rays). Data is HIPAA-compliant and de-identified. Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0.

## Environment Setup

```bash
virtualenv --system-site-packages -p python3 ./venv
source ./venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
jupyter notebook
```

On Windows, activate with `.\venv\Scripts\activate`.

## Running the Analysis

Open and run `notebooks/b_load_plot.ipynb` in Jupyter. There are no automated tests or CI pipelines — analysis is exploratory and executed manually through notebooks.

## Architecture

```
data/                        # Weekly CSV batches (04-07 through 10-20)
notebooks/
  a_utils.py                 # Utility library: constants, data loading, filtering, plotting
  b_load_plot.ipynb          # Main analysis notebook
```

**Data pipeline:** `a_utils.open_data()` concatenates all CSVs from `../data/` → filtering functions isolate cohorts (positives, symptomatic, patients with vitals) → derived columns are computed → `plot_fill_rates()` generates completeness visualizations.

### Key constants in [a_utils.py](notebooks/a_utils.py)

Column group constants (`SYMPTOMS`, `VITALS`, `COMORBIDITIES`, `RISKS`, `TEST_RESULTS`, `CXR_FIELDS`) define the feature groups used throughout analysis. Any new data fields should be added to the appropriate constant.

### CXR classification

`is_abnormal_cxr()` uses regex pattern lists (`ABNORMALITIES`, `NO_ABNORMALITIES`) to classify free-text X-ray impression fields. Patterns are applied in order — `NO_ABNORMALITIES` takes precedence as an override list.

### Symptom severity scoring

`get_symptom_severity_score()` maps cough severity, shortness of breath, and fever presence to an ordinal severity score. The scoring logic is in `a_utils.py` and is the primary derived clinical metric used in the notebook.
