import logging
import math
import os
import re

import numpy as np
import pandas as pd

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from typing import List

plt.style.use('fivethirtyeight')

logging.getLogger().setLevel('INFO')

# ---------------------------------------------------------------------------
# Data path & label constants
# ---------------------------------------------------------------------------

PATH = '../data'

LABEL = 'covid19_test_results'
LABEL_VALUES = ['Negative', 'Positive']

# ---------------------------------------------------------------------------
# Feature column groups
# ---------------------------------------------------------------------------

SYMPTOMS = [
    'labored_respiration',
    'rhonchi',
    'wheezes',
    'cough',
    'cough_severity',
    'loss_of_smell',
    'loss_of_taste',
    'runny_nose',
    'muscle_sore',
    'sore_throat',
    'fever',
    'sob',
    'sob_severity',
    'diarrhea',
    'fatigue',
    'headache',
    'ctab',
    'days_since_symptom_onset',
]

VITALS = [
    'temperature',
    'pulse',
    'sys',
    'dia',
    'rr',
    'sats',
]

COMORBIDITIES = [
    'diabetes',
    'chd',
    'htn',
    'cancer',
    'asthma',
    'copd',
    'autoimmune_dis',
    'smoker',
]

RISKS = [
    'age',
    'high_risk_exposure_occupation',
    'high_risk_interactions',
]

TEST_RESULTS = [
    'batch_date',
    LABEL,
    'rapid_flu_results',
    'rapid_strep_results',
    'swab_type',
    'test_name',
]

CXR_FIELDS = [
    'cxr_findings', 'cxr_impression', 'cxr_label', 'cxr_link',
]

# ---------------------------------------------------------------------------
# Visualization constants
# ---------------------------------------------------------------------------

LABELS = [
    'Test Results',
    'Epi Factors',
    'Comorbidities',
    'Vitals',
    'Symptoms',
    'Radiological Findings',
    'Other'
]

COLOR_PALETTE = sns.color_palette('husl', len(LABELS))
COLOR_PALETTE[-1] = 'gray'

# ---------------------------------------------------------------------------
# CXR analysis constants
# ---------------------------------------------------------------------------

ABNORMALITIES = [
    r'.+(lobe|RML|peribronchial|basilar) infiltrate',
    'lobe scarring or atelectasis',
    r'(perihilar|Trace).+opacity',
    'Peribronchial thickeneing',
    'Left lower lobe consolidation',
    r'Consolidation in the.+lung',
    r'(?<!No )(Multifocal|lung|pulmonary).+opacities',
    'left pulmonary nodules',
    r'(?<!no ) opacity',
    r'.?(left|Left) lung base',
    r'(Subtle left basilar|mass-like spiculated) density',
    'basilar atelectasis or scarring',
    'Elevated right hemidiaphragm',
    '(right hilar|septal) prominence',
]

NO_ABNORMALITIES = [
    r'No.+(acute|significant|definite|suspicious).+(abnormality|disease|opacities)',
    'Normal',
    'No pulmonary opacities visualized',
    'No evidence of acute cardiopulmonary disease',
    'No lobar consolidation',
]

# ---------------------------------------------------------------------------
# Symptom severity constants
# ---------------------------------------------------------------------------

SEVERITY_MAPPINGS = {
    'Mild': 1,
    'Moderate': 2,
    'Severe': 3,
}

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def get_percent(x: int, total: int) -> float:
    """Return x/total as a percentage, or 0 if total is zero."""
    if total == 0:
        logging.info(f'Returning 0 to avoid `division by 0` error.')
        return 0
    return (x / total) * 100


def is_any_true(row: pd.Series, cols: List[str]) -> bool:
    """Return True if any of the given boolean columns is True for this row."""
    return any(row[col] == True for col in cols)


def is_any_nonnull(row: pd.Series, cols: List[str]) -> bool:
    """Return True if any of the given numeric columns is non-NaN for this row."""
    return any(not math.isnan(row[col]) for col in cols)

# ---------------------------------------------------------------------------
# Data loading & inspection
# ---------------------------------------------------------------------------

def open_data() -> pd.DataFrame:
    """Load and concatenate all CSV files found under PATH."""
    return pd.concat(
        [
            pd.read_csv(f'{PATH}/{filename}')
            for filename in os.listdir(PATH)
            if filename.endswith('.csv')
        ]
    )


def log_column_names_with_single_unique_value(data: pd.DataFrame):
    """Log any columns that contain only a single unique value across the dataset."""
    for col in data.columns:
        if len(data[col].unique()) == 1:
            logging.info(
                f'`{col}` only has single unique value of {data[col].iloc[0]} '
                'in entire dataset.'
            )

# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def keep_positives(data: pd.DataFrame) -> pd.DataFrame:
    """Return only rows where the COVID-19 test result is Positive."""
    return data[data[LABEL] == 'Positive']


def keep_rows_with_any_filled(
    df: pd.DataFrame, cols_to_check: List[str], col_type: str = 'bool'
) -> pd.DataFrame:
    """Return rows where at least one of cols_to_check is filled / True.

    col_type must be 'bool' (keeps rows with any True value) or 'numeric'
    (keeps rows with any non-NaN value).
    """
    logging.info('Filtering out patients...')

    if col_type == 'bool':
        f = is_any_true
    elif col_type == 'numeric':
        f = is_any_nonnull
    else:
        logging.info('ERROR: `col_type` should be either `bool` or `numeric`.')
        return None

    df_filtered = df[df.apply(lambda x: f(x, cols_to_check), axis=1)]
    logging.info(
        f'    ---- {len(df)} --> {len(df_filtered)} '
        f'({get_percent(len(df_filtered), len(df)):.2f}%)'
    )
    return df_filtered

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def get_color(col: str) -> str:
    """Return the palette color assigned to a column based on its feature group."""
    if col in TEST_RESULTS:
        return COLOR_PALETTE[0]
    if col in RISKS:
        return COLOR_PALETTE[1]
    if col in COMORBIDITIES:
        return COLOR_PALETTE[2]
    if col in VITALS:
        return COLOR_PALETTE[3]
    if col in SYMPTOMS:
        return COLOR_PALETTE[4]
    if col in CXR_FIELDS:
        return COLOR_PALETTE[5]
    return 'gray'


def add_legend():
    """Attach a color-coded legend for all feature groups to the current axes."""
    mappings = {
        label: COLOR_PALETTE[i] for i, label in enumerate(LABELS)
    }
    patches = [
        mpatches.Patch(color=color, label=label)
        for label, color in mappings.items()
    ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1))


def plot_fill_rates(data: pd.DataFrame, title: str = ''):
    """Plot a horizontal bar chart showing the fill rate of each column in data."""
    total = len(data)
    cols = data.columns

    _, ax = plt.subplots(figsize=(7, 15), facecolor='white')
    ax.set_facecolor('white')

    x = range(len(cols))
    y = [sum(~data[col].isnull()) / total for col in cols]
    colors = [get_color(col) for col in cols]

    ax.barh(list(x), y, color=colors)

    plt.xlabel('Fill Rate')
    plt.yticks(x, cols)
    plt.title(title)
    add_legend()
    plt.show()

# ---------------------------------------------------------------------------
# CXR analysis
# ---------------------------------------------------------------------------

def is_abnormal_cxr(cxr_imp: str) -> bool:
    """
    Classify a CXR impression string as abnormal (True), normal (False), or unknown (None).

    Checks NO_ABNORMALITIES patterns first; if none match, checks ABNORMALITIES patterns.
    Returns None when the impression cannot be classified by either pattern list.
    """
    if any(re.search(x, cxr_imp) for x in NO_ABNORMALITIES):
        return False
    if any(re.search(x, cxr_imp) for x in ABNORMALITIES):
        return True
    return None

# ---------------------------------------------------------------------------
# Symptom severity
# ---------------------------------------------------------------------------

def get_symptom_severity_score(row: pd.Series) -> int:
    """Compute a numeric severity score from cough/SOB severity and fever presence.

    Returns -1 for asymptomatic patients (num_symptoms == 0).
    """
    if row['num_symptoms'] == 0:
        return -1

    return (
        SEVERITY_MAPPINGS.get(row['cough_severity'], 0) +
        SEVERITY_MAPPINGS.get(row['sob_severity'], 0) +
        (row['fever'] == True)
    )


def get_sym_severity(score: int) -> str:
    """Map a numeric severity score to a human-readable severity label."""
    if score < 0:
        return 'Asymptomatic'
    if score < 1:
        return 'Extremely Mild'
    if score < 2:
        return 'Mild'
    if score < 3:
        return 'Moderate'
    else:
        return 'Severe'
