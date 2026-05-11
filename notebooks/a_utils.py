import logging
import math
import os
import shap
import re

import numpy as np
import pandas as pd

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from typing import List

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

    for i, val in enumerate(y):
        ax.text(val + 0.01, i, f'{val * 100:.1f}%', va='center', fontsize=9)

    plt.xlabel('Fill Rate')
    plt.yticks(x, cols)
    plt.title(title)
    add_legend()
    plt.show()

# ---------------------------------------------------------------------------
# Feature importance analysis
# ---------------------------------------------------------------------------

SEVERITY_COLS = {'cough_severity', 'sob_severity'}
NUMERIC_COLS = set(VITALS) | {'age', 'days_since_symptom_onset'}
FEATURE_COLS = SYMPTOMS + VITALS + COMORBIDITIES + RISKS

LOW_FILL_THRESHOLD = 0.10


def _get_group_label(col: str) -> str:
    if col in SYMPTOMS:      return 'Symptoms'
    if col in VITALS:        return 'Vitals'
    if col in COMORBIDITIES: return 'Comorbidities'
    if col in RISKS:         return 'Epi Factors'
    return 'Other'


def _build_X_y(data: pd.DataFrame):
    """Encode features and labels for model training.

    Returns (X, y) filtered to rows where the label is non-null.
    """
    X = data[FEATURE_COLS].copy()
    for col in FEATURE_COLS:
        if col in SEVERITY_COLS:
            X[col] = X[col].map(SEVERITY_MAPPINGS).fillna(0).astype(int)
        elif col in NUMERIC_COLS:
            X[col] = X[col].fillna(X[col].median())
        else:
            # NaN encoded as -1: "not recorded" is clinically distinct from False
            X[col] = X[col].map({True: 1, False: 0}).fillna(-1).astype(int)

    y = (data[LABEL] == 'Positive').astype(int)
    mask = data[LABEL].notna()
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


def build_model(data: pd.DataFrame):
    """Train a RandomForest on FEATURE_COLS and return (classifier, X_train, y_train, X_test, y_test, fill_rates)."""
    fill_rates = {col: data[col].notna().mean() for col in FEATURE_COLS}
    X, y = _build_X_y(data)
    logging.info(f'Training RandomForest on {len(y)} rows, {y.sum()} positives ({y.mean():.1%})')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    logging.info(
        f'Train: {len(y_train)} rows, {y_train.sum()} positives | '
        f'Test: {len(y_test)} rows, {y_test.sum()} positives'
    )

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train)
    logging.info('Training complete.')
    return classifier, X_train, y_train, X_test, y_test, fill_rates


def compute_feature_importance(classifier: RandomForestClassifier, X: pd.DataFrame, fill_rates: dict) -> pd.DataFrame:
    """
    Extract MDI feature importances from a trained RandomForest.

    Returns a DataFrame with columns: feature, importance, fill_rate, group,
    sorted descending by importance.
    """
    result = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': classifier.feature_importances_,
        'fill_rate': [fill_rates[c] for c in FEATURE_COLS],
        'group': [_get_group_label(c) for c in FEATURE_COLS],
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    logging.info('Top 5 features:\n' + result.head().to_string(index=False))
    return result


def plot_feature_importance(importance_df: pd.DataFrame) -> None:
    """Horizontal bar chart of feature importances, sorted descending."""
    df = importance_df.sort_values('importance', ascending=True)
    colors = [get_color(col) for col in df['feature']]

    _, ax = plt.subplots(figsize=(7, 10), facecolor='white')
    ax.set_facecolor('white')
    ax.barh(range(len(df)), df['importance'], color=colors)

    for i, val in enumerate(df['importance']):
        ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8)

    plt.yticks(range(len(df)), df['feature'])
    plt.xlabel('Feature Importance (MDI)')
    plt.title('Feature Importance for Predicting COVID-19 Positive Result')
    add_legend()
    plt.tight_layout()
    plt.show()


def plot_importance_vs_fill_rate(importance_df: pd.DataFrame) -> None:
    """Scatter plot: X=fill rate (availability), Y=importance (predictive power).

    Top-right quadrant = high importance AND high availability (most actionable).
    Hollow markers flag features with fill_rate < LOW_FILL_THRESHOLD (sparse data).
    """
    _, ax = plt.subplots(figsize=(11, 7), facecolor='white')
    ax.set_facecolor('white')

    ax.axvspan(0, LOW_FILL_THRESHOLD, color='lightgray', alpha=0.3, zorder=0)

    for _, row in importance_df.iterrows():
        is_sparse = row['fill_rate'] < LOW_FILL_THRESHOLD
        color = get_color(row['feature'])
        ax.scatter(
            row['fill_rate'], row['importance'],
            s=80, zorder=3,
            facecolors='none' if is_sparse else color,
            edgecolors=color,
            linewidths=1.5,
        )
        ax.annotate(
            row['feature'], (row['fill_rate'], row['importance']),
            textcoords='offset points', xytext=(5, 5), fontsize=8,
        )

    med_fill = importance_df['fill_rate'].median()
    med_imp = importance_df['importance'].median()
    ax.axvline(med_fill, linestyle='--', color='gray', alpha=0.5)
    ax.axhline(med_imp, linestyle='--', color='gray', alpha=0.5)
    ax.text(
        0.99, 0.99, 'High priority', transform=ax.transAxes,
        ha='right', va='top', fontsize=9, color='gray',
    )

    plt.xlabel('Fill Rate (data availability)')
    plt.ylabel('Feature Importance (MDI)')
    plt.title(
        'Feature Importance vs. Data Availability\n'
        '(hollow markers = sparse data, <10% fill rate)'
    )
    add_legend()
    plt.tight_layout()
    plt.show()

def compute_shap_importance(
    classifier: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    fill_rates: dict,
    neg_pos_multiplier: int = 3,
):
    """
    Compute SHAP feature importances via TreeExplainer on a trained RandomForest.

    Uses stratified sampling (all positives + neg_pos_multiplier × n_positives negatives)
    to keep runtime manageable on imbalanced datasets while preserving minority-class signal.

    Returns (shap_df, shap_values, X_shap) where shap_df has columns:
    feature, importance (mean |SHAP|), fill_rate, group — sorted descending.
    shap_values and X_shap are passed directly to plot_shap_summary.
    """
    pos_idx = y[y == 1].index
    neg_idx = y[y == 0].sample(
        n=min(len(pos_idx) * neg_pos_multiplier, (y == 0).sum()), random_state=42
    ).index
    sample_idx = pos_idx.union(neg_idx)
    X_shap = X.loc[sample_idx]
    logging.info(
        f'SHAP sample: {(y.loc[sample_idx] == 1).sum()} pos, {(y.loc[sample_idx] == 0).sum()} neg'
    )

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_shap, check_additivity=False)

    # shap_values is [neg_class, pos_class] for binary RF — use positive class
    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)

    result = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': mean_abs_shap,
        'fill_rate': [fill_rates[c] for c in FEATURE_COLS],
        'group': [_get_group_label(c) for c in FEATURE_COLS],
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    logging.info('Top 5 SHAP features:\n' + result.head().to_string(index=False))
    return result, sv, X_shap


def plot_shap_importance(shap_df: pd.DataFrame) -> None:
    """Horizontal bar chart of mean absolute SHAP values, styled like plot_feature_importance."""
    df = shap_df.sort_values('importance', ascending=True)
    colors = [get_color(col) for col in df['feature']]

    _, ax = plt.subplots(figsize=(7, 10), facecolor='white')
    ax.set_facecolor('white')
    ax.barh(range(len(df)), df['importance'], color=colors)

    for i, val in enumerate(df['importance']):
        ax.text(val + 0.0005, i, f'{val:.4f}', va='center', fontsize=8)

    plt.yticks(range(len(df)), df['feature'])
    plt.xlabel('Mean |SHAP Value|')
    plt.title('SHAP Feature Importance for Predicting COVID-19 Positive Result')
    add_legend()
    plt.tight_layout()
    plt.show()


def plot_shap_summary(shap_values: np.ndarray, X: pd.DataFrame) -> None:
    """SHAP beeswarm summary plot showing both magnitude and direction of feature impact."""
    import shap
    shap.summary_plot(shap_values, X, show=True)


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def compute_column_outcome_correlation(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """Compute point-biserial correlation between a boolean column and the COVID-19 outcome.

    Only rows where both `col` and the outcome label are non-null are used.
    Returns a DataFrame with one row per outcome value (Positive / Negative) showing the
    True rate for `col`, plus overall correlation statistics.
    Columns: outcome, n_total, n_true, pct_true, correlation, p_value.
    """
    mask = data[LABEL].notna() & data[col].notna()
    df = data.loc[mask, [LABEL, col]].copy()
    df['_col_binary'] = df[col].map({True: 1, False: 0}).astype(float)

    rows = []
    for outcome_val in LABEL_VALUES:
        subset = df[df[LABEL] == outcome_val]
        n_total = len(subset)
        n_true = int(subset['_col_binary'].sum())
        rows.append({
            'outcome': outcome_val,
            'n_total': n_total,
            'n_true': n_true,
            'pct_true': (n_true / n_total * 100) if n_total else float('nan'),
        })

    corr, p_val = stats.pointbiserialr(df['_col_binary'], (df[LABEL] == 'Positive').astype(float))
    logging.info(f'{col} vs outcome: r={corr:.4f}, p={p_val:.4f}, n={len(df):,}')

    result = pd.DataFrame(rows)
    result['correlation'] = corr
    result['p_value'] = p_val
    return result


def plot_outcome_correlation(
    data: pd.DataFrame,
    group_col: str,
    rate_col: str,
    group_values: list,
    rate_positive_value,
    group_positive_value='Positive',
    xlabel: str = '',
    ylabel: str = 'Rate (%)',
    title: str = '',
) -> None:
    """Grouped bar chart: rate of rate_col == rate_positive_value for each category of group_col.

    Each bar shows the percentage within that group, annotated with the count.
    The overall point-biserial correlation and p-value are shown in the title.

    group_values defines the x-axis order and also acts as a filter on group_col.
    group_positive_value is the category treated as "positive" for the correlation direction.
    """
    mask = data[group_col].isin(group_values) & data[rate_col].notna()
    df = data.loc[mask, [group_col, rate_col]].copy()

    corr, p_val = stats.pointbiserialr(
        (df[group_col] == group_positive_value).astype(float),
        (df[rate_col] == rate_positive_value).astype(float),
    )

    summary = (
        df.groupby(group_col)[rate_col]
        .agg(pct=lambda x: (x == rate_positive_value).mean() * 100, n='count')
        .reindex(group_values)
        .reset_index()
    )

    _, ax = plt.subplots(figsize=(6, 4), facecolor='white')
    ax.set_facecolor('white')

    bars = ax.bar(
        summary[group_col], summary['pct'],
        color=[COLOR_PALETTE[4], COLOR_PALETTE[1]],
        width=0.4,
    )
    for bar, row in zip(bars, summary.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{row.pct:.1f}%\n(n={row.n:,})',
            ha='center', va='bottom', fontsize=10,
        )

    sig = f'p={p_val:.4f}' + (' *' if p_val < 0.05 else '')
    plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)
    plt.title(f'{title}\n(r={corr:.3f}, {sig})')
    plt.ylim(0, summary['pct'].dropna().max() * 1.35 if not summary['pct'].dropna().empty else 10)
    plt.tight_layout()
    plt.show()


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
