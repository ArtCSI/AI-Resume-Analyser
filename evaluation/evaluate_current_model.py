import sys
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# PATH SETUP
# ============================================================

# evaluation/
#     evaluate_current_model.py
#
# Project root is one level above evaluation/

EVALUATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVALUATION_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT YOUR EXISTING MATCHER
# ============================================================

print("Loading your existing Resume Analyzer matcher...")

from matcher import compute_similarity


print("Matcher loaded successfully.")


# ============================================================
# PATHS
# ============================================================

INPUT_FILE = (
    EVALUATION_DIR
    / "dataset"
    / "processed"
    / "resume_vacancy_evaluation.csv"
)

RESULTS_DIR = (
    EVALUATION_DIR
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "current_model_predictions.csv"
)


# ============================================================
# LOAD EVALUATION DATASET
# ============================================================

print("\nLoading evaluation dataset...")

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Evaluation dataset not found:\n{INPUT_FILE}"
    )


df = pd.read_csv(INPUT_FILE)


print(
    f"Loaded {len(df)} evaluation pairs."
)

print(
    f"Unique resumes: "
    f"{df['resume_id'].nunique()}"
)

print(
    f"Unique vacancies: "
    f"{df['vacancy_id'].nunique()}"
)


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "resume_id",
    "vacancy_id",
    "vacancy_title",
    "resume_text",
    "job_description",
    "annotator_1_rank",
    "annotator_2_rank",
    "average_human_rank"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# RUN CURRENT MODEL
# ============================================================

print("\n" + "=" * 70)

print(
    "RUNNING CURRENT RESUME ANALYZER"
)

print("=" * 70)

print(
    "This will evaluate every resume against "
    "all five vacancies."
)

print()


predictions = []

total = len(df)


for index, row in df.iterrows():

    resume_id = int(
        row["resume_id"]
    )

    vacancy_id = int(
        row["vacancy_id"]
    )

    vacancy_title = row[
        "vacancy_title"
    ]

    resume_text = str(
        row["resume_text"]
    )

    job_description = str(
        row["job_description"]
    )


    print(
        f"[{index + 1}/{total}] "
        f"CV {resume_id} → "
        f"Vacancy {vacancy_id}: "
        f"{vacancy_title}"
    )


    try:

        score = compute_similarity(
            resume_text,
            job_description
        )

        score = float(score)

        error = ""

    except Exception as e:

        print(
            f"   ERROR: {e}"
        )

        score = np.nan

        error = str(e)


    predictions.append({

        "resume_id":
            resume_id,

        "vacancy_id":
            vacancy_id,

        "vacancy_title":
            vacancy_title,

        "annotator_1_rank":
            row["annotator_1_rank"],

        "annotator_2_rank":
            row["annotator_2_rank"],

        "average_human_rank":
            row["average_human_rank"],

        "model_score":
            score,

        "error":
            error
    })


# ============================================================
# CREATE PREDICTION DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    predictions
)


# ============================================================
# CHECK FOR FAILED PREDICTIONS
# ============================================================

failed_predictions = results_df[
    results_df["model_score"].isna()
]

if len(failed_predictions) > 0:

    print(
        f"\nWARNING: "
        f"{len(failed_predictions)} "
        f"predictions failed."
    )

else:

    print(
        "\nAll model predictions completed successfully."
    )


# ============================================================
# RANK VACANCIES FOR EACH RESUME
# ============================================================

print(
    "\nCreating model rankings..."
)


# Higher model score = better match
#
# Therefore:
#
# highest score → model rank 1
# second highest → model rank 2
# etc.

results_df["model_rank"] = (
    results_df
    .groupby("resume_id")["model_score"]
    .rank(
        ascending=False,
        method="average"
    )
)


# ============================================================
# ADD HUMAN BEST VACANCY
# ============================================================

# Lower human rank = better match
#
# Human rank 1 = best vacancy


results_df["human_rank"] = (
    results_df["average_human_rank"]
)


# ============================================================
# SAVE RAW RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# DISPLAY EXAMPLE RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "EXAMPLE MODEL RESULTS"
)

print("=" * 70
)


for resume_id in sorted(
    results_df["resume_id"].unique()
)[:5]:

    resume_results = (
        results_df[
            results_df["resume_id"] == resume_id
        ]
        .sort_values(
            "model_score",
            ascending=False
        )
    )


    print(
        f"\nCV {resume_id}"
    )

    print(
        resume_results[
            [
                "vacancy_id",
                "vacancy_title",
                "model_score",
                "model_rank",
                "human_rank"
            ]
        ].to_string(index=False)
    )


# ============================================================
# RANKING METRICS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CALCULATING RANKING METRICS"
)

print("=" * 70
)


# ------------------------------------------------------------
# Try importing scipy
# ------------------------------------------------------------

try:

    from scipy.stats import (
        spearmanr,
        kendalltau
    )

    SCIPY_AVAILABLE = True

except ImportError:

    SCIPY_AVAILABLE = False

    print(
        "\nWARNING: scipy is not installed."
    )

    print(
        "Install it with:"
    )

    print(
        "python -m pip install scipy"
    )


# ============================================================
# IDENTIFY ANOMALOUS HUMAN RANKINGS
# ============================================================

anomalous_cvs = []

for resume_id in sorted(
    results_df["resume_id"].unique()
):

    resume_rows = results_df[
        results_df["resume_id"] == resume_id
    ]

    for annotator_column in [
        "annotator_1_rank",
        "annotator_2_rank"
    ]:

        ranking = (
            resume_rows[
                annotator_column
            ]
            .tolist()
        )

        ranking = [
            int(x)
            for x in ranking
        ]

        if sorted(ranking) != [
            1, 2, 3, 4, 5
        ]:

            if resume_id not in anomalous_cvs:

                anomalous_cvs.append(
                    resume_id
                )


# ============================================================
# CLEAN DATASET FOR PRIMARY EVALUATION
# ============================================================

clean_results = results_df[
    ~results_df["resume_id"].isin(
        anomalous_cvs
    )
].copy()


print(
    f"\nAnomalous CVs excluded "
    f"from primary ranking metrics: "
    f"{anomalous_cvs}"
)

print(
    f"Clean CVs used for primary evaluation: "
    f"{clean_results['resume_id'].nunique()}"
)


# ============================================================
# CALCULATE METRICS PER RESUME
# ============================================================

spearman_scores = []

kendall_scores = []

top1_matches = []



for resume_id in sorted(
    clean_results["resume_id"].unique()
):

    resume_results = clean_results[
        clean_results["resume_id"] == resume_id
    ].copy()


    # --------------------------------------------------------
    # Human ranking
    #
    # Lower = better
    # --------------------------------------------------------

    human_ranks = (
        resume_results
        .sort_values("vacancy_id")
        ["human_rank"]
        .values
    )


    # --------------------------------------------------------
    # Model ranking
    #
    # Lower = better
    # --------------------------------------------------------

    model_ranks = (
        resume_results
        .sort_values("vacancy_id")
        ["model_rank"]
        .values
    )


    # --------------------------------------------------------
    # Spearman
    # --------------------------------------------------------

    if SCIPY_AVAILABLE:

        spearman_result = spearmanr(
            human_ranks,
            model_ranks
        )

        spearman_scores.append(
            spearman_result.statistic
        )


    # --------------------------------------------------------
    # Kendall
    # --------------------------------------------------------

    if SCIPY_AVAILABLE:

        kendall_result = kendalltau(
            human_ranks,
            model_ranks
        )

        kendall_scores.append(
            kendall_result.statistic
        )


    # --------------------------------------------------------
    # Top-1 agreement
    # --------------------------------------------------------

    human_best = (
        resume_results
        .sort_values(
            [
                "human_rank",
                "vacancy_id"
            ],
            ascending=[
                True,
                True
            ]
        )
        .iloc[0]
        ["vacancy_id"]
    )


    model_best = (
        resume_results
        .sort_values(
            [
                "model_score",
                "vacancy_id"
            ],
            ascending=[
                False,
                True
            ]
        )
        .iloc[0]
        ["vacancy_id"]
    )


    top1_matches.append(
        int(
            human_best == model_best
        )
    )


# ============================================================
# FINAL METRICS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "CURRENT MODEL EVALUATION"
)

print("=" * 70
)


if SCIPY_AVAILABLE:

    mean_spearman = np.mean(
        spearman_scores
    )

    mean_kendall = np.mean(
        kendall_scores
    )

    print(
        f"Mean Spearman correlation: "
        f"{mean_spearman:.4f}"
    )

    print(
        f"Mean Kendall tau: "
        f"{mean_kendall:.4f}"
    )


top1_accuracy = np.mean(
    top1_matches
)


print(
    f"Top-1 agreement: "
    f"{top1_accuracy:.4f}"
)

print(
    f"Top-1 agreement percentage: "
    f"{top1_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "total_evaluation_pairs":
        len(results_df),

    "total_resumes":
        results_df["resume_id"].nunique(),

    "total_vacancies":
        results_df["vacancy_id"].nunique(),

    "anomalous_cvs_excluded":
        len(anomalous_cvs),

    "clean_resumes":
        clean_results["resume_id"].nunique(),

    "top1_accuracy":
        top1_accuracy
}


if SCIPY_AVAILABLE:

    metrics[
        "mean_spearman"
    ] = mean_spearman

    metrics[
        "mean_kendall_tau"
    ] = mean_kendall


metrics_df = pd.DataFrame(
    [metrics]
)


METRICS_FILE = (
    RESULTS_DIR
    / "current_model_metrics.csv"
)


metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print(
    "\nResults saved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nMetrics saved to:"
)

print(
    METRICS_FILE
)


print(
    "\n" + "=" * 70
)

print(
    "CURRENT MODEL EVALUATION COMPLETE"
)

print("=" * 70)