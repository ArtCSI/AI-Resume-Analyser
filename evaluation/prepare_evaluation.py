import re
import ast
from pathlib import Path

import pandas as pd
from docx import Document


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = (
    BASE_DIR
    / "dataset"
    / "vacancy-resume-matching-dataset"
)

CV_DIR = DATASET_DIR / "CV"

VACANCY_FILE = DATASET_DIR / "5_vacancies.csv"

ANNOTATION_FILE = (
    DATASET_DIR
    / "annotations-for-the-first-30-vacancies.txt"
)

OUTPUT_DIR = BASE_DIR / "dataset" / "processed"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 1. LOAD VACANCIES
# ============================================================

print("Loading vacancies...")

if not VACANCY_FILE.exists():
    raise FileNotFoundError(
        f"Vacancy file not found:\n{VACANCY_FILE}"
    )

vacancies = pd.read_csv(VACANCY_FILE)

# Remove accidental whitespace from column names
vacancies.columns = (
    vacancies.columns
    .astype(str)
    .str.strip()
)

print(f"Found {len(vacancies)} vacancies.")

print("\nActual CSV columns:")

for column in vacancies.columns:
    print(f"  {repr(column)}")


# ============================================================
# FIND VACANCY DESCRIPTION COLUMN
# ============================================================

description_candidates = [
    column
    for column in vacancies.columns
    if "description" in column.lower()
]

if not description_candidates:

    raise ValueError(
        "\nCould not find a vacancy description column.\n"
        f"Available columns: {list(vacancies.columns)}"
    )

DESCRIPTION_COLUMN = description_candidates[0]

print(
    f"\nUsing vacancy description column: "
    f"{DESCRIPTION_COLUMN}"
)


# ============================================================
# DISPLAY VACANCIES
# ============================================================

required_columns = [
    "job_title",
    "uid"
]

for column in required_columns:

    if column not in vacancies.columns:

        raise ValueError(
            f"Required column '{column}' "
            f"was not found in the vacancy CSV.\n"
            f"Available columns: "
            f"{list(vacancies.columns)}"
        )


print(
    "\nVacancies:"
)

print(
    vacancies[
        ["job_title", "uid"]
    ].to_string(index=False)
)


# ============================================================
# 2. READ HUMAN ANNOTATIONS
# ============================================================

print("\nReading human annotations...")

if not ANNOTATION_FILE.exists():
    raise FileNotFoundError(
        f"Annotation file not found:\n{ANNOTATION_FILE}"
    )

annotation_text = ANNOTATION_FILE.read_text(
    encoding="utf-8"
)


def extract_rankings(name):
    """
    Extract a ranking matrix from the annotation file.

    Each inner list contains the rankings of
    vacancies 1-5 for one CV.
    """

    pattern = rf"{name}\s*=\s*(\[\[.*?\]\])"

    match = re.search(
        pattern,
        annotation_text,
        re.DOTALL
    )

    if not match:

        raise ValueError(
            f"Could not find {name} "
            f"in:\n{ANNOTATION_FILE}"
        )

    rankings_text = match.group(1)

    try:

        rankings = ast.literal_eval(
            rankings_text
        )

    except (ValueError, SyntaxError) as error:

        raise ValueError(
            f"Could not parse {name}: {error}"
        )

    return rankings


annotator_1 = extract_rankings(
    "ANNOTATOR_1_RANKINGS"
)

annotator_2 = extract_rankings(
    "ANNOTATOR_2_RANKINGS"
)


print(
    f"Annotator 1 rankings: "
    f"{len(annotator_1)} CVs"
)

print(
    f"Annotator 2 rankings: "
    f"{len(annotator_2)} CVs"
)


# ============================================================
# 3. VALIDATE HUMAN ANNOTATIONS
# ============================================================

invalid_rankings = []

for annotator_name, rankings in [
    ("Annotator 1", annotator_1),
    ("Annotator 2", annotator_2)
]:

    if len(rankings) != 30:

        raise ValueError(
            f"{annotator_name} should contain "
            f"30 CV rankings, but contains "
            f"{len(rankings)}."
        )

    for cv_index, ranking in enumerate(
        rankings,
        start=1
    ):

        # Every CV should have exactly 5 values
        if len(ranking) != 5:

            invalid_rankings.append({
                "annotator": annotator_name,
                "cv": cv_index,
                "ranking": ranking,
                "reason": "Expected exactly 5 values"
            })

            continue

        # Every value must be between 1 and 5
        if not all(
            isinstance(rank, int)
            and rank in [1, 2, 3, 4, 5]
            for rank in ranking
        ):

            invalid_rankings.append({
                "annotator": annotator_name,
                "cv": cv_index,
                "ranking": ranking,
                "reason": "Rank outside valid range 1-5"
            })

        # Detect duplicate/missing ranks
        if sorted(ranking) != [1, 2, 3, 4, 5]:

            invalid_rankings.append({
                "annotator": annotator_name,
                "cv": cv_index,
                "ranking": ranking,
                "reason": (
                    "Duplicate or missing rank "
                    "(expected a permutation of 1-5)"
                )
            })


if invalid_rankings:

    print(
        "\nWARNING: Annotation anomalies detected."
    )

    print("-" * 60)

    for item in invalid_rankings:

        print(
            f"{item['annotator']} | "
            f"CV {item['cv']} | "
            f"{item['ranking']} | "
            f"{item['reason']}"
        )

    print("-" * 60)

    print(
        "\nIMPORTANT:"
    )

    print(
        "The original human annotations are being "
        "preserved exactly."
    )

    print(
        "No ranking values are being automatically corrected."
    )

else:

    print(
        "\nHuman annotations validated successfully."
    )


# ============================================================
# 4. READ CV DOCUMENTS
# ============================================================

def read_docx(path):
    """
    Extract text from a DOCX resume.
    """

    document = Document(path)

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n".join(paragraphs)


print("\nReading CVs...")

cv_texts = {}

for cv_number in range(1, 31):

    cv_path = CV_DIR / f"{cv_number}.docx"

    if not cv_path.exists():

        raise FileNotFoundError(
            f"Missing CV file:\n{cv_path}"
        )

    cv_texts[cv_number] = read_docx(
        cv_path
    )


print(
    f"Loaded {len(cv_texts)} CVs."
)


# ============================================================
# 5. CREATE RESUME–VACANCY EVALUATION PAIRS
# ============================================================

print(
    "\nCreating evaluation pairs..."
)

rows = []


for cv_number in range(1, 31):

    ranking_1 = annotator_1[
        cv_number - 1
    ]

    ranking_2 = annotator_2[
        cv_number - 1
    ]

    for vacancy_index in range(5):

        # ----------------------------------------------------
        # Human rankings
        # ----------------------------------------------------

        rank_1 = ranking_1[
            vacancy_index
        ]

        rank_2 = ranking_2[
            vacancy_index
        ]

        average_rank = (
            rank_1 + rank_2
        ) / 2


        # ----------------------------------------------------
        # Vacancy information
        # ----------------------------------------------------

        vacancy = vacancies.iloc[
            vacancy_index
        ]


        # ----------------------------------------------------
        # Create evaluation row
        # ----------------------------------------------------

        rows.append({

            "resume_id":
                cv_number,

            "vacancy_id":
                vacancy_index + 1,

            "vacancy_title":
                vacancy["job_title"],

            "vacancy_uid":
                vacancy["uid"],

            "resume_text":
                cv_texts[cv_number],

            "job_description":
                vacancy[DESCRIPTION_COLUMN],

            "annotator_1_rank":
                rank_1,

            "annotator_2_rank":
                rank_2,

            "average_human_rank":
                average_rank
        })


# ============================================================
# 6. CREATE DATAFRAME
# ============================================================

evaluation_df = pd.DataFrame(
    rows
)


# ============================================================
# 7. VALIDATE OUTPUT
# ============================================================

expected_rows = 30 * 5

if len(evaluation_df) != expected_rows:

    raise ValueError(
        f"Expected {expected_rows} evaluation rows, "
        f"but created {len(evaluation_df)}."
    )


if evaluation_df[
    "resume_id"
].nunique() != 30:

    raise ValueError(
        "Expected 30 unique resumes."
    )


if evaluation_df[
    "vacancy_id"
].nunique() != 5:

    raise ValueError(
        "Expected 5 unique vacancies."
    )


# ============================================================
# 8. SAVE PROCESSED DATASET
# ============================================================

output_file = (
    OUTPUT_DIR
    / "resume_vacancy_evaluation.csv"
)

evaluation_df.to_csv(
    output_file,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 9. DISPLAY SUMMARY
# ============================================================

print("\n" + "=" * 60)

print(
    "EVALUATION DATASET CREATED"
)

print("=" * 60)

print(
    f"Rows: {len(evaluation_df)}"
)

print(
    f"CVs: "
    f"{evaluation_df['resume_id'].nunique()}"
)

print(
    f"Vacancies: "
    f"{evaluation_df['vacancy_id'].nunique()}"
)


print("\nFirst 10 evaluation pairs:")

print(
    evaluation_df[
        [
            "resume_id",
            "vacancy_id",
            "vacancy_title",
            "annotator_1_rank",
            "annotator_2_rank",
            "average_human_rank"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print(
    "\nOutput file:"
)

print(
    output_file
)


print(
    "\nDone!"
)