from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "german.data"

COLUMNS = [
    "checking_account",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account",
    "employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_since",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "liable_people",
    "telephone",
    "foreign_worker",
    "class_label",
]

CATEGORY_LABELS = {
    "job": {
        "A171": "Unemployed / unskilled (non-resident)",
        "A172": "Unskilled (resident)",
        "A173": "Skilled employee / official",
        "A174": "Management / self-employed / highly qualified",
    },
    "housing": {
        "A151": "Rent",
        "A152": "Own",
        "A153": "Free housing",
    },
    "credit_history": {
        "A30": "No previous credits / all paid back",
        "A31": "Previous credits at this bank paid back",
        "A32": "Existing credits paid back duly",
        "A33": "Previous payment delays",
        "A34": "Critical account / other bank credits",
    },
    "savings_account": {
        "A61": "< 100 DM",
        "A62": "100–<500 DM",
        "A63": "500–<1000 DM",
        "A64": ">= 1000 DM",
        "A65": "Unknown / no savings",
    },
    "purpose": {
        "A40": "New car",
        "A41": "Used car",
        "A42": "Furniture / equipment",
        "A43": "Radio / television",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A47": "Vacation",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Other",
    },
}

SEX_MAP = {
    "A91": "male",
    "A92": "female",
    "A93": "male",
    "A94": "male",
    "A95": "female",
}

def load_german_credit(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    if df.shape != (1000, 21):
        raise ValueError(
            f"Unexpected dataset shape: {df.shape}. Expected (1000, 21)."
        )
    
    # Create human-readable versions of categorical features.
    for column, mapping in CATEGORY_LABELS.items():
        label_column = f"{column}_label"
        df[label_column] = df[column].map(mapping)

        if df[label_column].isna().any():
            unknown = df.loc[df[label_column].isna(), column].unique()
            raise ValueError(
                f"Unknown categories in {column}: {unknown}"
            )

    # Extract sex from the combined personal status and sex attribute.
    df["sex"] = df["personal_status_sex"].map(SEX_MAP)

    if df["sex"].isna().any():
        raise ValueError("Unknown values found in personal_status_sex.")

    # Create age groups: below 30 and 30 or older.
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, float("inf")],
        labels=["<30", ">=30"],
        right=False,
    )

    # Convert the original target into a binary classification target.
    # 0 = Good credit, 1 = Bad credit.
    df["target_bad"] = df["class_label"].map({
        1: 0,
        2: 1,
    })

    if df["target_bad"].isna().any():
        raise ValueError("Unexpected target class values.")

    return df


if __name__ == "__main__":
    df = load_german_credit()

    print("Dataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())

    print("\nTarget distribution:")
    print(df["target_bad"].value_counts().rename({
        0: "Good credit",
        1: "Bad credit",
    }))

    print("\nSex distribution:")
    print(df["sex"].value_counts())

    print("\nAge group distribution:")
    print(df["age_group"].value_counts())