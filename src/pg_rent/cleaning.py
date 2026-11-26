"""Data cleaning pipeline for MP048 (PG Room Rent - Indore)."""

from pathlib import Path
import subprocess
import zipfile
import numpy as np
import pandas as pd


FURNISHING_MAP = {
    "unfurnished": "unfurnished",
    "unf.": "unfurnished",
    "unf": "unfurnished",
    "semi": "semi",
    "sem.": "semi",
    "sem": "semi",
    "full": "full",
    "ful.": "full",
    "ful": "full",
}


def ensure_raw_csv(raw_path: Path) -> Path:
    """Ensure raw CSV exists locally; pull via DVC or extract from zip if only .dvc exists."""
    if raw_path.exists():
        return raw_path

    dvc_pointer = raw_path.with_suffix(raw_path.suffix + ".dvc")
    if dvc_pointer.exists():
        # 1. Try pulling from configured DVC remote/cache
        subprocess.run(["dvc", "pull", str(dvc_pointer)], check=False)
        if raw_path.exists():
            return raw_path

        # 2. Fallback: look for the original G034_MP048 zip in project root or Downloads
        search_dirs = [Path.cwd(), Path.home() / "Downloads", Path.home() / "Desktop"]
        for folder in search_dirs:
            if not folder.exists():
                continue
            for zip_file in folder.glob("*MP048*.zip"):
                with zipfile.ZipFile(zip_file, "r") as zf:
                    matches = [m for m in zf.namelist() if m.endswith(f"/{raw_path.name}")]
                    if matches:
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        raw_path.write_bytes(zf.read(matches[0]))
                        return raw_path

    raise FileNotFoundError(
        f"'{raw_path}' not found (only '{dvc_pointer}' is present). "
        f"Run 'dvc pull' or copy '{raw_path.name}' from your G034_MP048 zip into '{raw_path.parent}'."
    )


def normalize_furnishing(series: pd.Series) -> pd.Series:
    """Standardize furnishing category spelling, casing, and whitespace."""
    cleaned = series.astype(str).str.strip().str.lower()
    return cleaned.map(FURNISHING_MAP).fillna(cleaned)


def clean_training_data(
    df: pd.DataFrame,
    drop_leaking: bool = True,
    target_col: str = "rent_rs",
) -> pd.DataFrame:
    """Clean raw training data by resolving all identified issues in G034 MP048.

    Steps:
    1. Remove exact duplicate rows (54 extra copies).
    2. Drop post-event leaking column (`rent_in_signed_agreement`).
    3. Correct 10x target typos where `rent_rs` > 50,000 (17 rows).
    4. Replace sentinel value `-1` in `wifi_mbps` (46 rows) with median Wi-Fi speed.
    5. Standardize `furnishing` spelling variants into {'unfurnished', 'semi', 'full'}.
    """
    cleaned = df.copy()

    # 1. Remove exact duplicate rows
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # 2. Drop leaking post-event column
    if drop_leaking and "rent_in_signed_agreement" in cleaned.columns:
        cleaned = cleaned.drop(columns=["rent_in_signed_agreement"])

    # 3. Fix 10x target typos (e.g., 178910 -> 17891)
    if target_col in cleaned.columns:
        typo_mask = cleaned[target_col] > 50000
        cleaned.loc[typo_mask, target_col] = (
            (cleaned.loc[typo_mask, target_col] / 10.0).round().astype(int)
        )

    # 4. Replace sentinel value (-1) in wifi_mbps with column median
    if "wifi_mbps" in cleaned.columns:
        cleaned["wifi_mbps"] = cleaned["wifi_mbps"].replace(-1, np.nan)
        median_wifi = cleaned["wifi_mbps"].median()
        cleaned["wifi_mbps"] = cleaned["wifi_mbps"].fillna(median_wifi).round().astype(int)

    # 5. Normalize spelling variants in furnishing
    if "furnishing" in cleaned.columns:
        cleaned["furnishing"] = normalize_furnishing(cleaned["furnishing"])

    return cleaned


def run_cleaning_pipeline(
    input_path: str | Path = "data/raw/train.csv",
    output_path: str | Path = "data/processed/train_clean.csv",
) -> pd.DataFrame:
    """Ensure raw train.csv is materialized from .dvc/zip, clean it, and save to data/processed/."""
    in_file = ensure_raw_csv(Path(input_path))
    out_file = Path(output_path)

    raw_df = pd.read_csv(in_file)
    cleaned_df = clean_training_data(raw_df)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(out_file, index=False)
    return cleaned_df


if __name__ == "__main__":
    df_clean = run_cleaning_pipeline()
    print(f"Saved cleaned dataset to data/processed/train_clean.csv with shape {df_clean.shape}")
