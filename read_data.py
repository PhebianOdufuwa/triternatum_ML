import pandas as pd
from pathlib import Path
from Bio import SeqIO

def build_sequence_dataframe(
    fasta_path,
    mapping_csv_path,
    img_prefix,
    label_map: dict | None = None,
) -> pd.DataFrame:
    """
    Build a sequence DataFrame from a FASTA file, correct names using a mapping CSV,
    and add meta columns (id, name, img, label).
    """

    # Convert to Path (if user passed a string)
    fasta_path = Path(fasta_path)
    mapping_csv_path = Path(mapping_csv_path)
    img_prefix = Path(img_prefix)

    # 1. Load FASTA directly into DataFrame
    df = pd.DataFrame(
        [{"idx": r.description, "seqs": str(r.seq)}
         for r in SeqIO.parse(fasta_path, "fasta")]
    )

    # 2. Load mapping CSV (new_names -> old_names)
    teest = pd.read_csv(mapping_csv_path, names=["new_names", "old_names"])

    # 3. Dictionary: new → old and apply corrected names into idx_old
    name_map = dict(zip(teest["new_names"], teest["old_names"]))
    df["idx_old"] = df["idx"].map(name_map).fillna(df["idx"])

    # 4. Split idx into id and name
    df[["id", "name"]] = df["idx"].str.split(",", n=1, expand=True)
    df["id"] = df["id"].str.strip()
    df["name"] = df["name"].str.strip()

    # Drop specific classes by name (customizable list)
    drop_names = {"Lomatium brevifolium3"}
    if drop_names:
        df = df[~df["name"].isin(drop_names)].reset_index(drop=True)

    # 5. Build img path using corrected name (idx_old)
    df["img"] = df["idx_old"].astype(str).apply(lambda x: img_prefix / f"{x}.jpg")

    # 6. Apply label mapping
    if label_map is None:
        label_map = {
            "Clade 4": 0,
            "Clade 3": 1,
            "Clade 2": 2,
            "Clade 5": 3,
            "Clade 6": 4,
            "Clade 1": 5,
            "Clade 8": 6,
            "Clade 7": 7,
            "non complex: 8,
        }

    # Unmapped classes are grouped under label 10 ("others").
    df["label"] = df["name"].map(label_map).fillna(9).astype(int)
    df['seqs'] = df['seqs'].str.replace('[?-]', 'Z', regex=True)

    return df
