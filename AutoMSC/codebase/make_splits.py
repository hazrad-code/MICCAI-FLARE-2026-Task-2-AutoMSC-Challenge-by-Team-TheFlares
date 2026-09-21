import json
import os
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold

raw_dir = sys.argv[1]
out_dir = sys.argv[2]

df = pd.read_csv(os.path.join(raw_dir, "cls_data.csv"))

id_col = "identifier" if "identifier" in df.columns else df.columns[0]
df = df.rename(columns={id_col: "identifier"})

ids = df["identifier"].astype(str).to_numpy()
label_col = "label" if "label" in df.columns else df.columns[1]
y = df[label_col].to_numpy()

img_dir = os.path.join(raw_dir, "imagesTr")
have = {f.split(".nii")[0].rsplit("_", 1)[0] for f in os.listdir(img_dir)}
missing = sorted(set(ids) - have)
print("ids missing from imagesTr:", len(missing), missing[:5])

min_count = int(pd.Series(y).value_counts().min())
n_splits = max(2, min(5, min_count))

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
splits = []
for tr, va in skf.split(ids, y):
    splits.append({"train": ids[tr].tolist(), "val": ids[va].tolist()})
    print("fold", len(splits) - 1, "train", len(tr), "val", len(va),
          "val labels", pd.Series(y[va]).value_counts().to_dict())

os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "splits_final.json"), "w") as f:
    json.dump(splits, f, indent=2)
df[["identifier", "label"]].to_csv(os.path.join(out_dir, "cls_data.csv"), index=False)
print("wrote", n_splits, "folds to", out_dir)