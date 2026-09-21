import argparse
import ast
import glob
import json
import os
import shutil

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", help="nnUNet_raw/<Dataset0XX_NAME>")
    ap.add_argument("pred_dir", help="dir holding {case}.nii.gz and *results*.csv")
    ap.add_argument("out_root", help="submission root; predictions/ is created inside")
    a = ap.parse_args()

    name = os.path.basename(os.path.normpath(a.raw_dir))
    meta = json.load(open(os.path.join(a.raw_dir, "dataset.json")))

    tasks = meta.get("classification_labels", {})
    if len(tasks) != 1:
        raise SystemExit(f"expected one classification task, found {list(tasks)}")
    task = next(iter(tasks))
    n_classes = len(tasks[task])

    n_test = len(glob.glob(os.path.join(a.raw_dir, "imagesTs", "*_0000.nii.gz")))

    dst = os.path.join(a.out_root, "predictions", f"{name}_prediction")
    os.makedirs(dst, exist_ok=True)

    masks = sorted(glob.glob(os.path.join(a.pred_dir, "*.nii.gz")))
    for m in masks:
        shutil.copy(m, dst)

    res = glob.glob(os.path.join(a.pred_dir, "*results*.csv"))
    if len(res) != 1:
        raise SystemExit(f"expected one results csv, found {res}")

    df = pd.read_csv(res[0])
    probs = df["probs"].apply(ast.literal_eval)

    if n_classes == 2:
        out = pd.DataFrame({
            "case_id": df["identifier"],
            "label": probs.apply(lambda v: float(v[0])),
        })
    else:
        norm = probs.apply(
            lambda v: [float(x) / sum(v[:n_classes]) for x in v[:n_classes]])
        out = pd.DataFrame(norm.tolist(),
                           columns=[f"label_{i}" for i in range(n_classes)])
        out.insert(0, "case_id", df["identifier"].values)

    out.to_csv(os.path.join(dst, "predictions.csv"), index=False)

    ok = len(masks) == n_test and len(out) == n_test
    print(f"{name}: task={task} classes={n_classes} masks={len(masks)} "
          f"rows={len(out)} expected={n_test} {'OK' if ok else '*** MISMATCH ***'}")
    if n_classes > 2:
        cols = [c for c in out.columns if c.startswith("label_")]
        s = out[cols].sum(axis=1)
        print(f"row sums: min={s.min():.6f} max={s.max():.6f}")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    main()