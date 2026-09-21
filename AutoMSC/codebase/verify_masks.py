import glob
import os
import sys

import numpy as np
import SimpleITK as sitk

pred_dir, img_dir = sys.argv[1], sys.argv[2]
preds = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))

bad = 0
labels = set()
for p in preds:
    cid = os.path.basename(p)[:-7]
    src = os.path.join(img_dir, cid + "_0000.nii.gz")
    if not os.path.exists(src):
        print("MISSING SOURCE:", cid)
        bad += 1
        continue
    a, b = sitk.ReadImage(p), sitk.ReadImage(src)
    close = lambda x, y: all(abs(i - j) < 1e-4 for i, j in zip(x, y))
    if not (a.GetSize() == b.GetSize()
            and close(a.GetSpacing(), b.GetSpacing())
            and close(a.GetOrigin(), b.GetOrigin())
            and close(a.GetDirection(), b.GetDirection())):
        print("GEOMETRY MISMATCH:", cid, a.GetSize(), "vs", b.GetSize())
        bad += 1
    labels.update(np.unique(sitk.GetArrayFromImage(a)).tolist())

print(f"checked {len(preds)} masks, {bad} problems")
print("label values present:", sorted(labels))