"""
Run once on laptop, in the folder containing train.zip and test.zip.
Produces train_clean.zip and test_clean.zip with '&' replaced by 'and' in all paths.
Upload those two files (not the originals) to the Kaggle dataset.
"""
import zipfile, sys, os

def sanitize(name: str) -> str:
    return name.replace('&', 'and')

for orig in ['train.zip', 'test.zip']:
    if not os.path.exists(orig):
        print(f"SKIP: {orig} not found")
        continue
    out = orig.replace('.zip', '_clean.zip')
    print(f"{orig} -> {out}", flush=True)
    with zipfile.ZipFile(orig, 'r') as zin, \
         zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as zout:
        infos = zin.infolist()
        for i, info in enumerate(infos, 1):
            data = zin.read(info.filename)
            info.filename = sanitize(info.filename)
            zout.writestr(info, data)
            if i % 100 == 0:
                print(f"  {i}/{len(infos)}", flush=True)
    print(f"  Done -> {out}")
