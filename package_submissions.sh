#!/bin/bash
# Build the Small Model Track zips for every solution directory written by analysis/build_submission_artifacts.py, then move them
# into submitted_solutions/ under their short name (the "solution-spurious-" prefix dropped, e.g. confidence-cascade-small.zip),
# matching the zip_name in each zip's SUBMISSION_INFO.json. Usage: bash package_submissions.sh
set -euo pipefail
GS=/home/sabyasachi19/PycharmProjects/aimo/getting-started
OUT=/home/sabyasachi19/PycharmProjects/AimoAGAIN/submitted_solutions
mkdir -p "$OUT"
(cd "$GS" && python3 scripts/build.py solutions --small > /dev/null)
for f in "$GS"/dist/solution-spurious-*-small.zip; do
  short=$(basename "$f"); short=${short#solution-spurious-}
  mv -f "$f" "$OUT/$short"
  python3 - "$OUT/$short" <<'PY'
import json, sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); assert z.testzip() is None, "corrupt zip"
info = json.loads(z.read("SUBMISSION_INFO.json")); name = sys.argv[1].rsplit("/", 1)[1]
assert info["zip_name"] == name, f"SUBMISSION_INFO says {info['zip_name']}, file is {name}"
print(f"{name:34s} cut {info['cut']} | cap {info['cap']} | features {info['recipe_features']} | build {info['build_id']}")
PY
done
