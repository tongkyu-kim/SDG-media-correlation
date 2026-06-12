"""Print exact column names from one xlsx file, written to a temp txt so PowerShell doesn't mangle Korean."""
import pandas as pd, sys, pathlib

path = sys.argv[1]
df = pd.read_excel(path, nrows=1)
out = pathlib.Path("col_names.txt")
out.write_text("\n".join(f"{i}: {c}" for i, c in enumerate(df.columns)), encoding="utf-8")
print(f"Wrote {len(df.columns)} column names to {out}")
