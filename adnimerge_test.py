# verify_primary_tables.py
import pandas as pd

dxsum = pd.read_csv(r"D:\ADNI\DXSUM.csv", low_memory=False)
ptdemog = pd.read_csv(r"D:\ADNI\PTDEMOG.csv", low_memory=False)
raw = pd.read_csv(r"C:\Users\todor\PycharmProjects\PyCharm-Work\Msc-AD\ADNI\data\adni_raw_metadata.csv")

print(f"=== DXSUM ===  rows={len(dxsum)}")
print(f"Columns: {dxsum.columns.tolist()}")
print()

print(f"=== PTDEMOG ===  rows={len(ptdemog)}")
print(f"Columns: {ptdemog.columns.tolist()}")
print()



#



your_subjects = set(raw["subject_id"].astype(str))
print(f"=== Coverage check (you have {len(your_subjects)} subjects) ===")

for tname, t in [("DXSUM", dxsum), ("PTDEMOG", ptdemog)]:
    # Try common ID column names
    for idcol in ["PTID", "RID", "Subject", "SUBJECT_ID", "USUBJID"]:
        if idcol in t.columns:
            t_subjects = set(t[idcol].astype(str))
            n_overlap = len(your_subjects & t_subjects)
            print(f"  {tname}.{idcol}: {n_overlap}/{len(your_subjects)} of your subjects found")
            break

# Example row from each for subject 023_S_0388
target = "023_S_0388"
print(f"\n=== Sample data for {target} ===")
for tname, t in [("DXSUM", dxsum), ("PTDEMOG", ptdemog)]:
    if "PTID" in t.columns:
        sub = t[t["PTID"].astype(str) == target]
        print(f"\n{tname} ({len(sub)} rows):")
        if len(sub) > 0:
            print(sub.head(3).to_string())