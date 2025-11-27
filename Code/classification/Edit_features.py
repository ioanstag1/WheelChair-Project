# import pandas as pd

# csv_path = 'Features_All.csv'

# df = pd.read_csv(csv_path)

# # All knee columns
# knee_cols = [
#     'KneeAngle_Peak1_deg_ADL1',
#     'KneeAngle_Peak2_deg_ADL1',
#     'KneeAngle_Peak3_deg_ADL1',
#     'KneeAngle_Peak4_deg_ADL1',
#     'KneeAngle_Peak5_deg_ADL1',
#     'KneeROM_Peak1_deg_ADL1',
#     'KneeROM_Peak2_deg_ADL1',
#     'KneeROM_Peak3_deg_ADL1',
#     'KneeROM_Peak4_deg_ADL1',
#     'KneeROM_Peak5_deg_ADL1',
#     "HeadRoll_RMS_ADL2"
    
# ]

# # Remove only knee features that exist in the file
# df = df.drop(columns=[c for c in knee_cols if c in df.columns])

# df.to_csv('Features_NoKnee.csv', index=False)

# print("Removed knee features. New dataset saved as Features_NoKnee.csv")

#---------------2------------------------

# import pandas as pd

# # === Input and Output ===
# input_path = "Combined_Features.csv"
# output_path = "Combined_Features_Cleaned.csv"

# # === Load ===
# df = pd.read_csv(input_path)

# # === FEATURE FAMILIES TO DROP ===

# cols_to_drop = []

# # 1) Shoulder asymmetry (all peaks + ADL3)
# cols_to_drop += [c for c in df.columns if "ShoulderAsym" in c]

# # 2) Hand asymmetry (all peaks + ADL3)
# cols_to_drop += [c for c in df.columns if "HandAsym" in c]

# # 3) Head pitch at peak (biomechanically redundant)
# cols_to_drop += [c for c in df.columns if "HeadPitch_at_Peak" in c]

# # 4) Pelvis tilt (very redundant, correlated with trunk)
# cols_to_drop += [c for c in df.columns if "PelvisTilt" in c]

# # 5) Handrim contact onset lag (very noisy)
# cols_to_drop += [c for c in df.columns if "onset" in c.lower() or "lag" in c.lower()]

# # ----------------------------------------------------------
# # OPTIONAL – REMOVE PER-PEAK RAW ANGLES (keep ROM & RMS only)
# # Uncomment this block if you later want more reduction:
# #
# cols_to_drop += [c for c in df.columns 
#                   if "Peak" in c and 
#                      ("TrunkPeak" in c 
#                       or "ElbowAngle_Peak" in c
#                       or "KneeAngle_Peak" in c)]

# # ----------------------------------------------------------

# # === Remove duplicates in case multiple rules hit same column ===
# cols_to_drop = sorted(list(set(cols_to_drop)))

# print("Dropping", len(cols_to_drop), "columns:")
# for c in cols_to_drop:
#     print(" -", c)

# # === Drop ===
# df_clean = df.drop(columns=cols_to_drop, errors="ignore")

# # === Save ===
# df_clean.to_csv(output_path, index=False)

# print("\n✅ Saved cleaned file to:", output_path)
# print("Remaining features:", df_clean.shape[1])


#----------------------------------3-------------------------------
import pandas as pd

# Load your file
df = pd.read_csv("Combined_Features.csv")

# ----------- FAMILIES TO REMOVE -----------
drop_patterns = [
    # A. Shoulder asymmetry
    "ShoulderAsym_",

    # B. Hand asymmetry
    "HandAsym_",

    #KneeROM
    "KneeROM_Peak",
    # C. Raw head pitch at peaks
    "HeadPitch_at_Peak",

    # D. Pelvis tilt (2D unreliable)
    "PelvisTilt",

    # E. Onset / timing lag
    "OnsetLag",
    "onset",
    "lag",

    # F. Raw angles at peaks (NOT ROM)
    # "ElbowAngle_Peak",
    "KneeAngle_Peak",
    # "TrunkPeak",

    # Optional: ADL3 raw angles
    "Handrim",   # if present
]

# Collect columns to drop
cols_to_drop = []
for pattern in drop_patterns:
    cols_to_drop += [c for c in df.columns if pattern in c]

# Remove duplicates
cols_to_drop = sorted(list(set(cols_to_drop)))

print("Dropping", len(cols_to_drop), "columns:")
for c in cols_to_drop:
    print("  -", c)

# Drop them
df_clean = df.drop(columns=cols_to_drop, errors="ignore")

# Save cleaned dataset
df_clean.to_csv("Combined_Features_CLEANED_kneerom.csv", index=False)

print("\n✅ Saved cleaned file as Combined_Features_CLEANED.csv")
print("Remaining features:", df_clean.shape[1])
