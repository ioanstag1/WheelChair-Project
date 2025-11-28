import pandas as pd
import numpy as np
import os
import utils
from scipy.stats import linregress
from scipy.signal import find_peaks
from scipy.integrate import simpson

# =========================================================
# ADL1: FULL FEATURE EXTRACTION
# =========================================================
def extract_features_adl1(base_dir, summary_csv, output_file):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(summary_csv):
        print(f"   ❌ Summary CSV missing: {summary_csv}")
        return

    # Load summary file (Metadata)
    summary = pd.read_csv(summary_csv)
    
    # Paths to subfolders
    sagittal_dir = os.path.join(base_dir, "sagittal", "csv")
    frontal_dir = os.path.join(base_dir, "frontal", "csv")

    # --- Initialize New Feature Columns ---
    for i in range(1, 6):
        summary[f"TrunkPeak{i}_deg"] = np.nan
        summary[f"HeadPitch_at_Peak{i}_deg"] = np.nan
        summary[f"TrunkROM_Peak{i}_deg"] = np.nan
        summary[f"HeadComp_Peak{i}_deg"] = np.nan
        summary[f"ShoulderAsym_Peak{i}_deg"] = np.nan
        summary[f"HandAsym_Peak{i}_norm"] = np.nan
        summary[f"KneeAngle_Peak{i}_deg"] = np.nan 
        summary[f"KneeROM_Peak{i}_deg"] = np.nan   
        summary[f"ElbowAngle_Peak{i}_deg"] = np.nan
        summary[f"ElbowROM_Peak{i}_deg"] = np.nan   

    summary["LeftRightOnsetLag_s_calc"] = np.nan
    summary["TrunkROM_Repetition_SD"] = np.nan      # Variability
    summary["TrunkROM_Dropoff_Fatigue"] = np.nan    # Fatigue
    summary["KneeROM_Repetition_SD"] = np.nan       
    summary["ElbowROM_Repetition_Mean"] = np.nan    
    summary["HeadComp_Mean"] = np.nan               

    print(f"   Processing ADL1 features for {len(summary)} participants...")

    # --- Process Each Participant ---
    for idx, row in summary.iterrows():
        pid = row["patient"]
        
        # Construct file paths
        sagittal_path = os.path.join(sagittal_dir, f"{pid}ADL1S_sagittal.csv")
        frontal_path = os.path.join(frontal_dir, f"{pid}ADL1F_frontal.csv")

        # Skip if files are missing
        if not os.path.exists(sagittal_path) or not os.path.exists(frontal_path):
            print(f"   ⚠️ Missing data for {pid}")
            continue

        # Load Sagittal Data
        df_sag = pd.read_csv(sagittal_path)
        t = df_sag["Time (s)"].values
        trunk = df_sag["TrunkPitch(deg)"].values
        
        # Check if columns exist, otherwise fill with zeros
        head = df_sag["HeadPitch(deg)"].values if "HeadPitch(deg)" in df_sag else np.zeros_like(t)
        knee = df_sag["KneeAngle(deg)"].values if "KneeAngle(deg)" in df_sag else np.zeros_like(t)
        elbow = df_sag["ElbowAngle(deg)"].values if "ElbowAngle(deg)" in df_sag else np.zeros_like(t)

        # Calculate Baseline (first 0.5s)
        baseline_mask = t < 0.5
        if np.sum(baseline_mask) > 0:
            base_trunk = np.mean(trunk[baseline_mask])
            base_knee = np.mean(knee[baseline_mask])
            base_elbow = np.mean(elbow[baseline_mask])
        else:
            base_trunk = trunk[0]
            base_knee = knee[0]
            base_elbow = elbow[0]

        # Load Frontal Data
        df_front = pd.read_csv(frontal_path)
        tf = df_front["Time (s)"].values
        shoulder_asym = df_front["ShoulderAsym(deg)"].values
        hand_asym = df_front["HandAsym(norm)"].values

        # Lists for aggregation
        all_trunk_roms = []
        all_knee_roms = []
        all_elbow_roms = []
        all_head_comps = []

        # --- Compute metrics for each of the 5 peaks ---
        for i in range(1, 6):
            peak_col = f"trunk peak {i}"
            if peak_col not in row: continue

            peak_time = row.get(peak_col, np.nan)
            if pd.isna(peak_time): continue

            # Interpolate values at peak time
            theta_trunk = np.interp(peak_time, t, trunk)
            theta_head = np.interp(peak_time, t, head)
            theta_knee = np.interp(peak_time, t, knee)
            theta_elbow = np.interp(peak_time, t, elbow)
            
            theta_shoulder = np.interp(peak_time, tf, shoulder_asym)
            hand_val = np.interp(peak_time, tf, hand_asym)

            # Assign to DataFrame
            summary.loc[idx, f"TrunkPeak{i}_deg"] = theta_trunk
            summary.loc[idx, f"HeadPitch_at_Peak{i}_deg"] = theta_head
            
            # Ranges of Motion (ROM)
            trunk_rom = theta_trunk - base_trunk
            summary.loc[idx, f"TrunkROM_Peak{i}_deg"] = trunk_rom
            
            head_comp = theta_head - theta_trunk
            summary.loc[idx, f"HeadComp_Peak{i}_deg"] = head_comp 
            
            summary.loc[idx, f"ShoulderAsym_Peak{i}_deg"] = theta_shoulder
            summary.loc[idx, f"HandAsym_Peak{i}_norm"] = hand_val

            summary.loc[idx, f"KneeAngle_Peak{i}_deg"] = theta_knee
            summary.loc[idx, f"KneeROM_Peak{i}_deg"] = theta_knee - base_knee
            summary.loc[idx, f"ElbowAngle_Peak{i}_deg"] = theta_elbow
            summary.loc[idx, f"ElbowROM_Peak{i}_deg"] = theta_elbow - base_elbow
            
            # Collect for aggregates
            all_trunk_roms.append(trunk_rom)
            all_knee_roms.append(theta_knee - base_knee)
            all_elbow_roms.append(theta_elbow - base_elbow)
            all_head_comps.append(head_comp)

        # --- Compute Aggregate Metrics ---
        if all_trunk_roms:
            summary.loc[idx, "TrunkROM_Repetition_SD"] = np.nanstd(all_trunk_roms)
            summary.loc[idx, "KneeROM_Repetition_SD"] = np.nanstd(all_knee_roms)
            summary.loc[idx, "ElbowROM_Repetition_Mean"] = np.nanmean(all_elbow_roms)
            summary.loc[idx, "HeadComp_Mean"] = np.nanmean(all_head_comps)

            # Fatigue: Difference between 1st and Last Rep
            valid_roms = [r for r in all_trunk_roms if not np.isnan(r)]
            if len(valid_roms) >= 2:
                summary.loc[idx, "TrunkROM_Dropoff_Fatigue"] = valid_roms[0] - valid_roms[-1]
            else:
                summary.loc[idx, "TrunkROM_Dropoff_Fatigue"] = 0

        # Preserve Onset Lag calculation
        if "left right onset lag(s)" in row:
            summary.loc[idx, "LeftRightOnsetLag_s_calc"] = row["left right onset lag(s)"]

    # === CLEANUP: Remove Original Timing Columns ===
    drop_features = [
        'trunk peak 1', 'trunk peak 2', 'trunk peak 3', 'trunk peak 4', 'trunk peak 5',
        'elbow deg pick start(s)', 'Trunk onset(s)', 'left onset', 'wrist onset',
        'left right onset lag(s)', 'Handrim contact onset lag (s)'
    ]
    
    # Drop only columns that actually exist in the dataframe
    existing_to_drop = [c for c in drop_features if c in summary.columns]
    summary.drop(columns=existing_to_drop, inplace=True)

    # Save Final CSV
    summary.to_csv(output_file, index=False)
    print(f"ADL1 Features (Cleaned) saved to: {output_file}")


# =========================================================
# ADL2: FULL FEATURE EXTRACTION
# =========================================================
def extract_features_adl2(base_dir, summary_csv, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not os.path.exists(summary_csv): return

    times_df = pd.read_csv(summary_csv)
    results = []
    
    sag_dir = os.path.join(base_dir, "sagittal", "csv")
    front_dir = os.path.join(base_dir, "frontal", "csv")
    
    print(f"   Processing ADL2 features for {len(times_df)} participants...")

    for _, row in times_df.iterrows():
        pid = row['patient']
        t1, t2 = row['tstart(sec)'], row['tend(sec)']
        
        sag_path = os.path.join(sag_dir, f"{pid}ADL2S_sagittal.csv")
        front_path = os.path.join(front_dir, f"{pid}ADL2F_frontal.csv")
        
        if not os.path.exists(sag_path): continue
        if not os.path.exists(front_path): continue
        
        # --- SAGITTAL ---
        df_s = pd.read_csv(sag_path)
        mask_s = (df_s["Time (s)"] >= t1) & (df_s["Time (s)"] <= t2)
        if mask_s.sum() < 5: continue
        
        time_s = df_s.loc[mask_s, "Time (s)"].values
        p_phase = utils.smooth(df_s.loc[mask_s, "TrunkPitch(deg)"].values)
        knee_seg = utils.smooth(df_s.loc[mask_s, "KneeAngle(deg)"].values)
        elbow_seg = utils.smooth(df_s.loc[mask_s, "ElbowAngle(deg)"].values)
        
        # Metrics
        mean_pitch = np.mean(p_phase)
        rms_pitch = np.sqrt(np.mean((p_phase - mean_pitch)**2))
        slope, _, _, _, _ = linregress(time_s, p_phase)
        excursion = np.max(p_phase) - np.min(p_phase)
        time_in_tol = 100 * np.mean(np.abs(p_phase - mean_pitch) < 2.0)
        
        dp = np.gradient(p_phase, time_s)
        smoothness = np.mean(np.abs(dp))
        
        knee_rms = np.sqrt(np.mean((knee_seg - np.mean(knee_seg))**2))
        elbow_rms = np.sqrt(np.mean((elbow_seg - np.mean(elbow_seg))**2))

        # --- FRONTAL ---
        df_f = pd.read_csv(front_path)
        mask_f = (df_f["Time (s)"] >= t1) & (df_f["Time (s)"] <= t2)
        trunk_roll = utils.smooth(df_f.loc[mask_f, "TrunkRoll(deg)"].values)
        trunk_rms_roll = np.sqrt(np.mean((trunk_roll - np.mean(trunk_roll))**2))
        
        head_stab_ratio = np.nan
        if "HeadRoll(deg)" in df_f.columns:
            head_roll = utils.smooth(df_f.loc[mask_f, "HeadRoll(deg)"].values)
            head_rms = np.sqrt(np.mean((head_roll - np.mean(head_roll))**2))
            if trunk_rms_roll > 1e-4:
                head_stab_ratio = head_rms / trunk_rms_roll

        results.append({
            "patient": pid, "label": row.get('label', 0),
            "TrunkPitch_mean": round(mean_pitch, 3),
            "TrunkPitch_RMS": round(rms_pitch, 3),
            "PitchDrift_deg_s": round(slope, 3),
            "PitchExcursion_deg": round(excursion, 3),
            "TimeInTolerance_pct": round(time_in_tol, 2),
            "MeanAbsDerivative": round(smoothness, 3),
            "KneeAngle_RMS": round(knee_rms, 3),
            "ElbowAngle_RMS": round(elbow_rms, 3),
            "TrunkRoll_RMS": round(trunk_rms_roll, 3),
            "HeadStabilizationRatio": round(head_stab_ratio, 3)
        })
        
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"ADL2 Features saved to: {output_file}")


# =========================================================
# ADL3: FULL FEATURE EXTRACTION
# =========================================================
def extract_features_adl3(base_dir, summary_csv, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not os.path.exists(summary_csv): return

    timing = pd.read_csv(summary_csv)
    features = []
    
    sag_dir = os.path.join(base_dir, "sagittal", "csv")
    front_dir = os.path.join(base_dir, "frontal", "csv")
    
    print(f"   Processing ADL3 features for {len(timing)} participants...")

    for _, row in timing.iterrows():
        pid = row['patient']
        ton, te = row['tonset'], row['tend']
        tp = row['tpeak']
        
        sag_path = os.path.join(sag_dir, f"{pid}ADL3S_sagittal.csv")
        fro_path = os.path.join(front_dir, f"{pid}ADL3F_frontal.csv")
        
        if not os.path.exists(sag_path): continue
        if not os.path.exists(fro_path): continue
        
        sag = pd.read_csv(sag_path)
        fro = pd.read_csv(fro_path)
        
        ts_sag = sag["Time(s)"].values
        if len(ts_sag) < 2: continue
        dt = np.median(np.diff(ts_sag))
        fs = 1/dt
        
        # Smooth Signals
        trunk = utils.smooth(sag["TrunkPitch(deg)"].values)
        pelvis = utils.smooth(sag["PelvisPitch(deg)"].values)
        head = utils.smooth(sag["HeadPitch(deg)"].values)
        knee = utils.smooth(sag["KneeAngle(deg)"].values) if "KneeAngle(deg)" in sag else np.zeros_like(trunk)
        elbow = utils.smooth(sag["ElbowAngle(deg)"].values) if "ElbowAngle(deg)" in sag else np.zeros_like(trunk)
        
        roll = utils.smooth(fro["TrunkRoll(deg)"].values)
        shoulder = utils.smooth(fro["ShoulderAsym(deg)"].values)
        hand = utils.smooth(fro["HandAsym(norm)"].values)
        
        # Mask
        mask = (ts_sag >= ton) & (ts_sag <= te)
        if mask.sum() < 5: continue
        
        trunk_seg = trunk[mask]
        peak_trunk = np.max(trunk_seg)
        baseline = np.mean(trunk[:5]) if len(trunk) >= 5 else trunk[0]
        rom = peak_trunk - baseline
        
        # Jerk & SPARC
        vel = np.gradient(trunk_seg, dt)
        acc = np.gradient(vel, dt)
        jerk_cost = simpson(acc**2, dx=dt)
        sparc = utils.compute_sparc(trunk_seg, fs)
        
        # Peak Index & Coordination
        peak_idx = np.argmax(trunk_seg)
        pelvis_seg = pelvis[mask]
        head_seg = head[mask]
        
        pelvis_tilt = pelvis_seg[peak_idx] if peak_idx < len(pelvis_seg) else np.nan
        head_comp = head_seg[peak_idx] - trunk_seg[peak_idx] if peak_idx < len(head_seg) else np.nan
        corr = np.corrcoef(trunk_seg, pelvis_seg)[0,1] if len(pelvis_seg) == len(trunk_seg) else np.nan
        
        # Knee / Elbow
        knee_seg = knee[mask]
        elbow_seg = elbow[mask]
        knee_jerk = np.max(knee_seg) - np.min(knee_seg)
        knee_max = np.max(knee_seg)
        
        mid = len(elbow_seg) // 2
        elbow_return_rom = np.max(elbow_seg[mid:]) - np.min(elbow_seg[mid:]) if len(elbow_seg) > 0 else np.nan
        
        # Frontal Metrics
        mask_f = (fro["Time(s)"].values >= ton) & (fro["Time(s)"].values <= te)
        roll_seg = roll[mask_f]
        shoulder_seg = shoulder[mask_f]
        hand_seg = hand[mask_f]
        
        trunk_roll_rms = np.sqrt(np.mean(roll_seg**2)) if len(roll_seg) > 0 else np.nan
        shoulder_val = shoulder_seg[peak_idx] if peak_idx < len(shoulder_seg) else np.mean(shoulder_seg)
        hand_val = hand_seg[peak_idx] if peak_idx < len(hand_seg) else np.mean(hand_seg)

        features.append({
            "patient": pid,
            "label": row.get('label', 0),
            "PeakTrunkFlexion_deg": round(peak_trunk, 2),
            "TrunkROM_deg": round(rom, 2),
            "TimeToPeak_s": round(tp - ton, 2),
            "SPARC": round(sparc, 4),
            "JerkCost_norm": round(jerk_cost, 2),
            "PelvisTilt_deg": round(pelvis_tilt, 2),
            "TrunkPelvisCorr": round(corr, 3),
            "HeadTrunkDiff_deg": round(head_comp, 2),
            "Knee_Jerk_deg": round(knee_jerk, 2),
            "Knee_MaxExtension_deg": round(knee_max, 2),
            "ElbowROM_Return_deg": round(elbow_return_rom, 2),
            "TrunkRoll_RMS": round(trunk_roll_rms, 2),
            "ShoulderAsym_deg": round(shoulder_val, 2),
            "HandAsym_norm": round(hand_val, 3)
        })

    pd.DataFrame(features).to_csv(output_file, index=False)
    print(f"ADL3 Features saved to: {output_file}")