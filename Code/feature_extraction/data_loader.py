import numpy as np
import os
import json
import csv
import utils
from config import FPS

def process_adl_json(json_files, output_folders, task_type):
    """
    General function to process JSONs based on task type (ADL1, ADL2, ADL3).
    """
    for json_file in json_files:
        filename = os.path.basename(json_file).split(".")[0]
        clean_name = filename.replace("_poses", "")

        # --- FRONTAL ---
        if "F" in filename:
            csv_path = os.path.join(output_folders['frontal_csv'], f"{clean_name}_frontal.csv")
            with open(json_file, 'r') as fh: data = json.load(fh)
            
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                
                if task_type == "ADL1":
                    writer.writerow(["Time (s)", "ShoulderAsym(deg)", "HandAsym(norm)", "LeftWristNorm", "RightWristNorm"])
                    for idx, fd in enumerate(data.get('frames', [])):
                        kp = np.array(fd['keypoints'])
                        ls, rs = kp[5][:2], kp[6][:2]
                        lw, rw = kp[9][:2], kp[10][:2]
                        writer.writerow([idx/FPS, utils.calculate_shoulder_asymmetry(ls, rs), utils.calculate_hand_height_asymmetry(lw, rw, ls, rs), *utils.calculate_wrist_normalized_positions(ls, rs, lw, rw)])
                
                elif task_type in ["ADL2", "ADL3"]: # Similar headers for ADL2/3 frontal
                    cols = ['Time (s)', 'TrunkRoll(deg)', 'HeadRoll(deg)'] if task_type == "ADL2" else ['Time(s)', 'TrunkRoll(deg)', 'ShoulderAsym(deg)', 'HandAsym(norm)']
                    writer.writerow(cols)
                    for idx, fd in enumerate(data.get('frames', [])):
                        kp = np.array(fd['keypoints'])
                        kp_spine = np.array(fd['spine_keypoints'])
                        row = [idx/FPS, utils.calculate_trunk_roll(kp_spine[2][:2], kp_spine[-1][:2])]
                        if task_type == "ADL2":
                            row.append(utils.calculate_head_roll(kp[3][:2], kp[4][:2]))
                        else:
                            row.extend([utils.calculate_shoulder_asymmetry(kp[5][:2], kp[6][:2]), utils.calculate_hand_height_asymmetry(kp[9][:2], kp[10][:2], kp[5][:2], kp[6][:2])])
                        writer.writerow(row)

        # --- SAGITTAL ---
        elif "S" in filename:
            csv_path = os.path.join(output_folders['sagittal_csv'], f"{clean_name}_sagittal.csv")
            with open(json_file, 'r') as fh: data = json.load(fh)
            
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                
                if task_type == "ADL1":
                    writer.writerow(["Time (s)", "TrunkPitch(deg)", "KneeAngle(deg)", "ElbowAngle(deg)", "HeadPitch(deg)"])
                elif task_type == "ADL2":
                    writer.writerow(['Time (s)', 'TrunkPitch(deg)', 'KneeAngle(deg)', 'ElbowAngle(deg)'])
                elif task_type == "ADL3":
                    writer.writerow(['Time(s)', 'TrunkPitch(deg)', 'PelvisPitch(deg)', 'HeadPitch(deg)', 'KneeAngle(deg)', 'ElbowAngle(deg)'])

                for idx, fd in enumerate(data.get('frames', [])):
                    side = fd.get("side", "").strip()
                    kp_spine = np.array(fd['spine_keypoints'])
                    kp = np.array(fd['keypoints'])
                    hip, neck = kp_spine[-1][:2], kp_spine[2][:2]
                    
                    if side == "left": shoulder, elbow, wrist, knee = kp[5][:2], kp[7][:2], kp[9][:2], kp[14][:2]
                    else: shoulder, elbow, wrist, knee = kp[6][:2], kp[8][:2], kp[10][:2], kp[13][:2]

                    # Common calculations
                    tp = utils.calculate_trunk_pitch_angle(hip, neck, True, side=="right")
                    ka = utils.process_knee_angles(side, hip, knee)
                    ea = utils.calculate_elbow_angle(shoulder, elbow, wrist)
                    
                    if task_type == "ADL1":
                        hp = utils.calculate_trunk_pitch_angle(neck, kp_spine[0][:2], True, side=="right")
                        writer.writerow([idx/FPS, tp, ka, ea, hp])
                    elif task_type == "ADL2":
                        writer.writerow([idx/FPS, tp, ka, ea])
                    elif task_type == "ADL3":
                        hp = utils.calculate_head_pitch(neck, kp_spine[0][:2], right=(side=="right"))
                        pp = utils.calculate_pelvis_pitch(kp_spine[5][:2], hip, right=(side=="right"))
                        writer.writerow([idx/FPS, tp, pp, hp, ka, ea])