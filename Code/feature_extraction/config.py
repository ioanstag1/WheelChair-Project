import os

# =========================================================
# 1. GLOBAL SETTINGS & DYNAMIC PATHS
# =========================================================

# Automatically locate the root folder of this project script
BASE_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the absolute root directory where raw data is stored
# Note: Ensure this path is accessible in your environment (e.g., Google Drive)
DATA_ROOT = "/content/drive/MyDrive/Wheelchair_Project/"

# Sampling frequency of the video data
FPS = 50.0

# =========================================================
# 2. PATH RESOLVER FUNCTION
# =========================================================

def get_paths(dataset: str, task: str):
    """
    Resolves and returns all necessary file paths (JSON input, Video input, Output)
    for a specific Dataset and Task.
    
    Args:
        dataset (str): 'Poland' or 'Brasil'
        task (str): 'ADL1', 'ADL2', 'ADL3'
        
    Returns:
        dict: A dictionary containing all relevant directory paths.
    """
    dataset = dataset.lower()
    task = task.upper()

    # ----------------------------------------
    # A. Define Input Directories
    # ----------------------------------------
    if dataset == "poland":
        json_root = os.path.join(DATA_ROOT, "Poland_PoseEstimation", "VitSpineXLSmoothed")
        video_root = os.path.join(DATA_ROOT, "PP_selected_videos")
        # Main output directory for Poland
        output_root = os.path.join(DATA_ROOT, "PPPB_Features", "Poland")
        
    elif dataset == "brasil":
        json_root = os.path.join(DATA_ROOT, "Brasil_PoseEstimation", "VitSpineSmoothed")
        video_root = os.path.join(DATA_ROOT, "PB_selected_videos")
        # Main output directory for Brasil
        output_root = os.path.join(DATA_ROOT, "PPPB_Features", "Brasil")
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Please use 'Poland' or 'Brasil'.")

    # ----------------------------------------
    # B. Define Output Directories
    # ----------------------------------------
    # Specific folder for the current Task
    task_out_dir = os.path.join(output_root, task)
    
    # Subfolders for intermediate time-series CSVs
    sagittal_csv = os.path.join(task_out_dir, "sagittal", "csv")
    frontal_csv = os.path.join(task_out_dir, "frontal", "csv")
    
    # Path for the Timing (Metadata) CSV file
    timing_csv = os.path.join(output_root, "Timing_CSVs", f"{task.lower()}_timing.csv")
    
    # Directory for the final computed features
    features_dir = os.path.join(output_root, "Features")

    # Ensure all output directories exist; create them if necessary
    for path in [sagittal_csv, frontal_csv, features_dir, os.path.dirname(timing_csv)]:
        os.makedirs(path, exist_ok=True)

    # ----------------------------------------
    # C. Return Paths Dictionary
    # ----------------------------------------
    return {
        "json_root": json_root,
        "video_root": video_root,
        "output_root": output_root,
        "task_out_dir": task_out_dir,
        "sagittal_csv": sagittal_csv,
        "frontal_csv": frontal_csv,
        "timing_csv": timing_csv,
        "features_dir": features_dir
    }

# =========================================================
# 3. HELPER: FILE COLLECTOR
# =========================================================

def collect_files(root_path, valid_suffixes, extension=".json"):
    """
    Recursively searches a directory for files matching specific suffixes.
    
    Args:
        root_path (str): The directory to search.
        valid_suffixes (list): List of substrings to match (e.g., ['ADL1S', 'ADL1F']).
        extension (str): File extension to filter by (default: .json).
        
    Returns:
        list: A list of absolute file paths found.
    """
    found_files = []
    
    if not os.path.exists(root_path):
        print(f"   [WARNING] Directory not found -> {root_path}")
        return found_files

    for root, _, files in os.walk(root_path):
        for f in files:
            # Check if file has correct extension and contains a valid suffix
            if f.endswith(extension) and any(tag in f for tag in valid_suffixes):
                found_files.append(os.path.join(root, f))
    
    return found_files