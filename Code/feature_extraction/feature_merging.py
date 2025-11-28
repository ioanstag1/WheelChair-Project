import pandas as pd
import os
import config

def merge_adls_for_country(dataset_name):
    """
    Combines ADL1, ADL2, and ADL3 features for a specific country
    into a single CSV file with suffixes (e.g., _ADL1).
    """
    print(f"   [MERGE] Combining ADL tasks for {dataset_name}...")

    # Define the directory where the individual ADL files are stored
    # We use ADL1 to get the base features directory (it is the same for all tasks)
    paths = config.get_paths(dataset_name, "ADL1")
    features_dir = paths['features_dir']
    
    # Define expected filenames based on main.py output
    file_adl1 = os.path.join(features_dir, f"{dataset_name}_ADL1_Final_Metrics.csv")
    file_adl2 = os.path.join(features_dir, f"{dataset_name}_ADL2_Final_Metrics.csv")
    file_adl3 = os.path.join(features_dir, f"{dataset_name}_ADL3_Final_Metrics.csv")

    # Check if files exist
    if not (os.path.exists(file_adl1) and os.path.exists(file_adl2) and os.path.exists(file_adl3)):
        print(f"   [ERROR] Missing one or more ADL files for {dataset_name}. Cannot merge.")
        return None

    # Load DataFrames
    df1 = pd.read_csv(file_adl1)
    df2 = pd.read_csv(file_adl2)
    df3 = pd.read_csv(file_adl3)

    # Rename columns to add suffixes (excluding patient and label)
    # This ensures we know which task a feature belongs to (e.g., TrunkROM_ADL1 vs TrunkROM_ADL3)
    df1 = df1.rename(columns={c: f"{c}_ADL1" for c in df1.columns if c not in ['patient', 'label']})
    df2 = df2.rename(columns={c: f"{c}_ADL2" for c in df2.columns if c not in ['patient', 'label']})
    df3 = df3.rename(columns={c: f"{c}_ADL3" for c in df3.columns if c not in ['patient', 'label']})

    # Merge sequentially on patient and label
    # We use 'outer' join to keep patients even if they are missing data for one task
    merged_df = pd.merge(df1, df2, on=['patient', 'label'], how='outer')
    merged_df = pd.merge(merged_df, df3, on=['patient', 'label'], how='outer')

    # Define output path for the country-specific combined file
    # We create a new folder "Combined" in the root output directory
    combined_root = os.path.join(paths['output_root'], "Combined")
    os.makedirs(combined_root, exist_ok=True)
    
    output_path = os.path.join(combined_root, f"{dataset_name}_Combined_Features.csv")
    
    merged_df.to_csv(output_path, index=False)
    print(f"   [SUCCESS] Saved combined features to: {output_path}")
    
    return output_path


def combine_all_datasets(poland_path, brasil_path):
    """
    Merges the combined Poland file and the combined Brasil file
    into one final global dataset.
    """
    print("\n[MERGE] Creating Global Dataset (Poland + Brasil)...")

    if not poland_path or not os.path.exists(poland_path):
        print("   [ERROR] Poland combined file missing.")
        return
    if not brasil_path or not os.path.exists(brasil_path):
        print("   [ERROR] Brasil combined file missing.")
        return

    # Load data
    df_pol = pd.read_csv(poland_path)
    df_bra = pd.read_csv(brasil_path)

    # Add dataset origin column
    df_pol['dataset'] = 'Poland'
    df_bra['dataset'] = 'Brasil'

    # Align columns (ensure both dataframes have the same column order)
    all_cols = sorted(set(df_pol.columns).union(df_bra.columns))
    df_pol = df_pol.reindex(columns=all_cols)
    df_bra = df_bra.reindex(columns=all_cols)

    # Concatenate vertically
    df_final = pd.concat([df_pol, df_bra], ignore_index=True)

    # Define final output path (In the data root folder)
    final_dir = os.path.join(config.DATA_ROOT, "PPPB_Features", "Global_Combined")
    os.makedirs(final_dir, exist_ok=True)
    
    final_path = os.path.join(final_dir, "Global_Combined_Features.csv")

    # Save
    df_final.to_csv(final_path, index=False)
    
    print(f"   [SUCCESS] Global dataset saved to: {final_path}")
    print(f"   Total Samples: {len(df_final)}")
    
    if 'label' in df_final.columns:
        print("\n   Class Distribution:")
        print(df_final['label'].value_counts())