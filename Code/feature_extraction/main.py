import os
import config
import data_loader
import feature_extractor
import feature_merging 

def main():
    print("Starting Pipeline using Config Paths...")

    datasets = ['Poland', 'Brasil']
    tasks = ['ADL1', 'ADL2', 'ADL3']

    # ----------------------------------------------------
    # Phase 1: Processing & Feature Extraction
    # ----------------------------------------------------
    for dataset in datasets:
        print(f"\n================ {dataset.upper()} ================")
        
        for task in tasks:
            print(f"--> Processing {task}...")

            # 1. Get Paths
            try:
                paths = config.get_paths(dataset, task)
            except Exception as e:
                print(f"   [ERROR] Error getting paths for {dataset}/{task}: {e}")
                continue

            # 2. Collect JSON Files
            valid_suffixes = [f"{task}S", f"{task}F"]
            json_files = config.collect_files(paths['json_root'], valid_suffixes)
            
            if not json_files:
                print(f"   [WARNING] No JSON files found in: {paths['json_root']}")
                continue
            
            # 3. Process JSON -> CSV (Data Loader)
            out_folders = {
                'frontal_csv': paths['frontal_csv'],
                'sagittal_csv': paths['sagittal_csv']
            }
            data_loader.process_adl_json(json_files, out_folders, task)

            # 4. Extract Final Features
            if not os.path.exists(paths['timing_csv']):
                print(f"   [ERROR] Missing Timing CSV: {paths['timing_csv']}")
                continue

            final_output = os.path.join(paths['features_dir'], f"{dataset}_{task}_Final_Metrics.csv")
            print(f"   Extracting features -> {os.path.basename(final_output)}")

            if task == "ADL1":
                feature_extractor.extract_features_adl1(
                    base_dir=paths['task_out_dir'],
                    summary_csv=paths['timing_csv'], 
                    output_file=final_output
                )
            elif task == "ADL2":
                feature_extractor.extract_features_adl2(
                    base_dir=paths['task_out_dir'],
                    summary_csv=paths['timing_csv'],
                    output_file=final_output
                )
            elif task == "ADL3":
                feature_extractor.extract_features_adl3(
                    base_dir=paths['task_out_dir'],
                    summary_csv=paths['timing_csv'],
                    output_file=final_output
                )

    # ----------------------------------------------------
    # Phase 2: Data Merging
    # ----------------------------------------------------
    print("\n================ DATA MERGING ================")
    
    # 1. Merge ADL1 + ADL2 + ADL3 for Poland
    poland_combined_path = feature_merging.merge_adls_for_country("Poland")
    
    # 2. Merge ADL1 + ADL2 + ADL3 for Brasil
    brasil_combined_path = feature_merging.merge_adls_for_country("Brasil")
    
    # 3. Combine Poland + Brasil into one Master Dataset
    if poland_combined_path and brasil_combined_path:
        feature_merging.combine_all_datasets(poland_combined_path, brasil_combined_path)
    else:
        print("[ERROR] Could not create global dataset because one or more country files failed.")

    print("\nPipeline Finished Successfully.")

if __name__ == "__main__":
    main()