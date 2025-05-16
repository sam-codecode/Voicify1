import pandas as pd
import glob
import os

# Get all CSV files from all subfolders under 'dataset'
csv_files = glob.glob('dataset/*/*.csv')  # Assumes 3 folders are inside 'dataset/'

# Group files by base filename (e.g., A_dataset.csv)
grouped_files = {}
for file in csv_files:
    base_name = os.path.basename(file)
    grouped_files.setdefault(base_name, []).append(file)

# Create directory for intermediate combined files
os.makedirs('combined', exist_ok=True)

final_dfs = []

# Combine all files with same name
for base_name, files in grouped_files.items():
    combined_df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # Save the combined per-class file
    combined_path = os.path.join('combined', base_name)
    combined_df.to_csv(combined_path, index=False)

    # Add to list for final combination
    final_dfs.append(combined_df)

# Combine all per-class files into a final dataset (no label column added)
full_data = pd.concat(final_dfs, ignore_index=True)
full_data.to_csv('full_landmark_dataset3.csv', index=False)

print("✅ All datasets combined into full_landmark_dataset.csv (no labels included).")