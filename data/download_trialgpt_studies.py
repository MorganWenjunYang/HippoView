# Download datasets from TrialGPT
# Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10418514/

import os
import urllib.request
import json
import pandas as pd
from pathlib import Path

# Base URLs
TRIALGPT_BASE = "https://raw.githubusercontent.com/ncbi-nlp/TrialGPT/main/dataset"
TRIALGPT_DIR = Path("./data/raw/trialgpt/")

# Create directories
TRIALGPT_DIR.mkdir(parents=True, exist_ok=True)
SIGIR_DIR = TRIALGPT_DIR / "sigir"
TREC_DIR = TRIALGPT_DIR / "trec"
SIGIR_DIR.mkdir(exist_ok=True)
TREC_DIR.mkdir(exist_ok=True)

def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL to output path with error handling."""
    try:
        print(f"Downloading {url} to {output_path}")
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

# Download TREC 2021 dataset
if not (TREC_DIR / "trial_2021.json").exists():
    download_file(
        f"{TRIALGPT_BASE}/trec_2021/retrieved_trials.json",
        TREC_DIR / "trial_2021.json"
    )

# Download TREC 2022 dataset
if not (TREC_DIR / "trial_2022.json").exists():
    download_file(
        f"{TRIALGPT_BASE}/trec_2022/retrieved_trials.json",
        TREC_DIR / "trial_2022.json"
    )

# Download SIGIR dataset
if not (SIGIR_DIR / "trial_sigir.json").exists():
    download_file(
        f"{TRIALGPT_BASE}/sigir/retrieved_trials.json",
        SIGIR_DIR / "trial_sigir.json"
    )

# # Download trial2info mapping
# if not (TRIALGPT_DIR / "trial2info.json").exists():
#     if download_file(
#         f"{TRIALGPT_BASE}/trial2info.json",
#         TRIALGPT_DIR / "trial2info.json"
#     ):
#         # Process trial2info.json to create studies list
#         try:
#             with open(TRIALGPT_DIR / "trial2info.json", 'r') as f:
#                 data = json.load(f)
            
#             df = pd.DataFrame.from_dict(data, orient='index')
#             df.index.name = "nct_id"
#             df.to_csv(TRIALGPT_DIR / "trialgpt.studies_list.csv")
#             print("Successfully created trialgpt.studies_list.csv")
#         except Exception as e:
#             print(f"Error processing trial2info.json: {str(e)}")

print("Download process completed. Check the directories for downloaded files.")


def get_studies_list(data_dir: Path) -> pd.DataFrame:
    """Get unique study NCT IDs from all TrialGPT datasets.
    
    Args:
        data_dir: Path to the root directory containing TrialGPT data
        
    Returns:
        DataFrame containing unique NCT IDs and their source
    """
    nct_ids = []
    sources = []
    
    def process_json_file(file_path: Path, source: str):
        """Process a JSON file and extract NCT IDs."""
        if file_path.exists():
            print(f"\nProcessing {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f)
                print(f"Data type: {type(data)}")
                if isinstance(data, list):
                    print(f"Number of items in list: {len(data)}")
                    for item in data:
                        if isinstance(item, dict):
                            # Check for trials in the item
                            for key in item:
                                if key.isdigit():  # This is a trial list
                                    trials = item[key]
                                    if isinstance(trials, list):
                                        for trial in trials:
                                            if isinstance(trial, dict) and 'NCTID' in trial:
                                                nct_ids.append(trial['NCTID'])
                                                sources.append(source)
                                            elif isinstance(trial, dict):
                                                print(f"Trial missing NCTID. Available keys: {trial.keys()}")
                else:
                    print(f"Unexpected data type: {type(data)}")
        else:
            print(f"File not found: {file_path}")
    
    # Process SIGIR data
    process_json_file(data_dir / "sigir" / "trial_sigir.json", 'sigir')
    
    # Process TREC 2021 data
    process_json_file(data_dir / "trec" / "trial_2021.json", 'trec_2021')
    
    # Process TREC 2022 data
    process_json_file(data_dir / "trec" / "trial_2022.json", 'trec_2022')
    
    # Create DataFrame with unique NCT IDs
    df = pd.DataFrame({
        'nct_id': nct_ids,
        'source': sources
    })
    
    # Remove duplicates while keeping the first occurrence
    df = df.drop_duplicates(subset=['nct_id'], keep='first')
    
    # Save to CSV
    output_file = data_dir / "trialgpt.studies_list.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} unique NCT IDs to {output_file}")
    
    return df

get_studies_list(TRIALGPT_DIR)
