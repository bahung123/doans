from pathlib import Path
import os

def download_data():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub not installed. Installing...")
        os.system("pip install huggingface_hub")
        from huggingface_hub import snapshot_download
    
    HF_REPO_ID = os.getenv("HF_DATA_REPO", "hungnha/do_an_tot_nghiep")
    
    data_path = Path("data")
    
    if data_path.exists() and any(data_path.iterdir()):
        print("Data folder already exists. Skipping download.")
        print(f"To re-download, delete the 'data/' folder first.")
        return
    
    print(f"Downloading data from HuggingFace: {HF_REPO_ID}")
    print("This may take a few minutes...")
    
    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir="data",
            local_dir_use_symlinks=False,
        )
        print("Download complete!")
        print(f"Data saved to: {data_path.absolute()}")
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise

if __name__ == "__main__":
    download_data()
