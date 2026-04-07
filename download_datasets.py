#! /usr/bin/env python3
'''
MIT License

Copyright (c) 2025 Connor Malone, Somayeh Hussaini, Tobias Fischer and Michael Milford 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import tarfile
import argparse
import kagglehub
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download datasets by name.")
    parser.add_argument("--dataset", type=str, required=True, choices=["nordland", "google-landmark-v2"],
                        help="Name of the dataset to download: 'nordland', or 'google-landmark-v2'")
    args = parser.parse_args()

    if args.dataset == "nordland":
        dataset_path = os.path.join("datasets", "Nordland")
        print("Downloading Nordland dataset...")
        snapshot_download(repo_id="Somayeh-h/Nordland", repo_type="dataset", local_dir=dataset_path)
        
        data_folder = os.path.join(dataset_path, "data")
        if os.path.exists(data_folder):
            tar_files = [f for f in os.listdir(data_folder) if f.endswith(".tar.gz")]
            
            if not tar_files:
                print("No .tar.gz files found in the data folder.")
            else:
                for tar_file in tar_files:
                    tar_path = os.path.join(data_folder, tar_file)
                    print(f"Extracting {tar_file}...")
                    try:
                        with tarfile.open(tar_path, "r:gz") as tar:
                            tar.extractall(path=data_folder)
                        print(f"Extracted {tar_file} successfully.")
                    except Exception as e:
                        print(f"Failed to extract {tar_file}: {e}")
                
                print(f"The Nordland dataset is saved and extracted at: {os.path.abspath(data_folder)}")
        else:
            print(f"Data folder not found at {data_folder}")

    elif args.dataset == "google-landmark-v2":
        dataset_path = "datasets/google-landmark-dataset-v2-micro"
        os.makedirs(dataset_path, exist_ok=True)
        print("Downloading Google Landmarks v2 dataset...")
        path = kagglehub.dataset_download("confirm/google-landmark-dataset-v2-micro",
                                            output_dir=dataset_path)
        print(f"The google landmarks dataset is saved at: {path}")




if __name__ == "__main__":
    main()
    
    
    