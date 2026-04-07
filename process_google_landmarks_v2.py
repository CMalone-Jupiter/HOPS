#! /usr/bin/env python3
"""
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

Google Landmarks feature extraction and retrieval evaluation.

Extracts SALAD VPR descriptors from Google Landmarks v2, averages them
per landmark ID, and evaluates recall@1 on the validation set.

Part of the HOPS codebase:
"A Hyperdimensional One Place Signature to Represent Them All:
 Stackable Descriptors For Visual Place Recognition"
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tfm
from PIL import Image
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"




def main():

    args = parse_args()
    root = os.path.dirname(args.dataset_path.rstrip("/"))

    output_dir = args.output_dir or os.path.join(root, "features_salad")
    avg_dir = args.avg_dir or os.path.join(root, "features_salad_averaged")
    train_csv = args.train_csv or os.path.join(root, "train.csv")
    val_csv = args.val_csv or os.path.join(root, "val.csv")

    print(f"Dataset path : {args.dataset_path}")
    print(f"Output dir   : {output_dir}")
    print(f"Avg dir      : {avg_dir}")
    print(f"Train CSV    : {train_csv}")
    print(f"Val CSV      : {val_csv}")
    print(f"Device       : {device}\n")


    if not args.skip_extraction:
        model, preprocess = load_salad_model()
        extract_features(model, preprocess, args.dataset_path, output_dir)
        del model  # free GPU memory
        torch.cuda.empty_cache()


    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    landmark_ids, fused_ref = fuse_descriptors(train_df, output_dir, avg_dir)
    print(f"\nNumber of landmarks: {len(landmark_ids)}")
    print(f"Number of validation images: {len(val_df)}\n")
    print(f"HOPS ref descriptors shape: {fused_ref.shape}")


    evaluate_recall(val_df, output_dir, landmark_ids, fused_ref)

    evaluate_recall_original(train_df, val_df, output_dir)
    
    
    

def load_salad_model():
    """Load the DINOv2-SALAD model and return (model, preprocessing)."""
    model = torch.hub.load("serizba/salad", "dinov2_salad")
    model.eval().to(device)
    preprocess = tfm.Compose([tfm.Resize([322, 322], antialias=True), tfm.ToTensor()])
    return model, preprocess


def extract_features(model, preprocess, dataset_path, output_dir, batch_size=1):
    """
    Extract per-image SALAD descriptors and save as individual .npy files.
    Skips images whose .npy already exists so the process is resumable.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_names = sorted(os.listdir(dataset_path))

    for img_name in tqdm(image_names, desc="Extracting features"):
        stem = os.path.splitext(img_name)[0]
        out_path = os.path.join(output_dir, f"{stem}.npy")
        if os.path.isfile(out_path):
            continue

        img = Image.open(os.path.join(dataset_path, img_name)).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(tensor).cpu().numpy()

        np.save(out_path, features)


def load_feature(output_dir, img_name):
    """Load a single image's descriptor; returns None if missing."""
    stem = os.path.splitext(img_name)[0]
    path = os.path.join(output_dir, f"{stem}.npy")
    if not os.path.isfile(path):
        return None
    return np.load(path).squeeze().astype(np.float32)


def fuse_descriptors(train_df, output_dir, avg_dir):
    """
    Compute HOPS descriptors by landmark ID and save one .npy each.

    Returns
    -------
    landmark_ids : list[int]
    fused_ref : np.ndarray of shape (n_landmarks, D)
    """

    os.makedirs(avg_dir, exist_ok=True)
    groups = train_df.groupby("landmark_id")["filename"].apply(list)

    landmark_ids, fused_ref = [], []

    for lid, filenames in tqdm(groups.items(), total=len(groups), desc="Averaging descriptors"):
        out_path = os.path.join(avg_dir, f"landmark_{lid}_avg.npy")

        # Re-use cached average if it exists
        if os.path.isfile(out_path):
            avg = np.load(out_path).squeeze().astype(np.float32)
        else:
            feats = [load_feature(output_dir, f) for f in filenames]
            feats = [f for f in feats if f is not None]
            avg = np.mean(feats, axis=0).astype(np.float32)
            np.save(out_path, avg)

        landmark_ids.append(lid)
        fused_ref.append(avg)

    return landmark_ids, np.stack(fused_ref)


def evaluate_recall(val_df, output_dir, landmark_ids, fused_ref):
    """
    Recall@1 — for each validation image, check whether the closest
    HOPS landmark descriptor matches the true landmark ID.

    Parameters
    ----------
    val_df        : DataFrame with columns [filename, landmark_id]
    output_dir    : directory with per-image .npy descriptors
    landmark_ids  : list[int] aligned with rows of *fused_ref*
    fused_ref     : (n_landmarks, D) normalised descriptor matrix

    Returns
    -------
    recall : float
    """
    norm_fused_ref = normalize(fused_ref, axis=1)

    correct, total = 0, 0
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Evaluating"):
        feat_qry = load_feature(output_dir, row["filename"])
        norm_feat_qry = normalize(feat_qry.reshape(1, -1))

        similarities = norm_fused_ref @ norm_feat_qry.T  # (n_landmarks, 1)
        predicted_id = landmark_ids[np.argmax(similarities)]

        correct += int(predicted_id == row["landmark_id"])
        total += 1

    recall = correct / total if total else 0.0
    print(f"\nRecall@1 (HOPS references): {recall:.4f} ({correct}/{total})\n")
    return recall


def evaluate_recall_original(train_df, val_df, output_dir):
    """
    Recall@1 using every individual training image as a candidate.
    """
    train_feats, train_labels = [], []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Loading train descriptors"):
        feat = load_feature(output_dir, row["filename"])
        train_feats.append(feat)
        train_labels.append(row["landmark_id"])
    train_matrix = normalize(np.stack(train_feats), axis=1)
    train_labels = np.array(train_labels)

    val_feats, val_labels = [], []
    for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Loading val descriptors"):
        feat = load_feature(output_dir, row["filename"])
        val_feats.append(feat)
        val_labels.append(row["landmark_id"])
    val_matrix = normalize(np.stack(val_feats), axis=1)
    val_labels = np.array(val_labels)
    
    print(f"Original ref descriptor shape: {train_matrix.shape}")

    # Vectorised cosine similarity
    sim = val_matrix @ train_matrix.T  # (n_val, n_train)
    predicted = train_labels[np.argmax(sim, axis=1)]

    correct = int(np.sum(predicted == val_labels))
    total = len(val_labels)
    recall = correct / total if total else 0.0
    print(f"\nRecall@1 (original): {recall:.4f}  ({correct}/{total})")
    return recall


def parse_args():
    
    dataset_path = "./datasets/google-landmark-dataset-v2-micro/gldv2_micro/images"
    output_dir = "./datasets/google-landmark-dataset-v2-micro/features_salad"
    agv_dir = "./datasets/google-landmark-dataset-v2-micro/features_salad_averaged"
    train_csv = "./datasets/google-landmark-dataset-v2-micro/gldv2_micro/train.csv"
    val_csv = "./datasets/google-landmark-dataset-v2-micro/gldv2_micro/val.csv"
    
    
    parser = argparse.ArgumentParser(description="Google Landmarks feature extraction and evaluation")
    parser.add_argument("--dataset_path", type=str, default=dataset_path,
                        help="Path to the image directory (e.g. gldv2_micro/images).")
    parser.add_argument("--output_dir", type=str, default=output_dir,
                        help="Directory for per-image .npy descriptors. Default: <dataset_path>/../features_salad")
    parser.add_argument("--avg_dir", type=str, default=agv_dir,
                        help="Directory for averaged landmark descriptors. Default: <dataset_path>/../features_salad_averaged")
    parser.add_argument("--train_csv", type=str, default=train_csv,
                        help="Path to train.csv. Default: <dataset_path>/../train.csv")
    parser.add_argument("--val_csv", type=str, default=val_csv,
                        help="Path to val.csv. Default: <dataset_path>/../val.csv")
    parser.add_argument("--skip_extraction", action="store_true", default=False,
                        help="Skip feature extraction (use existing .npy files).")
    parser.add_argument("--bruteforce", action="store_true", default=True,
                        help="Also run brute-force (per-image) recall evaluation.")
    return parser.parse_args()




if __name__ == "__main__":
    main()