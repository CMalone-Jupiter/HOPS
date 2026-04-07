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

VPR analysis script for:
"A Hyperdimensional One Place Signature to Represent Them All:
 Stackable Descriptors For Visual Place Recognition"

Evaluates Visual Place Recognition performance across multiple reference
set combinations using HOPS.
"""

import argparse
import numpy as np
import os
from itertools import combinations
import torch
from scipy.spatial.distance import cdist

torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("CUDA availability: " + str(torch.cuda.is_available()))


DATASETS_PRESET = {
    "Nordland": {
        "conditions": [ "spring", "summer", "fall", "winter"],
        "img_tol": 1,
    },
    "RobotCar": {
        "conditions": ["Sun", "Rain", "Dusk", "Overcast", "Overcast3", "Night"],
        "img_tol": 3,
    },
    "SFU-Mountain": {
        "conditions": ["dry", "dusk", "jan", "night", "nov", "sept", "wet"],
        "img_tol": 2,
    },
}




def main():
    args = parse_args()

    preset = DATASETS_PRESET[args.dataset]
    reference_sets = args.conditions if args.conditions else preset["conditions"]
    query_sets = args.query_sets if args.query_sets else reference_sets
    img_tol = args.img_tol if args.img_tol is not None else preset["img_tol"]

    print(f"Dataset     : {args.dataset}")
    print(f"Method      : {args.method}")
    print(f"Conditions  : {reference_sets}")
    print(f"Query sets  : {query_sets}")
    print(f"Image tol.  : {img_tol}")
    print(f"Metric      : {args.metric}")
    print(f"Device      : {'CPU' if args.use_cpu else device}\n")


    all_ftrs = load_descriptors(args.descriptor_path, args.method, args.dataset, reference_sets)
    
    for cond, ftrs in all_ftrs.items():
        print(f"{cond} descriptor shape: {ftrs.shape}")

    evaluate(all_ftrs, query_sets, reference_sets, img_tol, use_gpu=not args.use_cpu, metric=args.metric)




def getMatchIndsCPU(ft_ref, ft_qry, topK=20, metric="cosine"):
    """
    CPU-based cosine or euclidean nearest-neighbour matching.
    Returns match indices (topK x num_queries) and the full distance matrix.
    """
    if metric == "euclidean":
        dMat = cdist(ft_ref, ft_qry, metric)
    else: # cosine
        ft_qry_norm = ft_qry / np.linalg.norm(ft_qry, axis=1, keepdims=True) # Shape (M, N)
        ft_ref_norm = ft_ref / np.linalg.norm(ft_ref, axis=1, keepdims=True) # Shape (C, N)
        dMat = 1 - (ft_ref_norm @ ft_qry_norm.T)

    mInds = np.argsort(dMat, axis=0)[:topK].squeeze() # shape: K x ft_qry.shape[0]
    return mInds, dMat


def getMatchIndsGPU(ft_ref, ft_qry, topK=20, metric="cosine"):
    """
    GPU-accelerated cosine or euclidean nearest-neighbour matching.
    Accepts numpy arrays; returns torch tensors on *device*.
    """
    ft_qry_tensor = torch.tensor(ft_qry, dtype=torch.float32, device=device)
    ft_ref_tensor = torch.tensor(ft_ref, dtype=torch.float32, device=device)

    if metric == "euclidean":
        dMat = torch.cdist(ft_ref_tensor, ft_qry_tensor)
    else:  # cosine
        ft_qry_norm = ft_qry_tensor / ft_qry_tensor.norm(dim=1, keepdim=True)
        ft_ref_norm = ft_ref_tensor / ft_ref_tensor.norm(dim=1, keepdim=True)
        dMat = 1 - ft_ref_norm @ ft_qry_norm.t()

    mInds = torch.argsort(dMat, dim=0)[:topK].squeeze()
    return mInds, dMat


def load_descriptors(descriptor_path, method, dataset, conditions):
    """
    Load pre-computed descriptors for every condition.

    Expected file layout (inside *descriptor_path*):
        VPR-methods-evaluation/logs/default/{method}_{dataset}_{ref_cond}_{qry_cond}.npy

    Because every pair (ref, qry) is stored as a separate file we load
    *database_descriptors.npy* (reference) and *queries_descriptors.npy*
    (query) for every ordered pair.  For the brute-force combination
    evaluation we only need the per-condition feature matrices, so we
    deduplicate by loading each condition once as a reference descriptor.

    Returns
    -------
    all_ftrs : dict[str, np.ndarray]   - {condition_name: features}
    """
    all_ftrs = {}
    base = os.path.join(descriptor_path, "VPR-methods-evaluation", "logs", "default")

    for cond in conditions:
        # Try to find ANY file where this condition appears as the
        # reference (3rd element) so we can grab its database descriptors.
        loaded = False
        for other in conditions:
            if other == cond:
                continue
            folder = os.path.join(base, f"{method}_{dataset}_{cond}_{other}")
            fpath = os.path.join(folder, "database_descriptors.npy")
            # print(f"Looking for {fpath}...")
            
            if os.path.isfile(fpath):
                print(f"Found {fpath}, loading descriptors for condition '{cond}'...")
                all_ftrs[cond] = (np.load(fpath).squeeze().astype(np.float32))
                loaded = True
                break

        # Fallback: try loading as a query descriptor instead.
        if not loaded:
            for other in conditions:
                if other == cond:
                    continue
                folder = os.path.join(base, f"{method}_{dataset}_{other}_{cond}")
                fpath = os.path.join(folder, "queries_descriptors.npy")
                # print(f"Looking for {fpath}...")
                
                if os.path.isfile(fpath):
                    print(f"Found {fpath}, loading descriptors for condition '{cond}'...")
                    all_ftrs[cond] = (np.load(fpath).squeeze().astype(np.float32))
                    loaded = True
                    break

        if not loaded:
            raise FileNotFoundError(f"Could not find descriptors for condition '{cond}' under {base}/")

    return all_ftrs


def compute_recall_at_1(mInds, img_tol):
    """Recall@1 assuming 1-to-1 query<->reference index correspondence."""
    gt = torch.arange(mInds.shape[0], device=mInds.device)
    correct = (torch.abs(mInds - gt) < img_tol).float()
    return correct.sum() / mInds.shape[0], correct


def fuse_descriptors(all_ftrs, ref_conditions):
    """
    Element-wise sum of descriptors from the given reference conditions.

    Parameters
    ----------
    all_ftrs        : dict[str, np.ndarray]  - per-condition features
    ref_conditions  : list[str]              - conditions to fuse

    Returns
    -------
    fused : np.ndarray of shape (N_images, D_features)
    """
    fused = np.zeros_like(all_ftrs[ref_conditions[0]])
    for cond in ref_conditions:
        fused += all_ftrs[cond]
    return fused


def evaluate(all_ftrs, query_sets, reference_sets, img_tol, use_gpu=True, metric="cosine"):
    """
    Brute-force evaluation over all combinations of reference sets.

    For each query condition and for every k in [1 .. len(reference_sets)-1],
    all C(n-1, k) reference-set combinations are evaluated.
    """
    match_fn = getMatchIndsGPU if use_gpu else getMatchIndsCPU

    for qry in query_sets:
        n_queries = all_ftrs[qry].shape[0]
        success = torch.zeros(n_queries)

        # Available reference conditions (everything except the query)
        available_refs = [r for r in reference_sets if r != qry]

        # --- Single-reference evaluation (also accumulates oracle) ----------
        for ref in available_refs:
            mInds, _ = match_fn(all_ftrs[ref], all_ftrs[qry], topK=1, metric=metric)
            recall, correct = compute_recall_at_1(mInds, img_tol)
            
            print(f"Recall@1 for {qry} Query and {ref} Ref: {recall:.4f}")
            success += correct.cpu()

        # --- Multi-reference (fused) evaluation -----------------------------
        k = len(available_refs)  # Only evaluate the full fusion case for simplicity
        for combo in combinations(available_refs, k):
            fused_ref = fuse_descriptors(all_ftrs, list(combo))
            mInds, _ = match_fn(fused_ref, all_ftrs[qry], topK=1, metric=metric)
            recall, _ = compute_recall_at_1(mInds, img_tol)
            combo_str = "-".join(combo)
            print(f"Recall@1 for {qry} Query and {combo_str} Ref: {recall:.4f}")

        # --- Oracle performance ceiling -------------------------------------
        ceiling = (success > 0).float().sum() / n_queries
        print(f"Performance Ceiling for {qry} Query: {ceiling:.4f}\n")


def parse_args():
    
    dataset = "Nordland"
    method = "mixvpr"
    descriptor_path = "./"
    conditions = ["spring", "summer", "fall", "winter"]
    query_sets = ["winter"]
    
    dataset = "RobotCar"
    method = "mixvpr"
    descriptor_path = "./"
    conditions = ["Sun", "Rain", "Dusk", "Overcast", "Overcast3", "Night"]
    query_sets = ["Night"]
    
    dataset = "SFU-Mountain"
    method = "mixvpr"
    descriptor_path = "./"
    conditions = ["dry", "dusk", "jan", "night", "nov", "sept", "wet"]
    query_sets = ["dry"]
    
    
    
    parser = argparse.ArgumentParser(description=("Evaluate VPR recall@1 across all reference-set combinations (HOPS)."))
    parser.add_argument("--dataset", type=str, choices=list(DATASETS_PRESET.keys()), default=dataset,
                        help="Name of the evaluation dataset.")
    parser.add_argument("--method", type=str, default=method,
                        help=("VPR method name (e.g. cosplace, eigenplaces, mixvpr, salad, cricavpr, boq, anyloc, netvlad)."))
    parser.add_argument("--descriptor_path", type=str, default=descriptor_path, 
                        help=("Root path that contains the 'VPR-methods-evaluation/logs/default/' folder tree. Default: current directory."))
    parser.add_argument("--conditions", type=str,  nargs="+", default=conditions,
                        help=("Override the list of conditions (traversals) to use. If not provided the dataset preset is used."))
    parser.add_argument("--query_sets", type=str, nargs="+", default=query_sets,
                        help=("Subset of conditions to use as queries. Defaults to all conditions."))
    parser.add_argument("--img_tol", type=int, default=None,
                        help=("Tolerance in image indices for a match to be considered correct. Overrides the dataset default."))
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"],
                        help="Distance metric for descriptor matching (default: cosine).")
    parser.add_argument("--use_cpu", action="store_true", default=False,
                        help="Force CPU matching even when a GPU is available.")
    return parser.parse_args()




if __name__ == "__main__":
    main()

