# A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For Visual Place Recognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![QUT Centre for Robotics](https://img.shields.io/badge/collection-QUT%20Robotics-%23043d71?style=flat)](https://qcr.ai)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)
[![stars](https://img.shields.io/github/stars/CMalone-Jupiter/HOPS?style=flat)](https://github.com/CMalone-Jupiter/HOPS/stargazers)
[![GitHub repo size](https://img.shields.io/github/repo-size/CMalone-Jupiter/HOPS?style=flat)](./README.md)



The code repository for our ICCV 2025 paper "A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For Visual Place Recognition". Our approach, named HOPS, fuses VPR reference sets using hyperdimensional framework, which improves visual place recognition by efficiently combining descriptors from images of a place captured in different environmental conditions, enabling better localisation performance without increasing inference computational demands.

![Paper Image Placeholder](./assets/HOPS-Banner.svg)


> **Paper:** *A Hyperdimensional One Place Signature to Represent Them All: Stackable Descriptors For Visual Place Recognition*  
> **Authors:** Connor Malone, Somayeh Hussaini, Tobias Fischer and Michael Milford

> 📄 **Arxiv paper link:** https://arxiv.org/abs/2412.06153

> 📄 **Published paper link:** https://openaccess.thecvf.com/content/ICCV2025/html/Malone_A_Hyperdimensional_One_Place_Signature_to_Represent_Them_All_Stackable_ICCV_2025_paper.html

> ▶️ **Video:** [Watch on YouTube](https://www.youtube.com/watch?v=PpuTfu-y0Zk)

## Citation
If you find this work useful, please cite:
```
@inproceedings{malone2025hyperdimensional,
  title={A hyperdimensional one place signature to represent them all: Stackable descriptors for visual place recognition},
  author={Malone, Connor and Hussaini, Somayeh and Fischer, Tobias and Milford, Michael},
  booktitle={IEEE/CVF International Conference on Computer Vision},
  pages={9822--9833},
  year={2025}
}
```


## Quick Start
### Installation 
We provide installation guidelines for [pixi](https://pixi.prefix.dev/latest/) workflow.

Pixi is an easy-to-use and fast and reproducible package manager. If you don't already have pixi installed, please follow the instructions from their website [here](https://pixi.sh/latest/installation/).  

From the repo root:
```
# Create the environment defined in pixi.toml
pixi install

# Activate the environment
pixi shell
```

Our main code assumes that you have pre-extracted the VPR feature descriptors for the datasets you plan to use, and the features descriptors are stored under the below folder structure: 

```
<parent_directory>
├── VPR-HDC-Fuse/
|  ├── main.py
|  └── ...
|  ├── VPR-methods-evaluation/logs/default/
|    ├── {VPR_method_name}_{dataset_name}_{condition1_name}_{condition2_name} # features as extracted by VPR method for condition 1 and 2, stored as .npy files 
|    └── ...
|  ├── Dataset 1                               # Root directory for dataset
|    ├── Condition 1                           # Condition 1 for the dataset
|       ├── (images stored directly within)    # Directory level where images are stored            
|    ├── Condition 2                           # Condition 2 for the dataset
|       ├── (images stored directly within)    # Directory level where images are stored  
```

Below we describe how we obtain the datasets and extract the VPR feature descriptors. 

## Download the datasets
### Public datasets 
We provide the download_datasets.py script to enable downloading Nordland and Google Landmarks v2 as below. 
```
pixi run python download_datasets.py --dataset nordland
pixi run python download_datasets.py --dataset google-landmark-v2
```
Please note that if you have activated your pixi environment pixi shell, then you do not need to have pixi run at the start of the commands above. 


### SFU Mountain and Oxford RobotCar
For SFU Mountain and Oxford RobotCar datasets, visit:

SFU Mountain: [autonomy.cs.sfu.ca/sfu-mountain-dataset/](https://autonomy.cs.sfu.ca/sfu-mountain-dataset/)

Oxford RobotCar: [robotcar-dataset.robots.ox.ac.uk/datasets/](https://robotcar-dataset.robots.ox.ac.uk/datasets/)

Download and extract to datasets directory. For SFU-Mountain dataset, we put images from part 1 and part b of each condition in one folder. 

For Oxford RobotCar, we provide e.g `dataset_imagenames/ORC_Dusk_timestamps.csv` files which maps the original timestamped dataset image names to our sampled index-based image names. 


## Extract VPR Features
This repository includes a forked version of the [VPR-methods-evaluation](https://github.com/gmberton/VPR-methods-evaluation) repository as a Git submodule, available here: https://github.com/Somayeh-h/VPR-methods-evaluation. Initialise it first by running:

```
cd VPR-methods-evaluation
git submodule update --init --recursive
```

Generate descriptors for your chosen VPR method using the provided configuration:

```
# Example: Extract CosPlace features on Nordland dataset
pixi run python main.py --method=cosplace --backbone=VGG16 --descriptors_dimension=512 \
  --no_labels --img_filetype=png --save_descriptors --num_workers=1 --num_images=10000 \
  --database_folder=path/to/reference/images \
  --queries_folder=path/to/query/images
```
Refer to [`VPR-methods-evaluation/feature_generation_configs.sh`](https://github.com/CMalone-Jupiter/HOPS/blob/main/VPR-methods-evaluation/feature_generation_configs.sh) for pre-configured extraction commands across a range of datasets and methods.

## Fuse VPR reference features using HOPS and evaluate them 

Fuse reference condition descriptors and compute Recall@1:
```
pixi run python main.py --dataset RobotCar --method cosplace --descriptor_path . --conditions Sun Rain Dusk Overcast Overcast3 Night 
```

Key parameters:

* --dataset: Dataset name (RobotCar, Nordland, SFU-Mountain)
* --method: VPR method (cosplace, eigenplaces, mixvpr, salad, cricavpr, boq, anyloc, netvlad, etc)
* --conditions: Reference conditions to fuse
* --query_sets: Query conditions (defaults to all conditions)
* --metric: Distance metric (cosine or euclidean)
* --use_cpu: Force CPU matching instead of GPU


## Acknowledgements
This research was partially supported and funded by the QUT Centre for Robotics, ARC Laureate Fellowship FL210100156 to MM, and ARC DECRA Fellowship DE240100149 to TF.

