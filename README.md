# xAI-Proj-M: Robust Machine Learning

- Student: Sebastian Drefahl
- GITHUB-images: https://github.com/deadPixelsGreta/xAI-proj-m-ws2526_img
- StudentID: IOKJ
- PhoneID: x11
  - iPhone 11

- Student: Stephanus Park
- GITHUB-images: https://github.com/stu-rdy/xai-proj-m-ws2526
- StudentID: GBQL
- PhoneID: x14
  - iPhone 14

- Student: Mehdi Huseynov
- GITHUB-images: https://github.com/MehdiHuseynovLive/xAI-proj-m-ws2526_Pictures 
- StudentID: NLUB
- PhoneID: x14Pro
  - iphone 14Pro

# Task images
We have to take 30 pictures per person per class.
We have 10 classes:
coffee mug, wooden spoon, notebook, teapot, soup bowl,remote control, computer keyboard, mouse, toilet tissue, binder

*naming convention:*
studentID_phoneID_classname_imageID.jpg


*Deadline:* 
* Deadline 1: 23.11.2025
* Deadline 2: 14.12.2025

---

# Project Documentation

A comprehensive framework for training, evaluating, and analyzing robust machine learning models using ensembles and mixture-of-experts (MoE). This project focuses on improving model performance and reliability on difficult subsets of ImageNet through diversity-aware training and expert routing.

## Key Features

-   **Robust Ensembles**: Train diverse ensembles using specialized augmentation policies (Blur, Color, Geometric, SOTA).
-   **Mixture of Experts**: route samples to the most capable expert model using learned routers.
-   **Failure Analysis**: Detailed breakdowns of model failures and "missed opportunities" where at least one expert was correct.
-   **Temperature Calibration**: Calibrate model confidence for better reliability and routing.
-   **WandB Integration**: Full logging support for Weights & Biases.

## BOX Installation

To get started, clone the repository and set up the environment:

```bash
# 1. Clone the repository
git clone https://github.com/deadPixelsGreta/xAI-proj-m-ws2526.git
cd xAI-proj-m-ws2526

# 2. Create the conda environment
conda env create -f environment.yaml

# 3. Activate the environment
conda activate xai
```

## Getting Started

### 1. Training Single Models

Train individual backbone models (e.g., DenseNet121, ResNet34) with specific augmentation policies to induce diversity.

```bash
python experiments/base_ensemble/scripts/train_single_model.py \
    --model densenet121 \
    --data-dir datasets \
    --aug-policy geometric \
    --epochs 30 \
    --wandb
```

**Supported Augmentation Policies:**
-   `blur`: Focuses on gaussian/motion blur and compression artifacts.
-   `color`: Focuses on color jittering, solarization, and grayscale.
-   `geometric`: Focuses on affine transforms and perspective shifts.
-   `vit`: Strong augmentations suitable for Vision Transformers.

### 2. Ensemble Inference

Evaluate adherence and performance of your trained ensemble.

```bash
python experiments/base_ensemble/scripts/ensemble_inference.py \
    --checkpoints checkpoints/model1.pth checkpoints/model2.pth \
    --evaluate \
    --split test
```

### 3. Mixture of Experts (Router)

Train a routing network to dynamically select the best expert for each input.

```bash
python experiments/mixture_of_experts/scripts/train_router.py \
    --config experiments/mixture_of_experts/configs/default_router.yaml
```

## Project Structure

```
xAI-proj-m-ws2526/
├── experiments/
│   ├── base_ensemble/          # Core training and ensemble logic
│   │   ├── src/
│   │   │   ├── data/           # Datasets and diversity augmentations
│   │   │   ├── models/         # Model architectures
│   │   │   └── training/       # Training loops
│   │   └── scripts/            # CLI scripts for training/inference
│   ├── mixture_of_experts/     # MoE routing and analysis
│   └── temp_calibration/       # Temperature scaling
├── environment.yaml            # Conda environment definition
└── README.md                   # Project documentation
```
