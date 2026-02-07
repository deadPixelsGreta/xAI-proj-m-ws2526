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
```
```bash
# 2. Create the conda environment
conda env create -f environment.yaml
```
```bash
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

### 4. ResNet Augmentation Experiments

Systematic experiments with ResNet models (18,34,50) to evaluate different augmentation strategies and hyperparameter optimization.

#### Start
First config the parameters and decide, whiche model do you want to train and what kind of training. You can choose learning rate (lr) and batch size (bs) training or augmentation training. We first train on lr and bs, then use the best model for different augmentation tryouts.

```bash
# Standard parameters
code experiments/resnets_aug/configs/default.yaml
```

and for example for Augmentation strategies

```bash
# Specialized parameters for augmentation sweep
code experiments/resnets_aug/configs/sweep_resnet50_augment.yaml
```
then following the instruction in the notebook

```bash
# Train the model in a sweep
code experiments/resnets_aug/notebooks/augment-sweep.ipynb
```

### 5. Corrupted Dataset Evaluation

Systematic robustness evaluation of single models or ensembles across various corruption types and severity levels.

#### Setup
```bash
# Install requirements
pip install -r /experiments/base_ensemble/requirements.txt
```
or for a conda enviroment
```bash
conda env create -f environment.yaml
conda activate xai
```
Then unsing the notebook `eval_corruption.ipnyb` and following the instructions
```bash
code experiments/base_ensemble/notebooks/eval_corruption.ipynb
```

**Available Corruption Types:**
- `gaussian_noise`: Gaussian noise
- `pixelate`: Pixelation

**Severity Levels**: 1-5 (1 = mild, 5 = severe)

**Systematic Evaluation**: The notebook `experiments/base_ensemble/notebooks/eval_corruption.ipynb` automatically performs a complete evaluation across all corruption types and severity levels, generating:
- CSV file with all results
- Visualization of robustness curves
- Comparison tables between different ensembles

## Project Structure

```
xAI-proj-m-ws2526/
├── experiments/
│   ├── base_ensemble/                  
│   │   ├── src/
│   │   │   ├── data/                   # Datasets and diversity augmentations
│   │   │   ├── models/                 # Model architectures
│   │   │   └── training/               # Training loops
│   │   ├── scripts/                    # CLI scripts for training/inference
│   │   └── notebooks/                  
│   │       ├── ensemble_learning.ipynb # Core training and ensemble logic  
│   │       ├── train-ResNet18.ipynb    # First tryouts with resnet-models
│   │       └── eval_corruption.ipynb   # Corrupted data evaluation
│   ├── mixture_of_experts/             # MoE routing and analysis
│   ├── temp_calibration/               # Temperature scaling
│   └── resnets_aug/                    # ResNet augmentation experiments
│       ├── configs/                    # Training and sweep configurations
│       ├── scripts/                    # Training and inference scripts
│       ├── notebooks/                  # Jupyter notebooks for analysis
│       └── checkpoints/                # Saved model weights
├── datasets/
│   └── test/                           # Test datasets
├── environment.yaml                    # Conda environment definition
└── README.md                           # Project documentation
```
