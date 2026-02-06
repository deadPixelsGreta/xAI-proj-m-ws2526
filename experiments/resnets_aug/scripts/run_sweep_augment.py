"""Script to run augmentation sweeps from command line."""

import os
import sys
import argparse
import yaml
from pathlib import Path

import wandb

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_sweep_config(config_path: str) -> dict:
    """Load sweep configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_sweep(
    model_name: str,
    config_path: str = None,
    count: int = 30,
    project: str = None,
):
    """
    Run augmentation sweep for a specific model.
    
    Args:
        model_name: resnet18, resnet34, or resnet50
        config_path: Path to sweep config (optional)
        count: Number of sweep runs
        project: WandB project name (optional)
    """
    # Default config path
    if config_path is None:
        config_path = f"configs/sweep_{model_name}_augment.yaml"
    
    # Default project name
    if project is None:
        project = f"imagenet-augment-{model_name}"
    
    print(f"\n{'='*70}")
    print(f"Starting Augmentation Sweep for {model_name.upper()}")
    print('='*70)
    print(f"Config: {config_path}")
    print(f"Project: {project}")
    print(f"Runs: {count}")
    print('='*70)
    
    # Load configuration
    sweep_config = load_sweep_config(config_path)
    
    # Create sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
    )
    
    print(f"\n✓ Sweep created with ID: {sweep_id}")
    print(f"View at: https://wandb.ai/your-entity/{project}/sweeps/{sweep_id}\n")
    
    # Import training function
    from src.training.train_augmentation_sweep import train
    
    # Run sweep agent
    print(f"Starting sweep agent with {count} runs...\n")
    wandb.agent(sweep_id, function=train, count=count)
    
    print(f"\n✓ Sweep completed for {model_name}")


def run_all_sweeps(count: int = 30):
    """Run sweeps for all three ResNet models."""
    models = ['resnet18', 'resnet34', 'resnet50']
    
    print("\n" + "="*70)
    print("Running Augmentation Sweeps for All Models")
    print("="*70)
    
    for model in models:
        try:
            run_sweep(model, count=count)
        except Exception as e:
            print(f"\n✗ Error running sweep for {model}: {e}")
            continue
    
    print("\n" + "="*70)
    print("All Sweeps Completed")
    print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run augmentation parameter sweeps for ResNets"
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['resnet18', 'resnet34', 'resnet50', 'all'],
        default='all',
        help='Model to run sweep for (default: all)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to sweep config file (optional)'
    )
    
    parser.add_argument(
        '--count',
        type=int,
        default=30,
        help='Number of sweep runs (default: 30)'
    )
    
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help='WandB project name (optional)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default=None,
        help='Data directory path (optional, uses DATA_DIR env var by default)'
    )
    
    args = parser.parse_args()
    
    # Set data directory if provided
    if args.data_dir:
        os.environ['DATA_DIR'] = args.data_dir
        print(f"Data directory set to: {args.data_dir}")
    
    # Run sweeps
    if args.model == 'all':
        run_all_sweeps(count=args.count)
    else:
        run_sweep(
            model_name=args.model,
            config_path=args.config,
            count=args.count,
            project=args.project,
        )


if __name__ == "__main__":
    main()