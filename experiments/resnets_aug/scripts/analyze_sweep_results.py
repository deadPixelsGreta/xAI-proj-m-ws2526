"""
Example: Analyzing WandB Sweep Results Programmatically

This script demonstrates how to download and analyze sweep results
from your WandB project after the sweep completes.
"""

import wandb
import pandas as pd
import json
from typing import List, Dict, Any


def get_sweep_results(entity: str, project: str, sweep_id: str) -> pd.DataFrame:
    """
    Download sweep results and convert to DataFrame for analysis.
    
    Args:
        entity: Your WandB entity/username
        project: Project name (e.g., "imagenet-subset-randSeed")
        sweep_id: The sweep ID
        
    Returns:
        DataFrame with all runs and their metrics
    """
    api = wandb.Api()
    
    # Construct the full sweep path
    sweep_path = f"{entity}/{project}/{sweep_id}"
    print(f"Fetching results for sweep: {sweep_path}")
    
    # Get the sweep object
    sweep = api.sweep(sweep_path)
    
    # Collect all run data
    results = []
    
    for run in sweep.runs:
        result = {
            'run_name': run.name,
            'run_id': run.id,
            'state': run.state,
            'val_acc': run.summary.get('val_acc', None),
            'val_loss': run.summary.get('val_loss', None),
            'train_acc': run.summary.get('train_acc', None),
            'train_loss': run.summary.get('train_loss', None),
            'best_val_acc': run.summary.get('best_val_acc', None),
            'lr': run.config.get('lr', None),
            'batch_size': run.config.get('batch_size', None),
            'momentum': run.config.get('momentum', None),
            'weight_decay': run.config.get('weight_decay', None),
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    return df


def analyze_sweep_results(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze sweep results and provide insights.
    
    Args:
        df: DataFrame from get_sweep_results()
        
    Returns:
        Dictionary with analysis results
    """
    
    # Filter completed runs only
    completed = df[df['state'] == 'finished'].copy()
    
    if len(completed) == 0:
        print("No completed runs found yet!")
        return {}
    
    print(f"\n{'='*70}")
    print("SWEEP ANALYSIS")
    print(f"{'='*70}")
    print(f"Total runs: {len(df)}")
    print(f"Completed runs: {len(completed)}")
    print(f"Success rate: {len(completed)/len(df)*100:.1f}%\n")
    
    # Overall statistics
    print("OVERALL STATISTICS:")
    print(f"  Mean validation accuracy: {completed['val_acc'].mean():.4f}")
    print(f"  Std validation accuracy: {completed['val_acc'].std():.4f}")
    print(f"  Best validation accuracy: {completed['val_acc'].max():.4f}")
    print(f"  Worst validation accuracy: {completed['val_acc'].min():.4f}\n")
    
    # Best run
    best_run = completed.loc[completed['val_acc'].idxmax()]
    print("BEST HYPERPARAMETERS:")
    print(f"  Learning rate: {best_run['lr']:.6f}")
    print(f"  Batch size: {best_run['batch_size']}")
    print(f"  Momentum: {best_run['momentum']}")
    print(f"  Weight decay: {best_run['weight_decay']:.6f}")
    print(f"  Validation accuracy: {best_run['val_acc']:.4f}")
    print(f"  Training accuracy: {best_run['train_acc']:.4f}\n")
    
    # Learning rate analysis
    lr_groups = completed.groupby('lr')['val_acc'].agg(['mean', 'count', 'max', 'min'])
    print("LEARNING RATE ANALYSIS (sorted by mean validation accuracy):")
    print(lr_groups.sort_values('mean', ascending=False).to_string())
    print()
    
    # Batch size analysis
    if completed['batch_size'].nunique() > 1:
        bs_groups = completed.groupby('batch_size')['val_acc'].agg(['mean', 'count', 'max', 'min'])
        print("BATCH SIZE ANALYSIS (sorted by mean validation accuracy):")
        print(bs_groups.sort_values('mean', ascending=False).to_string())
        print()
    
    # Top 5 configurations
    print("TOP 5 CONFIGURATIONS:")
    top_5 = completed.nlargest(5, 'val_acc')[['lr', 'batch_size', 'momentum', 'val_acc', 'train_acc']]
    for idx, (_, row) in enumerate(top_5.iterrows(), 1):
        print(f"  {idx}. LR={row['lr']:.6f}, BS={int(row['batch_size'])}, Val Acc={row['val_acc']:.4f}")
    
    return {
        'best_run': best_run.to_dict(),
        'mean_val_acc': completed['val_acc'].mean(),
        'max_val_acc': completed['val_acc'].max(),
        'num_completed': len(completed),
    }


def export_results_to_csv(df: pd.DataFrame, output_path: str = "sweep_results.csv"):
    """Export results to CSV for further analysis."""
    
    # Select relevant columns
    export_df = df[[
        'run_name', 'state', 'val_acc', 'val_loss', 'train_acc', 'train_loss',
        'lr', 'batch_size', 'momentum', 'weight_decay'
    ]]
    
    export_df.to_csv(output_path, index=False)
    print(f"\n✓ Results exported to {output_path}")
    return export_df


def recommend_next_sweep(df: pd.DataFrame):
    """Suggest refined hyperparameter ranges for a follow-up sweep."""
    
    completed = df[df['state'] == 'finished'].copy()
    best_run = completed.loc[completed['val_acc'].idxmax()]
    
    best_lr = best_run['lr']
    
    # Find the range around the best LR
    lr_values = sorted(completed['lr'].unique())
    best_idx = lr_values.index(best_lr)
    
    print("\n" + "="*70)
    print("RECOMMENDED FOLLOW-UP SWEEP (to refine results)")
    print("="*70)
    
    # Suggest new range
    if best_idx == 0:
        suggested_min = best_lr / 10
        suggested_max = best_lr * 2
    elif best_idx == len(lr_values) - 1:
        suggested_min = best_lr / 2
        suggested_max = best_lr * 10
    else:
        suggested_min = lr_values[best_idx - 1]
        suggested_max = lr_values[best_idx + 1]
    
    print(f"\nCurrent best learning rate: {best_lr:.6f}")
    print(f"Suggested new range: {suggested_min:.6f} to {suggested_max:.6f}\n")
    
    print("Update sweep_learning_rate.yaml with:")
    print(f"""
  lr:
    distribution: log_uniform_values
    min: {suggested_min:.10f}
    max: {suggested_max:.10f}
""")


# Example usage
if __name__ == "__main__":
    # TODO: Replace with your actual values
    ENTITY = "your-entity"  # Replace with your WandB username/entity
    PROJECT = "imagenet-subset-randSeed"
    SWEEP_ID = "abc123def456"  # From the sweep initialization output
    
    print("SWEEP RESULT ANALYZER")
    print("=" * 70)
    print(f"Entity: {ENTITY}")
    print(f"Project: {PROJECT}")
    print(f"Sweep ID: {SWEEP_ID}\n")
    
    # Fetch results
    try:
        df = get_sweep_results(ENTITY, PROJECT, SWEEP_ID)
        
        # Analyze
        analysis = analyze_sweep_results(df)
        
        # Export
        export_results_to_csv(df)
        
        # Recommend next steps
        recommend_next_sweep(df)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure to:")
        print("1. Update ENTITY, PROJECT, and SWEEP_ID in this script")
        print("2. Run: wandb login")
        print("3. Wait for sweep to complete (at least some runs)")
