#!/usr/bin/env python3
"""
Utility script to manage WandB Sweeps for hyperparameter tuning.
Makes it easy to initialize and run sweeps from Python.
"""

import subprocess
import sys
from pathlib import Path
from typing import Optional


def initialize_sweep(config_path: str, project: str = "imagenet-subset-randSeed") -> Optional[str]:
    """
    Initialize a new sweep with the given configuration.
    
    Args:
        config_path: Path to sweep YAML configuration
        project: WandB project name
        
    Returns:
        sweep_id: The ID of the created sweep (e.g., "entity/project/sweep-id")
    """
    print("=" * 70)
    print("INITIALIZING WANDB SWEEP")
    print("=" * 70)
    print(f"Config: {config_path}")
    print(f"Project: {project}")
    
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}")
        return None
    
    cmd = ["wandb", "sweep", config_path]
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(result.stdout)
        
        if result.returncode != 0:
            print("ERROR:", result.stderr)
            return None
        
        # Extract sweep ID from output
        for line in result.stdout.split('\n'):
            if 'Sweep with ID' in line or 'sweep ID' in line.lower():
                # Try to extract the sweep ID
                parts = line.split('/')
                if len(parts) >= 3:
                    sweep_id = '/'.join(parts[-3:])
                    print(f"\n✓ Sweep ID: {sweep_id}")
                    return sweep_id
        
        return None
        
    except Exception as e:
        print(f"ERROR: {e}")
        return None


def run_agent(sweep_id: str, num_runs: Optional[int] = None):
    """
    Run a sweep agent.
    
    Args:
        sweep_id: The sweep ID (from initialize_sweep)
        num_runs: Optional number of runs before stopping (None = infinite)
    """
    print("=" * 70)
    print("STARTING WANDB SWEEP AGENT")
    print("=" * 70)
    print(f"Sweep ID: {sweep_id}")
    if num_runs:
        print(f"Will stop after {num_runs} runs")
    print("\nPress Ctrl+C to stop the agent\n")
    
    cmd = ["wandb", "agent"]
    if num_runs:
        cmd.extend(["--count", str(num_runs)])
    cmd.append(sweep_id)
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n\n✓ Agent stopped by user")
    except Exception as e:
        print(f"ERROR: {e}")


def print_usage():
    """Print usage instructions."""
    print("""
WANDB Sweep Management Utility
==============================

Usage:
    python experiments/scripts/wandb_sweep_manager.py init <config_path>
        Initialize a new sweep
        
    python experiments/scripts/wandb_sweep_manager.py run <sweep_id> [num_runs]
        Run a sweep agent
        
    python experiments/scripts/wandb_sweep_manager.py help
        Show this message

Examples:
    # Initialize learning rate sweep
    python experiments/scripts/wandb_sweep_manager.py init experiments/configs/sweep_learning_rate.yaml
    
    # Run agent for 10 runs
    python experiments/scripts/wandb_sweep_manager.py run entity/project/sweep-xyz 10
    
    # Run agent continuously
    python experiments/scripts/wandb_sweep_manager.py run entity/project/sweep-xyz

Requirements:
    - WandB installed: pip install wandb
    - Logged in: wandb login
    - Valid config file
    """)


def main():
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == "init" and len(sys.argv) >= 3:
        config_path = sys.argv[2]
        sweep_id = initialize_sweep(config_path)
        if sweep_id:
            print(f"\n{'='*70}")
            print("NEXT STEPS:")
            print(f"{'='*70}")
            print(f"Run agents with:")
            print(f"  wandb agent {sweep_id}")
            print(f"\nOr use this script:")
            print(f"  python experiments/scripts/wandb_sweep_manager.py run {sweep_id}")
            
    elif command == "run" and len(sys.argv) >= 3:
        sweep_id = sys.argv[2]
        num_runs = int(sys.argv[3]) if len(sys.argv) > 3 else None
        run_agent(sweep_id, num_runs)
        
    elif command in ["help", "-h", "--help"]:
        print_usage()
        
    else:
        print(f"Unknown command: {command}\n")
        print_usage()


if __name__ == "__main__":
    main()
