# BRACE

# BRACE: [Your Paper Title/Project Name Here] - Supplementary Material

This repository contains the supplementary code for the NeurIPS paper "[Your Paper Title Here]".
The code implements the BRACE (Bayesian Reasoning and Action C)ollaboration Engine, including:
-   Expert policy training using PPO (Proximal Policy Optimization).
-   Bayesian goal inference filter pre-training.
-   BRACE assistance policy training with curriculum learning.

## File Structure

The project is organized as follows:

BRACE_supplementary/
├── brace/                     # Main Python package for BRACE
│   ├── init.py
│   ├── envs/                  # Custom environment definitions (e.g., for Cursor, Reacher)
│   │   └── ...
│   ├── expert.py              # PPO expert training script
│   ├── inference.py           # Bayesian goal filter pre-training script
│   ├── policy.py              # BRACE assistance policy training script (with curriculum)
│   └── utils.py               # Utility functions (seeds, GAE, action blending, etc.)
├── configs/                   # Configuration files (e.g., for 'train_brace.py')
│   └── reacher_smoke.yaml
├── scripts/                   # Main executable scripts
│   └── train_brace.py         # Simplified BRACE training loop (e.g., for Reacher)
├── notebooks/                 # (Optional) Jupyter notebooks for exploration/analysis
├── prototypes/                # (Optional) Experimental code
├── tests/                     # (Optional) More formal tests
├── check.py                   # Basic script to check model instantiation
├── smoke_test.sh              # Script to run a quick smoke test for 'train_brace.py'
├── LICENSE                    # License for the code
└── README.md                  # This file




## Requirements

* Python 3.9+
* PyTorch 1.12
* Gymnasium 0.28
* NumPy
* PyYAML

## Setup Instructions


1.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv_brace
    source venv_brace/bin/activate  # On Windows: venv_brace\Scripts\activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r "libraries listed above"
    ```

## Running the Code

### 1. Training Expert Policies

The `brace/expert.py` script is used to train expert policies for different scenarios.
These experts can then be used by the BRACE assistance policy.

```bash
python brace/expert.py --scenario [cursor|reacher] --steps [total_steps] --save_ckpt experts/[scenario]_expert.pt --save_traj experts/[scenario]_demos.npz --device [cpu|cuda]

For example:

python brace/expert.py --scenario cursor --steps 100000 --save_ckpt experts/cursor_actor.pt --save_traj experts/cursor_demos.npz

python brace/expert.py --scenario reacher --steps 200000 --save_ckpt experts/reacher_actor.pt --save_traj experts/reacher_demos.npz

The script can use custom environments from brace.envs or fall back to default Gym environments (CartPole for cursor, Pendulum for reacher) if brace.envs is not fully set up. For paper results, ensure the correct brace.envs are used.
