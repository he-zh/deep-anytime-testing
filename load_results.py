import sys, os
import wandb
import pandas as pd
from hydra import initialize, compose
from omegaconf import OmegaConf

# Register resolver for group name consistency
try:
    OmegaConf.register_new_resolver("join", lambda sep, xs: sep.join(str(x) for x in xs))
except: pass

def main():
    # 1. Capture Hydra overrides from CLI
    overrides = [arg for arg in sys.argv[1:] if arg != "-m"]
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config", overrides=overrides)
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    config_tags = cfg_resolved['wandb']['tags'] + [
        cfg.wandb.task,
    ]
    api = wandb.Api()
    # 2. Find all runs in the generated group
    filters = {
        "$and": [
            {"group": cfg.wandb.group},
            {"tags": {"$all": config_tags+[cfg.data.type]}}, # Match all tags in the list

        ]
    }
    runs = api.runs("zhhe/seq-kci", filters=filters)
    
    
    expected_steps = cfg.train.seqs
    all_runs_data = []

    print(f"\n--- Validating {len(runs)} Runs for Group: {cfg.wandb.group} ---")
    
    for run in runs:
        seed = run.config.get("train", {}).get("seed", "unknown")
        
        # 1. Pre-download check using summary
        # Note: _step is 0-indexed, so 499 means 500 steps were logged
        actual_steps = run.summary.get("_step", 0) + 1 
        
        if actual_steps < expected_steps:
            print(f" ❌ Seed {seed}: CRASHED (Only {actual_steps}/{expected_steps} steps). Skipping...")
            continue
        
        # 2. If it passed validation, download the full history
        print(f" ✅ Seed {seed}: Complete ({actual_steps} steps). Downloading...")
        
        history = run.scan_history(keys=["wealth", "reject_null", "all_sample_nums"])
        df_run = pd.DataFrame(history)
        df_run["seed"] = seed
        all_runs_data.append(df_run)

    if not all_runs_data:
        print("No completed runs found to save.")
        return

    # 4. Merge all seeds into one DataFrame
    final_df = pd.concat(all_runs_data, ignore_index=True)
    
    # Standardize column name (W&B uses _step internally)
    if "_step" in final_df.columns:
        final_df = final_df.rename(columns={"_step": "step"})
    if "aggregated_test_e-value" in final_df.columns:
        final_df = final_df.rename(columns={"aggregated_test_e-value": "wealth"})

    # 5. Save the file with your naming convention
    tags_str = "_".join(config_tags)
    if not os.path.exists("saved_results"):
        os.makedirs("saved_results")
    if not os.path.exists(f"saved_results/{tags_str}"):
        os.makedirs(f"saved_results/{tags_str}")
    filename = f"saved_results/{tags_str}/{cfg.wandb.group}_{cfg.data.type}.csv"
    final_df.to_csv(filename, index=False)
    
    print(f"\n Success! Saved {len(final_df)} rows to {filename}")
    print(final_df.head())

if __name__ == "__main__":
    main()