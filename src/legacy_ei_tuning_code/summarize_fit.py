import numpy as np
import argparse

from utils import *

def main(result_path):
    pkl_files = [f for f in os.listdir(result_path) if f.endswith('.pkl')]
    
    if not pkl_files:
        print(f"No .pkl files found in {result_path}")
        return

    print(f"Found {len(pkl_files)} .pkl files")
    ps = []
    ps_best_seen = []
    total_loss_rmse = []
    nan_count = 0
    for file in pkl_files:
        (p, p_best_seen, total_loss_rmse, total_loss_fcr) = load_results(os.path.join(result_path, file))
        print(f"final loss of {file}: rmse {total_loss_rmse[-1]:.3f} fc r {total_loss_fcr[-1]:.3f}")
        if not np.isfinite(total_loss_rmse[-1]):
            nan_count = nan_count + 1
        # ps.append(p)
        # ps_best_seen.append(p_best_seen)
    print(f"found {nan_count} nans out of {len(pkl_files)} .pkl files")

    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process subject data.")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result directory")
    
    args = parser.parse_args() 
    
    main(args.result_path)