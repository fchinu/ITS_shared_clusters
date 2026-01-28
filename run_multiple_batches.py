import subprocess
import argparse
import os
import sys

def run_simulations(base_name, iterations, start_splitid):
    """
    Runs the Makefile pipeline multiple times using environment-aware defaults.
    """
    print(f"Base Name: {base_name}")
    print(f"Iterations: {iterations}")
    print(f"Start SplitID: {start_splitid}")
    print("-" * 40)

    for i in range(1, iterations + 1):
        current_name = f"{base_name}/{i}"
        current_splitid = start_splitid + (i - 1)
        
        print(f"[Run {i}/{iterations}]: NAME={current_name} | SPLITID={current_splitid}")

        cmd = [
            "make",
            f"NAME={current_name}",
            f"SPLITID={current_splitid}",
            "all"
        ]

        try:
            # We use check=True to catch Makefile errors immediately
            subprocess.run(cmd, check=True)
            print(f"Success: {current_name}\n")
        except subprocess.CalledProcessError:
            print(f"Error: Makefile failed at {current_name}. Aborting.")
            sys.exit(1)
        except KeyboardInterrupt:
            print("\nProcess interrupted by user. Exiting.")
            sys.exit(0)

if __name__ == "__main__":
    env_name = os.getenv("NAME", "shared_clusters_test")
    env_splitid = int(os.getenv("SPLITID", 42))

    parser = argparse.ArgumentParser(description="Batch run Makefile simulations using Env Vars.")
    parser.add_argument("--name", type=str, default=env_name, 
                        help=f"Base name (Current Env: {env_name})")
    parser.add_argument("--n", type=int, default=5, 
                        help="Number of iterations to run")
    parser.add_argument("--splitid", type=int, default=env_splitid, 
                        help=f"Starting SplitID (Current Env: {env_splitid})")

    args = parser.parse_args()

    run_simulations(args.name, args.n, args.splitid)