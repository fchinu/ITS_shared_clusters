import argparse
import os
from pathlib import Path

def get_all_tf_folders(input_folder: str) -> list[str]:
    """
    Get all timeframe folders from the input folder.
    
    Parameters:
    -----------
    input_folder (str): Path to the input folder containing timeframe folders.
    
    Returns:
    -----------
    list[str]: List of paths to timeframe folders.
    """
    input_folder = Path(input_folder)

    # Check if there are multiple batches in simulation
    folders = [str(f.name) for f in input_folder.iterdir() if f.is_dir()]
    batched = not ("with_shared_clusters" in folders or "without_shared_clusters" in folders)
    tf_folders = []


    if batched:
        for folder in folders:
            subfolders = [f for f in (input_folder / folder).iterdir() if f.name in ("with_shared_clusters", "without_shared_clusters")]
            tf_folders.extend([s for subfolder in subfolders for s in subfolder.iterdir() if s.is_dir() and "tf" in s.name])
    else:
        subfolders = [f for f in input_folder.iterdir() if f.name in ("with_shared_clusters", "without_shared_clusters")]
        tf_folders = [s for subfolder in subfolders for s in subfolder.iterdir() if s.is_dir() and "tf" in s.name]

    return tf_folders


def run_task(trial_dir):
    # Load configuration
    config_path = os.path.join(Path(__file__).parent.parent, "configs", "configuration_track_sel.json")
    output_director_path = os.path.join(Path(__file__).parent.parent, "configs", "OutputDirector.json")

    trial_dir = Path(trial_dir)

    pwd = os.getcwd()

    tf_folders = get_all_tf_folders(trial_dir)

    for tf_folder in tf_folders:
        os.chdir(tf_folder)
        print(f"in folder: {os.getcwd()}")
        if (tf_folder / "AO2D.root").exists() and not (tf_folder / "AO2D_old.root").exists():
            os.rename(
                tf_folder / "AO2D.root",
                tf_folder / "AO2D_old.root"
            )

        bash_path = tf_folder / "extend_aod_with_tracksel.sh"
        if bash_path.exists():
            os.remove(bash_path)
        
        with open(bash_path, "w") as bash_file:
            bash_file.write(
                f"""
#!/bin/bash

LOGFILE="log_track_selection.txt"
OPTION="-b --configuration json://{config_path}"

# Tree creator
o2-analysis-propagationservice $OPTION |
o2-analysis-event-selection-service $OPTION |
o2-analysis-trackselection $OPTION --aod-file AO2D_old.root --aod-writer-json {output_director_path} --shm-segment-size 3000000000 --aod-parent-access-level 1  > "$LOGFILE" 2>&1

# report status
rc=$?
if [ $rc -eq 0 ]; then
  echo "No problems!"
else
  echo "Error: Exit code $rc"
  echo "Check the log file $LOGFILE"
  exit $rc
fi
"""
            )

        os.system(f"bash {bash_path}")

        os.system(f"hadd -f {tf_folder / 'AO2D_with_tracksel.root'} {tf_folder / 'AO2D_old.root'} {tf_folder / 'AO2D.root'}")
        os.remove(tf_folder / "AO2D.root")
        os.remove(tf_folder / "AnalysisResults.root")
        os.remove(bash_path)
    os.chdir(pwd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extend AOD with track selection information."
    )
    parser.add_argument(
        "--trial-dir",
        required=True,
        help="Path to the trial directory",
    )
    args = parser.parse_args()

    run_task(args.trial_dir)
