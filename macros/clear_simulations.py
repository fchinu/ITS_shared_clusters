"""Script to clear unused simulation data from the simulations directory."""
import argparse
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
            subfolders = [
                f for f in (input_folder / folder).iterdir()
                    if f.name in ("with_shared_clusters", "without_shared_clusters")
            ]
            tf_folders.extend([
                s for subfolder in subfolders for s in subfolder.iterdir()
                    if s.is_dir() and "tf" in s.name
            ])
    else:
        subfolders = [
            f for f in input_folder.iterdir()
                if f.name in ("with_shared_clusters", "without_shared_clusters")
        ]
        tf_folders = [
            s for subfolder in subfolders for s in subfolder.iterdir()
                if s.is_dir() and "tf" in s.name
        ]

    return tf_folders

def clear_simulations(simulations_dir: str):
    """
    Main function to clear all simulation data.
    
    Parameters:
    -----------
    simulations_dir (str): The simulations directory.
    """

    tf_folders = get_all_tf_folders(simulations_dir)
    print(tf_folders)

    for tf_folder in tf_folders:
        files_to_remove = [
            "tpc_driftime_digits_lane*.root",
            "tpc-native-clusters-part*.root",
            "sgn_HitsTPC.root",
            "sgn_HitsTRD.root",
            "trddigits.root",
            "fdddigits.root",
            "mftdigits.root",
            "sgn_HitsMFT.root",
            "sgn_HitsEMC.root",
            "tpc_polya.root",
            "mfttracks.root",
            "sgn_HitsPHS.root",
            "mchreco_1.log",
            "mchclusters.root",
            "trdmatches_tpc.root",
            "trdreco2_1.log",
            "sgn_HitsHMP.root",
            "sgn_HitsMID.root",
            "trdmatches_itstpc.root",
            "mchdigits.root",
            "trddigi_1.log",
            "o2reco_fdd.root",
            "sgn_HitsMCH.root",
            "trdtracklets.root",
            "trdcalibratedtracklets.root",
            "mftclusters.root"
        ]

        for pattern in files_to_remove:
            for file_path in tf_folder.glob(pattern):
                # print(f"Removing file: {file_path}")
                file_path.unlink()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clear all simulation data from the simulations directory."
    )
    parser.add_argument(
        "--trial-dir",
        type=str,
        help="Path to the simulations directory",
    )
    args = parser.parse_args()

    clear_simulations(args.trial_dir)
