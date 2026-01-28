"""
Study doubly reconstructed ITS tracks in AOD.
"""

import argparse
from pathlib import Path
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot

K_TRACK_TYPE = 1 << 0
K_PT_RANGE = 1 << 1
K_ETA_RANGE = 1 << 2
K_TPC_NCLS = 1 << 3
K_TPC_CROSSED_ROWS = 1 << 4
K_TPC_CROSSED_ROWS_OVER_NCLS = 1 << 5
K_TPC_CHI2_NDF = 1 << 6
K_TPC_REFIT = 1 << 7
K_ITS_NCLS = 1 << 8
K_ITS_CHI2_NDF = 1 << 9
K_ITS_REFIT = 1 << 10
K_ITS_HITS = 1 << 11
K_GOLDEN_CHI2 = 1 << 12
K_DCAXY = 1 << 13
K_DCAZ = 1 << 14

MASTER_MASK = (
    K_TRACK_TYPE |
    K_ITS_NCLS |
    K_ITS_CHI2_NDF |
    K_ITS_REFIT |
    K_ITS_HITS |
    K_TPC_NCLS |
    K_TPC_CROSSED_ROWS |
    K_TPC_CROSSED_ROWS_OVER_NCLS |
    K_TPC_CHI2_NDF |
    K_TPC_REFIT |
    K_GOLDEN_CHI2 |
    K_DCAXY |
    K_DCAZ |
    K_PT_RANGE |
    K_ETA_RANGE
)

masks = {
    "Track Type": K_TRACK_TYPE,
    "Pt Range": K_PT_RANGE,
    "Eta Range": K_ETA_RANGE,
    "TPC NCls": K_TPC_NCLS,
    "TPC Crossed Rows": K_TPC_CROSSED_ROWS,
    "TPC Rows/NCls": K_TPC_CROSSED_ROWS_OVER_NCLS,
    "TPC Chi2/NDF": K_TPC_CHI2_NDF,
    "TPC Refit": K_TPC_REFIT,
    "ITS NCls": K_ITS_NCLS,
    "ITS Chi2/NDF": K_ITS_CHI2_NDF,
    "ITS Refit": K_ITS_REFIT,
    "ITS Hits": K_ITS_HITS,
    "Golden Chi2": K_GOLDEN_CHI2,
    "DCAxy": K_DCAXY,
    "DCAz": K_DCAZ
}

K_QUALITY_TRACKS_ITS = K_TRACK_TYPE | K_ITS_NCLS | K_ITS_CHI2_NDF | K_ITS_REFIT | K_ITS_HITS
K_QUALITY_TRACKS_TPC = K_TRACK_TYPE | K_TPC_NCLS | K_TPC_CROSSED_ROWS | K_TPC_CROSSED_ROWS_OVER_NCLS | K_TPC_CHI2_NDF | K_TPC_REFIT  #pylint: disable=line-too-long
K_QUALITY_TRACKS = K_TRACK_TYPE | K_QUALITY_TRACKS_ITS | K_QUALITY_TRACKS_TPC
K_QUALITY_TRACKS_WO_TPC_CLUSTER = K_QUALITY_TRACKS_ITS | K_TPC_CHI2_NDF | K_TPC_REFIT
K_PRIMARY_TRACKS = K_GOLDEN_CHI2 | K_DCAXY | K_DCAZ
K_IN_ACCEPTANCE_TRACKS = K_PT_RANGE | K_ETA_RANGE
K_GLOBAL_TRACK = K_QUALITY_TRACKS | K_PRIMARY_TRACKS | K_IN_ACCEPTANCE_TRACKS
K_GLOBAL_TRACK_WO_TPC_CLUSTER = K_QUALITY_TRACKS_WO_TPC_CLUSTER | K_PRIMARY_TRACKS | K_IN_ACCEPTANCE_TRACKS  #pylint: disable=line-too-long
K_GLOBAL_TRACK_WO_PT_ETA = K_QUALITY_TRACKS | K_PRIMARY_TRACKS
K_GLOBAL_TRACK_WO_DCA = K_QUALITY_TRACKS | K_IN_ACCEPTANCE_TRACKS
K_GLOBAL_TRACK_WO_DCA_XY = K_QUALITY_TRACKS | K_IN_ACCEPTANCE_TRACKS | K_DCAZ
K_GLOBAL_TRACK_WO_DCA_TPC_CLUSTER = K_QUALITY_TRACKS_WO_TPC_CLUSTER | K_IN_ACCEPTANCE_TRACKS

def get_all_tf_folders(input_folder: str, with_sc: bool) -> list[str]:
    """
    Get all timeframe folders from the input folder.

    Parameters:
    -----------
    input_folder (str): Path to the input folder containing timeframe folders.
    with_sc (bool): Whether to look for 'with_shared_clusters' or 'without_shared_clusters' folders.

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
                if f.name == ("with_shared_clusters" if with_sc else "without_shared_clusters")
            ]
            tf_folders.extend([
                s for subfolder in subfolders
                for s in subfolder.iterdir()
                if s.is_dir() and "tf" in s.name
            ])
    else:
        subfolders = [
            f for f in input_folder.iterdir()
            if f.name == ("with_shared_clusters" if with_sc else "without_shared_clusters")
        ]
        tf_folders = [
            s for subfolder in subfolders
            for s in subfolder.iterdir()
            if s.is_dir() and "tf" in s.name
        ]

    return tf_folders

def get_dfs_from_aod(input_aod_folder: str, with_sc: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read AOD ROOT files from the input folder and return pandas DataFrames with track and MC info.

    Args:
        input_aod_folder (str): Name of the folder with the timeframe output

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: tuple containing the reconstructed and MC information
    """
    # Look for all ROOT files in the input folder
    folders = get_all_tf_folders(input_aod_folder, with_sc)
    tfs = [int(f.name.split("tf")[-1]) for f in folders]

    df_trackextra, df_trackiu, df_trackselection, df_tracklabels = [], [], [], []
    for folder, tf in zip(folders, tfs):
        with uproot.open(os.path.join(folder, "AO2D_with_tracksel.root")) as f:
            for key, tree in f.items():
                if "O2trackextra_002" in key:
                    df_trackextra.append(tree.arrays(library="pd"))
                    df_trackextra[-1]["tf"] = tf
                elif "O2track_iu" in key:
                    df_trackiu.append(tree.arrays(library="pd"))
                elif "O2mctracklabel" in key:
                    df_tracklabels.append(tree.arrays(library="pd"))
                elif "O2trackselection" in key:
                    df_trackselection.append(tree.arrays(library="pd"))

    df_trackextra = pd.concat(df_trackextra, ignore_index=True)
    df_trackiu = pd.concat(df_trackiu, ignore_index=True)
    df_trackselection = pd.concat(df_trackselection, ignore_index=True)
    df_tracklabels = pd.concat(df_tracklabels, ignore_index=True)
    tracks_df = pd.concat(
        [df_trackextra, df_tracklabels, df_trackiu, df_trackselection],
        axis=1, join="inner"
    )
    tracks_df["fHasITS"] = tracks_df["fITSClusterSizes"] > 0
    tracks_df["fHasTPC"] = tracks_df["fTPCNClsFindable"] > 0
    tracks_df["fIsGlobal"] = (tracks_df['fTrackCutFlag'] & K_GLOBAL_TRACK) == K_GLOBAL_TRACK
    tracks_df["fHasTOF"] = (tracks_df["fTOFChi2"] > 0) & (tracks_df["fTOFExpMom"] > 0)

    tracks_df["fPt"] = np.abs(tracks_df["fSigned1Pt"]**-1)

    return tracks_df

def study_doubly_reco(
    input_folder: str,
    with_sc: bool,
    output_pdf: str
):  #pylint: disable=too-many-locals, too-many-statements
    """
    Study doubly reconstructed ITS tracks in AOD data.

    Parameters:
    -----------
    input_folder (str): Path to the input folder containing simulation files.
    with_sc (bool): Whether shared clusters were used in the simulation.
    output_pdf (str): Path to the output PDF file to save the plots.
    """

    folders = [str(f.name) for f in Path(input_folder).iterdir() if f.is_dir()]
    batched = not ("with_shared_clusters" in folders or "without_shared_clusters" in folders)

    if batched:
        folders = [str(Path(input_folder) / f) for f in folders]
        # Remove folders that do not contain shared cluster info
        folders_to_keep = []
        for f in folders:
            for subf in (Path(input_folder) / f).iterdir():
                if "with_shared_clusters" in subf.name or "without_shared_clusters" in subf.name:
                    folders_to_keep.append(f)
                    break
        folders = folders_to_keep
    else:
        folders = [input_folder]

    doubly_reco_info = {"n_total": [], "n_has_its": [], "n_has_tpc": [], "n_global": []}
    for folder in folders:
        # Load ITS track data
        input_its = os.path.join(
            folder,
            "outputs",
            f"{'with' if with_sc else 'without'}_shared_clusters",
            f"CheckTracksCA{'with' if with_sc else 'without'}_shared_clusters.parquet"
        )
        df_its = pd.read_parquet(input_its)

        # Load AOD data
        df_aod = get_dfs_from_aod(folder, with_sc)
        df_its_doubly_reco = df_its.query("same_mc_track_id")
        doubly_reco_groups = df_its_doubly_reco.groupby(["fIndexMcParticles", "tf"])


        for (mc_id, tf), group in doubly_reco_groups:  # pylint: disable=unused-variable
            aod_tracks = df_aod.query("fIndexMcParticles == @mc_id and tf == @tf")

            doubly_reco_info["n_total"].append(len(aod_tracks))
            doubly_reco_info["n_has_its"].append(len(aod_tracks.query("fHasITS")))
            doubly_reco_info["n_has_tpc"].append(len(aod_tracks.query("fHasTPC")))
            doubly_reco_info["n_global"].append(len(aod_tracks.query("fIsGlobal")))

    doubly_reco_info = pd.DataFrame(doubly_reco_info)

    # Produce figures
    if not os.path.exists(os.path.dirname(output_pdf)):
        os.makedirs(os.path.dirname(output_pdf))
    with PdfPages(output_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(8,6))
        doubly_reco_info['n_total'].value_counts().sort_index().plot.bar(ax=ax)
        ax.set_yscale('log')
        ax.set_xlabel("Number of AOD tracks matched to doubly-reco ITS tracks")
        ax.set_ylabel("Counts")
        ax.set_title("Doubly reconstructed ITS tracks matched to AOD tracks")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,6))
        doubly_reco_info['n_has_its'].value_counts().sort_index().plot.bar(ax=ax)
        ax.set_yscale('log')
        ax.set_xlabel("Number of AOD tracks with ITS matched to doubly-reco ITS tracks")
        ax.set_ylabel("Counts")
        ax.set_title("Doubly reconstructed ITS tracks matched to AOD tracks with ITS")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,6))
        doubly_reco_info['n_has_tpc'].value_counts().sort_index().plot.bar(ax=ax)
        ax.set_yscale('log')
        ax.set_xlabel("Number of AOD tracks with TPC matched to doubly-reco ITS tracks")
        ax.set_ylabel("Counts")
        ax.set_title("Doubly reconstructed ITS tracks matched to AOD tracks with TPC")
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,6))
        doubly_reco_info['n_global'].value_counts().sort_index().plot.bar(ax=ax)
        ax.set_yscale('log')
        ax.set_xlabel("Number of global AOD tracks matched to doubly-reco ITS tracks")
        ax.set_ylabel("Counts")
        ax.set_title("Doubly reconstructed ITS tracks matched to global AOD tracks")
        pdf.savefig(fig)
        plt.close(fig)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Study doubly reconstructed tracks in ITS AOD data."
    )
    parser.add_argument(
        "sim_folder",
        type=str,
        help="Input folder containing simulation files.",
    )
    parser.add_argument(
        "--with_sc", "-w",
        action="store_true",
        help="If shared clusters were used in the simulation.",
    )
    parser.add_argument(
        "--without_sc", "-wo",
        action="store_true",
        help="If shared clusters were not used in the simulation.",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output PDF file to save the plots.",
    )
    args = parser.parse_args()

    if (args.with_sc and args.without_sc) or (not args.with_sc and not args.without_sc):
        raise ValueError("Please select either --with_sc or --without_sc, not both.")

    study_doubly_reco(args.sim_folder, args.with_sc, args.output)
