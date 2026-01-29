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

CUT_LABELS = {
    K_TRACK_TYPE: "Track Type",
    K_PT_RANGE: "pT Range",
    K_ETA_RANGE: "Eta Range",
    K_TPC_NCLS: "TPC NCls",
    K_TPC_CROSSED_ROWS: "TPC Crossed Rows",
    K_TPC_CROSSED_ROWS_OVER_NCLS: "TPC Rows/NCls",
    K_TPC_CHI2_NDF: "TPC Chi2/NDF",
    K_TPC_REFIT: "TPC Refit",
    K_ITS_NCLS: "ITS NCls",
    K_ITS_CHI2_NDF: "ITS Chi2/NDF",
    K_ITS_REFIT: "ITS Refit",
    K_ITS_HITS: "ITS Hits",
    K_GOLDEN_CHI2: "Golden Chi2",
    K_DCAXY: "DCA xy",
    K_DCAZ: "DCA z"
}

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

def plot_histogram(pdf, data_list, labels, xlabel, title=None, bins=50, x_range=None, log_y=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    for data, label in zip(data_list, labels):
        ax.hist(data, bins=bins, alpha=0.5, label=label, range=x_range)
    
    if log_y:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.set_title(title)
    if len(labels) > 1 or labels[0]:
        ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

def plot_bars(pdf, data_list, labels, xlabel, title=None, log_y=True):
    fig, ax = plt.subplots(figsize=(8, 6))
    for data, label in zip(data_list, labels):
        data.value_counts().sort_index().plot.bar(ax=ax, alpha=0.5, label=label)

    if log_y:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.set_title(title)
    if len(labels) > 1 or labels[0]:
        ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

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
                if subf.is_dir() and ("with_shared_clusters" in subf.name or "without_shared_clusters" in subf.name):
                    folders_to_keep.append(f)
                    break
        folders = folders_to_keep
    else:
        folders = [input_folder]

    doubly_reco_info = {"n_total": [], "n_has_its": [], "n_has_tpc": [], "n_global": []}
    doubly_reco_more_five_tpc_tracks = []
    doubly_reco_more_two_its_tracks = []
    doubly_reco_less_five_tpc_tracks = []
    doubly_reco_less_two_its_tracks = []
    tpc_its_correlation = {"n_tpc": [], "n_its": []}
    df_aod_tot = []
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

        df_aod["fEta"] = -1. * np.log(np.tan(np.pi/4. - 0.5*np.atan(df_aod["fTgl"])))
        phi_raw = np.arcsin(df_aod["fSnp"]) + df_aod["fAlpha"]
        df_aod["fPhi"] = phi_raw % (2 * np.pi)
        df_aod["fTPCCrossedRows"] = df_aod['fTPCNClsFindable'] - df_aod["fTPCNClsFindableMinusCrossedRows"]
        df_aod["fTPCCrossedRowsOverFindable"] = df_aod["fTPCCrossedRows"] / df_aod['fTPCNClsFindable']
        df_aod_tot.append(df_aod)

        for (mc_id, tf), group in doubly_reco_groups:  # pylint: disable=unused-variable
            aod_tracks = df_aod.query("fIndexMcParticles == @mc_id and tf == @tf")

            doubly_reco_info["n_total"].append(len(aod_tracks))
            doubly_reco_info["n_has_its"].append(len(aod_tracks.query("fHasITS")))
            doubly_reco_info["n_has_tpc"].append(len(aod_tracks.query("fHasTPC")))
            doubly_reco_info["n_global"].append(len(aod_tracks.query("fIsGlobal")))

            if len(aod_tracks.query("fHasTPC")) > 5:
                doubly_reco_more_five_tpc_tracks.append(aod_tracks)
            if len(aod_tracks.query("fHasITS")) > 2:
                doubly_reco_more_two_its_tracks.append(aod_tracks)
            if len(aod_tracks.query("fHasTPC")) <= 5:
                doubly_reco_less_five_tpc_tracks.append(aod_tracks)
            if len(aod_tracks.query("fHasITS")) <= 2:
                doubly_reco_less_two_its_tracks.append(aod_tracks)

            tpc_its_correlation["n_tpc"].append(len(aod_tracks.query("fHasTPC")))
            tpc_its_correlation["n_its"].append(len(aod_tracks.query("fHasITS")))

    doubly_reco_info = pd.DataFrame(doubly_reco_info)
    doubly_reco_more_five_tpc_tracks = pd.concat(doubly_reco_more_five_tpc_tracks, ignore_index=True)
    doubly_reco_more_two_its_tracks = pd.concat(doubly_reco_more_two_its_tracks, ignore_index=True)
    doubly_reco_less_five_tpc_tracks = pd.concat(doubly_reco_less_five_tpc_tracks, ignore_index=True)
    doubly_reco_less_two_its_tracks = pd.concat(doubly_reco_less_two_its_tracks, ignore_index=True)
    df_aod_tot = pd.concat(df_aod_tot, ignore_index=True)

    # Produce figures
    if not os.path.exists(os.path.dirname(output_pdf)):
        os.makedirs(os.path.dirname(output_pdf))
    with PdfPages(output_pdf) as pdf:
        # Draw the number of AOD tracks matched to doubly-reco ITS tracks
        keys = ["n_total", "n_has_its", "n_has_tpc", "n_global"]
        titles = [
            "Doubly reconstructed ITS tracks matched to AOD tracks",
            "Doubly reconstructed ITS tracks matched to AOD tracks with ITS",
            "Doubly reconstructed ITS tracks matched to AOD tracks with TPC",
            "Doubly reconstructed ITS tracks matched to global AOD tracks"
        ]
        xlabels = [
            "Number of AOD tracks matched to doubly-reco ITS tracks",
            "Number of AOD tracks with ITS matched to doubly-reco ITS tracks",
            "Number of AOD tracks with TPC matched to doubly-reco ITS tracks",
            "Number of global AOD tracks matched to doubly-reco ITS tracks"
        ]
        for key, title, xlabel in zip(keys, titles, xlabels):
            plot_bars(
                pdf,
                [doubly_reco_info[key]],
                [""],
                xlabel,
                title=title,
                log_y=True
            )

        # Draw distributions of track parameters for doubly-reco ITS tracks with different number of AOD tracks
        vars_to_plot = ['fPt', 'fEta', 'fTPCNClsFindable', 'fTPCCrossedRows', 'fTPCCrossedRowsOverFindable', 'fTPCChi2NCl', 'fPhi']
        xaxis_title = [r"$p_\mathrm{T} (\mathrm{GeV}/c)$", r"$\eta$", "TPC findable clusters", "TPC crossed rows", "TPC crossed rows/findable clusters", r"$\chi^2$/NCl", r"$\phi$"]
        bins = [50, 50, 50, 50, 50, 50, 50]
        ranges = [(0, 10), (-1, 1), (0, 200), (0, 200), (0, 2), (0, 10), (0, 2*np.pi)]

        for var, xlabel, bin_num, x_range in zip(vars_to_plot, xaxis_title, bins, ranges):
            plot_histogram(
                pdf,
                [doubly_reco_more_five_tpc_tracks[var], doubly_reco_less_five_tpc_tracks[var]],
                ["> 5 TPC tracks", "<= 5 TPC tracks"],
                xlabel,
                bins=bin_num,
                x_range=x_range,
                log_y=True
            )

            plot_histogram(
                pdf,
                [doubly_reco_more_two_its_tracks[var], doubly_reco_less_two_its_tracks[var]],
                ["> 2 ITS tracks", "<= 2 ITS tracks"],
                xlabel,
                bins=bin_num,
                x_range=x_range,
                log_y=True
            )

        fig, ax = plt.subplots(figsize=(8,6))
        h, xedges, yedges = np.histogram2d(
            doubly_reco_more_five_tpc_tracks['fTPCCrossedRowsOverFindable'],
            doubly_reco_more_five_tpc_tracks['fTPCNClsFindable'],
            bins=(50, 50),
            range=((0, 3), (0, 160))
        )
        pcm = ax.pcolormesh(xedges, yedges, h.T, norm=plt.matplotlib.colors.LogNorm())
        fig.colorbar(pcm, ax=ax, label='Counts')
        ax.set_xlabel("Crossed rows / findable")
        ax.set_ylabel("Findable")
        ax.set_title("Correlation between number of AOD tracks with TPC and ITS")
        pdf.savefig(fig)
        plt.close(fig)

        failed_global = doubly_reco_more_five_tpc_tracks[~doubly_reco_more_five_tpc_tracks["fIsGlobal"]].query("fTPCCrossedRows > 70")
        rejection_counts = {}

        for bit, label in CUT_LABELS.items():
            # A cut is 'failed' if the bit is required by K_GLOBAL_TRACK 
            # but NOT present in fTrackCutFlag
            if bit & K_GLOBAL_TRACK:
                num_failed = (failed_global["fTrackCutFlag"] & bit == 0).sum()
                rejection_counts[label] = num_failed

        # Convert to Series for easy plotting
        df_rejection = pd.Series(rejection_counts).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        df_rejection.plot.barh(ax=ax, color='salmon', edgecolor='black')
        ax.set_xlabel("Number of Tracks Failing Cut")
        ax.set_title("Reasons for Failing Global Track Selection")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,6))
        h, xedges, yedges = np.histogram2d(
            tpc_its_correlation["n_tpc"],
            tpc_its_correlation["n_its"],
            bins=(range(0, 15), range(0, 10))
        )
        pcm = ax.pcolormesh(xedges, yedges, h.T, norm=plt.matplotlib.colors.LogNorm())
        fig.colorbar(pcm, ax=ax, label='Counts')
        ax.set_xlabel("Number of AOD tracks with TPC")
        ax.set_ylabel("Number of AOD tracks with ITS")
        ax.set_title("Correlation between number of AOD tracks with TPC and ITS")
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
