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
    tracks_df["fIsGlobalWoDca"] = (tracks_df['fTrackCutFlag'] & K_GLOBAL_TRACK_WO_DCA) == K_GLOBAL_TRACK_WO_DCA  # pylint: disable=line-too-long
    tracks_df["fHasTOF"] = (tracks_df["fTOFChi2"] > 0) & (tracks_df["fTOFExpMom"] > 0)

    tracks_df["fPt"] = np.abs(tracks_df["fSigned1Pt"]**-1)

    return tracks_df

def plot_histogram(pdf, data_list, labels, xlabel, title=None, bins=50, x_range=None, log_y=True):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """
    Plot histograms of the given data and save to PDF.

    Parameters:
    -----------
    pdf (PdfPages): PDF file to save the plots.
    data_list (list[pd.Series]): List of data series to plot.
    labels (list[str]): List of labels for the data series.
    xlabel (str): Label for the x-axis.
    title (str, optional): Title of the plot. Defaults to None.
    bins (int, optional): Number of bins for the histogram. Defaults to 50.
    x_range (tuple, optional): Range for the x-axis. Defaults to None.
    log_y (bool, optional): Whether to use logarithmic scale for y-axis. Defaults to True.
    """
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

def plot_bars(pdf, data_list, labels, xlabel, title=None, log_y=True):  # pylint: disable=too-many-arguments, too-many-positional-arguments
    """
    Plot bar charts of the given data and save to PDF.

    Parameters:
    -----------
    pdf (PdfPages): PDF file to save the plots.
    data_list (list[pd.Series]): List of data series to plot.
    labels (list[str]): List of labels for the data series.
    xlabel (str): Label for the x-axis.
    title (str, optional): Title of the plot. Defaults to None.
    log_y (bool, optional): Whether to use logarithmic scale for y-axis. Defaults to True.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (data, label) in enumerate(zip(data_list, labels)):
        data.value_counts().sort_index().plot.bar(ax=ax, color=f"C{i}", alpha=0.5, label=label)

    if log_y:
        ax.set_yscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Counts")
    ax.set_title(title)
    if len(labels) > 1 or labels[0]:
        ax.legend()
    pdf.savefig(fig)
    plt.close(fig)

def get_folders_to_process(input_folder):
    """
    Get the list of folders to process from the input folder.

    Parameters:
    -----------
    input_folder (str): Path to the input folder containing simulation files.

    Returns:
    -----------
    list[str]: List of paths to folders to process.
    """
    folders = [str(f.name) for f in Path(input_folder).iterdir() if f.is_dir()]
    batched = not ("with_shared_clusters" in folders or "without_shared_clusters" in folders)

    if batched:
        folders = [str(Path(input_folder) / f) for f in folders]
        # Remove folders that do not contain shared cluster info
        folders_to_keep = []
        for f in folders:
            for subf in (Path(input_folder) / f).iterdir():
                if subf.is_dir() and (
                    "with_shared_clusters" in subf.name or "without_shared_clusters" in subf.name
                ):
                    folders_to_keep.append(f)
                    break
        folders = folders_to_keep
    else:
        folders = [input_folder]
    return folders

def study_doubly_reco(
    input_folder: str,
    output_pdf: str
):  #pylint: disable=too-many-locals, too-many-statements
    """
    Study doubly reconstructed ITS tracks in AOD data.

    Parameters:
    -----------
    input_folder (str): Path to the input folder containing simulation files.
    output_pdf (str): Path to the output PDF file to save the plots.
    """

    folders = get_folders_to_process(input_folder)
    doubly_reco_info = {True: {}, False: {}} # Whether with shared clusters or not
    for with_sc in [True, False]:
        doubly_reco_info[with_sc] = {
            "n_total": [],
            "n_has_its": [],
            "n_has_tpc": [],
            "n_global": [],
            "n_global_wo_dca": [],
        }
    doubly_reco_more_five_tpc_tracks = {True: [], False: []}
    doubly_reco_more_two_its_tracks = {True: [], False: []}
    doubly_reco_less_five_tpc_tracks = {True: [], False: []}
    doubly_reco_less_two_its_tracks = {True: [], False: []}
    tpc_its_correlation = {True: {"n_tpc": [], "n_its": []}, False: {"n_tpc": [], "n_its": []}}
    df_aod_tot = {True: [], False: []}
    for with_sc in [True, False]:
        # Lists to store results from ALL folders for the current 'with_sc' setting
        all_info_dfs = []
        all_tracks_more_5_tpc = []
        all_tracks_less_5_tpc = []
        all_tracks_more_2_its = []
        all_tracks_less_2_its = []
        all_aod_tracks = []

        for folder in folders:
            input_its = os.path.join(
                folder, "outputs",
                f"{'with' if with_sc else 'without'}_shared_clusters",
                f"CheckTracksCA{'with' if with_sc else 'without'}_shared_clusters.parquet"
            )
            df_its = pd.read_parquet(input_its)
            df_aod = get_dfs_from_aod(folder, with_sc)

            df_aod["fEta"] = -1. * np.log(np.tan(np.pi/4. - 0.5*np.atan(df_aod["fTgl"])))
            phi_raw = np.arcsin(df_aod["fSnp"]) + df_aod["fAlpha"]
            df_aod["fPhi"] = phi_raw % (2 * np.pi)
            df_aod["fTPCCrossedRows"] = df_aod['fTPCNClsFindable'] - df_aod["fTPCNClsFindableMinusCrossedRows"]  # pylint: disable=line-too-long
            df_aod["fTPCCrossedRowsOverFindable"] = df_aod["fTPCCrossedRows"] / df_aod['fTPCNClsFindable']  # pylint: disable=line-too-long

            aod_summary = df_aod.groupby(["fIndexMcParticles", "tf"]).agg(
                n_total=("fPt", "count"),
                n_has_its=("fHasITS", "sum"),
                n_has_tpc=("fHasTPC", "sum"),
                n_global=("fIsGlobal", "sum"),
                n_global_wo_dca=("fIsGlobalWoDca", "sum")
            ).reset_index()

            dr_its_ids = df_its.query(
                "same_mc_track_id"
            )[["fIndexMcParticles", "tf"]].drop_duplicates()

            folder_info = pd.merge(
                dr_its_ids,
                aod_summary,
                on=["fIndexMcParticles", "tf"],
                how="inner"
            )
            all_info_dfs.append(folder_info)

            df_aod_dr = pd.merge(df_aod, folder_info, on=["fIndexMcParticles", "tf"])

            all_tracks_more_5_tpc.append(df_aod_dr.query("n_has_tpc > 5"))
            all_tracks_less_5_tpc.append(df_aod_dr.query("n_has_tpc <= 5"))
            all_tracks_more_2_its.append(df_aod_dr.query("n_has_its > 2"))
            all_tracks_less_2_its.append(df_aod_dr.query("n_has_its <= 2"))
            all_aod_tracks.append(df_aod)

        doubly_reco_info[with_sc] = pd.concat(all_info_dfs, ignore_index=True)

        doubly_reco_more_five_tpc_tracks[with_sc] = pd.concat(
            all_tracks_more_5_tpc,
            ignore_index=True
        )
        doubly_reco_less_five_tpc_tracks[with_sc] = pd.concat(
            all_tracks_less_5_tpc,
            ignore_index=True
        )
        doubly_reco_more_two_its_tracks[with_sc] = pd.concat(
            all_tracks_more_2_its,
            ignore_index=True
        )
        doubly_reco_less_two_its_tracks[with_sc] = pd.concat(
            all_tracks_less_2_its,
            ignore_index=True
        )

        df_aod_tot[with_sc] = pd.concat(all_aod_tracks, ignore_index=True)

        tpc_its_correlation[with_sc]["n_tpc"] = doubly_reco_info[with_sc]["n_has_tpc"]
        tpc_its_correlation[with_sc]["n_its"] = doubly_reco_info[with_sc]["n_has_its"]

    # Produce figures
    if not os.path.exists(os.path.dirname(output_pdf)):
        os.makedirs(os.path.dirname(output_pdf))
    with PdfPages(output_pdf) as pdf:
        # Draw the number of AOD tracks matched to doubly-reco ITS tracks
        keys = ["n_total", "n_has_its", "n_has_tpc", "n_global", "n_global_wo_dca"]
        titles = [
            "Doubly reconstructed ITS tracks matched to AOD tracks",
            "Doubly reconstructed ITS tracks matched to AOD tracks with ITS",
            "Doubly reconstructed ITS tracks matched to AOD tracks with TPC",
            "Doubly reconstructed ITS tracks matched to global AOD tracks",
            "Doubly reconstructed ITS tracks matched to global without DCA AOD tracks"
        ]
        xlabels = [
            "Number of AOD tracks matched to doubly-reco ITS tracks",
            "Number of AOD tracks with ITS matched to doubly-reco ITS tracks",
            "Number of AOD tracks with TPC matched to doubly-reco ITS tracks",
            "Number of global AOD tracks matched to doubly-reco ITS tracks",
            "Number of global without DCA AOD tracks matched to doubly-reco ITS tracks"
        ]
        for key, title, xlabel in zip(keys, titles, xlabels):
            plot_bars(
                pdf,
                [doubly_reco_info[True][key], doubly_reco_info[False][key]],
                ["With shared clusters", "Without shared clusters"],
                xlabel,
                title=title,
                log_y=True
            )

        # Draw distributions of track parameters for doubly-reco ITS tracks
        # with different number of AOD tracks
        vars_to_plot = [
            'fPt',
            'fEta',
            'fTPCNClsFindable',
            'fTPCCrossedRows',
            'fTPCCrossedRowsOverFindable',
            'fTPCChi2NCl',
            'fPhi'
        ]
        xaxis_title = [
            r"$p_\mathrm{T} (\mathrm{GeV}/c)$",
            r"$\eta$",
            "TPC findable clusters",
            "TPC crossed rows",
            "TPC crossed rows/findable clusters",
            r"$\chi^2$/NCl",
            r"$\phi$"
        ]
        bins = [50, 50, 50, 50, 50, 50, 50]
        ranges = [(0, 10), (-1, 1), (0, 200), (0, 200), (0, 2), (0, 10), (0, 2*np.pi)]

        for var, xlabel, bin_num, x_range in zip(vars_to_plot, xaxis_title, bins, ranges):
            plot_histogram(
                pdf,
                [
                    doubly_reco_less_five_tpc_tracks[True][var],
                    doubly_reco_less_five_tpc_tracks[False][var],
                    doubly_reco_more_five_tpc_tracks[True][var],
                    doubly_reco_more_five_tpc_tracks[False][var],
                ],
                [
                    "<= 5 TPC tracks, With shared clusters",
                    "<= 5 TPC tracks, Without shared clusters",
                    "> 5 TPC tracks, With shared clusters",
                    "> 5 TPC tracks, Without shared clusters",
                ],
                xlabel,
                bins=bin_num,
                x_range=x_range,
                log_y=True
            )

            plot_histogram(
                pdf,
                [
                    doubly_reco_less_two_its_tracks[True][var],
                    doubly_reco_less_two_its_tracks[False][var],
                    doubly_reco_more_two_its_tracks[True][var],
                    doubly_reco_more_two_its_tracks[False][var],
                ],
                [
                    "<= 2 ITS tracks, With shared clusters",
                    "<= 2 ITS tracks, Without shared clusters",
                    "> 2 ITS tracks, With shared clusters",
                    "> 2 ITS tracks, Without shared clusters",
                ],
                xlabel,
                bins=bin_num,
                x_range=x_range,
                log_y=True
            )

        failed_global = doubly_reco_more_five_tpc_tracks[True][
            ~doubly_reco_more_five_tpc_tracks[True]["fIsGlobal"]  # pylint: disable=invalid-sequence-index
        ].query("fTPCCrossedRows > 70")
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
        ax.set_title(
            "Failed global track selection for doubly-reco ITS tracks " \
            "with >5 TPC AOD tracks and >70 TPC crossed rows"
        )
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,6))
        h, xedges, yedges = np.histogram2d(
            tpc_its_correlation[True]["n_tpc"],
            tpc_its_correlation[True]["n_its"],
            bins=(range(0, 15), range(0, 10))
        )
        pcm = ax.pcolormesh(xedges, yedges, h.T, norm=plt.matplotlib.colors.LogNorm())
        fig.colorbar(pcm, ax=ax, label='Counts')
        ax.set_xlabel("Number of AOD tracks with TPC")
        ax.set_ylabel("Number of AOD tracks with ITS")
        ax.set_title(
            "Number of AOD tracks with TPC and ITS for doubly-reco ITS tracks"
            " (With shared clusters)"
        )
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,6))
        h, xedges, yedges = np.histogram2d(
            tpc_its_correlation[False]["n_tpc"],
            tpc_its_correlation[False]["n_its"],
            bins=(range(0, 15), range(0, 10))
        )
        pcm = ax.pcolormesh(xedges, yedges, h.T, norm=plt.matplotlib.colors.LogNorm())
        fig.colorbar(pcm, ax=ax, label='Counts')
        ax.set_xlabel("Number of AOD tracks with TPC")
        ax.set_ylabel("Number of AOD tracks with ITS")
        ax.set_title(
            "Number of AOD tracks with TPC and ITS for doubly-reco ITS tracks"
            " (Without shared clusters)"
        )
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
        "output",
        type=str,
        help="Output PDF file to save the plots.",
    )
    args = parser.parse_args()


    study_doubly_reco(args.sim_folder, args.output)
