"""Refactored script to draw shared clusters analysis results"""

import argparse
from collections import Counter
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors


class TrackType(Enum):
    """Enumeration for track types"""
    FAKE = "fake"
    GOOD = "good"
    DOUBLY_RECO = "doubly_reco"
    DOUBLY_RECO_ALSO_WITHOUT_SHARED = "doubly_reco_also_without_shared"
    DOUBLY_RECO_ONLY_WITH_SHARED = "doubly_reco_only_with_shared"


@dataclass
class TrackTypeConfig:
    """Configuration for track type visualization"""
    color: str
    label_template: str


@dataclass
class PlotConfig:
    """Configuration for plot appearance"""
    figure_size: Tuple[int, int] = (15, 8)
    bar_width: float = 1.0
    alpha: float = 0.7
    edge_color: str = "black"
    rotation: int = 90
    text_color: str = "white"
    font_weight: str = "bold"


class TrackClassifier:
    """Handles track classification logic"""

    @staticmethod
    def classify_tracks(df: pd.DataFrame) -> Dict[TrackType, pd.DataFrame]:
        """
        Classify tracks into fake, good, and doubly reconstructed

        Args:
            df (pd.DataFrame): Input dataframe with track data
        Returns:
            Dict[TrackType, pd.DataFrame]: Dictionary with classified track dataframes
                fake: df[~df["isGoodMother"]]
                good: df[(df["isGoodMother"]) & (~df["same_mc_track_id"])]
                doubly_reco: df[df["same_mc_track_id"]]
        """
        fake = df[~df["isGoodMother"]]
        good = df[(df["isGoodMother"]) & (~df["same_mc_track_id"])]
        doubly_reco = df[df["same_mc_track_id"]]

        print("Track classification:")
        print(f"  Fake tracks: {len(fake)}")
        print(f"  Good tracks: {len(good)}")
        print(f"  Doubly reconstructed tracks: {len(doubly_reco)}")

        return {
            TrackType.FAKE: fake,
            TrackType.GOOD: good,
            TrackType.DOUBLY_RECO: doubly_reco
        }

    @staticmethod
    def classify_doubly_reco_details(
        df_doubly_with: pd.DataFrame,
        df_doubly_without: pd.DataFrame
    ) -> Dict[TrackType, pd.DataFrame]:
        """
        Split doubly reconstructed tracks into:
        - also without sharing
        - only with sharing

        Args:
            df_doubly_with (pd.DataFrame): Doubly reco tracks with shared clusters
            df_doubly_without (pd.DataFrame): Doubly reco tracks without shared clusters
        Returns:
            Dict[TrackType, pd.DataFrame]: Dictionary with split doubly reco track dataframes
        """
        dfs_with = df_doubly_with.groupby(["tf", "iteration"])
        dfs_without = df_doubly_without.groupby(["tf", "iteration"])

        also_without = []
        only_with = []

        for (tf, iteration), df_with in dfs_with:
            df_without = dfs_without.get_group((tf, iteration)) if (tf, iteration) in dfs_without.groups else pd.DataFrame(columns=df_with.columns)

            ids_with = set(df_with["mcTrackID"])
            ids_without = set(df_without["mcTrackID"])

            # Tracks that are doubly reco in both cases (intersection)
            ids_also_without = ids_with.intersection(ids_without)
            also_without.append(df_with[df_with["mcTrackID"].isin(ids_also_without)])

            # Tracks that are doubly reco only due to sharing (difference)
            ids_only_with = ids_with - ids_without
            only_with.append(df_with[df_with["mcTrackID"].isin(ids_only_with)])

        also_without = pd.concat(also_without, ignore_index=True)
        only_with = pd.concat(only_with, ignore_index=True)
        print(f"Found {len(only_with)} doubly reconstructed tracks only due to shared clusters.")

        return {
            TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED: also_without,
            TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: only_with,
        }

    # @staticmethod
    # def classify_doubly_reco_details(
    #     df_doubly_with: pd.DataFrame,
    #     df_doubly_without: pd.DataFrame
    # ) -> Dict[TrackType, pd.DataFrame]:
    #     """
    #     Split doubly reconstructed tracks into:
    #     - also without sharing
    #     - only with sharing

    #     Args:
    #         df_doubly_with (pd.DataFrame): Doubly reco tracks with shared clusters
    #         df_doubly_without (pd.DataFrame): Doubly reco tracks without shared clusters
    #     Returns:
    #         Dict[TrackType, pd.DataFrame]: Dictionary with split doubly reco track dataframes
    #     """
    #     ids_with = set(df_doubly_with["mcTrackID"])
    #     ids_without = set(df_doubly_without["mcTrackID"])

    #     # Tracks that are doubly reco in both cases (intersection)
    #     ids_also_without = ids_with.intersection(ids_without)
    #     also_without = df_doubly_with[df_doubly_with["mcTrackID"].isin(ids_also_without)]

    #     # Tracks that are doubly reco only due to sharing (difference)
    #     ids_only_with = ids_with - ids_without
    #     only_with = df_doubly_with[df_doubly_with["mcTrackID"].isin(ids_only_with)]

    #     print(f"Found {len(only_with)} doubly reconstructed tracks only due to shared clusters.")

    #     return {
    #         TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED: also_without,
    #         TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: only_with,
    #     }


class HistogramPlotter:
    """Handles histogram plotting functionality"""

    TRACK_CONFIGS = {
        TrackType.FAKE: TrackTypeConfig("red", "Fake (Total: {})"),
        TrackType.GOOD: TrackTypeConfig("blue", "Good (Total: {})"),
        TrackType.DOUBLY_RECO: TrackTypeConfig("green", "Multiply-Reco (Total: {})"),
        TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED: TrackTypeConfig("darkgreen", "Multiply-Reco Also w/o Shared (Total: {})"),  # pylint: disable=line-too-long
        TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: TrackTypeConfig("limegreen", "Multiply-Reco Only w/ Shared (Total: {})"),  # pylint: disable=line-too-long
    }

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()

    def prepare_labels(self, all_labels: List[str]) -> Tuple[List[str], Dict[str, int]]:
        """Prepare labels with consistent ordering, moving 'others' to end"""
        label_counts = Counter(all_labels)
        others_count = label_counts.pop("others", 0)

        sorted_items = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        if others_count > 0:
            sorted_items.append(("others", others_count))

        plot_labels = [item[0] for item in sorted_items]
        label_to_index = {label: i for i, label in enumerate(plot_labels)}

        return plot_labels, label_to_index

    def _add_bar_annotations(self, bars, counts: List[int], y_offset: float = 0.1):
        """Add count annotations to bars"""
        for b, count in zip(bars, counts):
            if count > 0:
                plt.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + y_offset,
                    str(count),
                    ha="center", va="bottom"
                )

    def create_single_histogram(self, labels: List[str], title: str, color: str) -> plt.Figure:
        """Create a single histogram with ordered bars"""
        if not labels:
            return None

        plot_labels, _ = self.prepare_labels(labels)
        label_counts = Counter(labels)
        counts = [label_counts.get(label, 0) for label in plot_labels]

        fig, ax = plt.subplots(figsize=self.config.figure_size)
        bars = ax.bar(
            range(len(plot_labels)), counts,
            width=1.0, alpha=self.config.alpha, color=color,
            edgecolor=self.config.edge_color
        )

        ax.set_xlabel("Particle Origin", fontsize=14)
        ax.set_ylabel("Counts", fontsize=14)
        ax.set_title(title)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.set_xticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels, rotation=self.config.rotation, ha="right")

        self._add_bar_annotations(bars, counts)
        plt.tight_layout()
        return fig



    def create_stacked_histogram_by_origin(
            self,
            track_data: Dict[TrackType, pd.DataFrame]
        ) -> plt.Figure:
        """Create stacked histogram grouped by particle origin"""
        data_types = list(track_data.keys())
        data_types.pop(data_types.index(TrackType.DOUBLY_RECO))
        all_labels = []
        for typ in data_types:
            df = track_data[typ]
            if len(df) > 0:
                all_labels.extend(df["label"].tolist())

        if not all_labels:
            return None

        # Use prepare_labels to get properly sorted labels
        plot_labels, _ = self.prepare_labels(all_labels)

        # Count occurrences for each track type using the sorted labels
        track_counts = {}
        for track_type, df in track_data.items():
            labels = df["label"].tolist() if len(df) > 0 else []
            label_counter = Counter(labels)
            track_counts[track_type] = [label_counter.get(label, 0) for label in plot_labels]

        return self._create_stacked_bar_plot(
            plot_labels, track_counts,
            "Particle Origin", "Track Types by Particle Origin"
        )

    def create_stacked_histogram_by_layer(
            self,
            track_data: Dict[TrackType, pd.DataFrame]
        ) -> plt.Figure:
        """Create stacked histogram grouped by shared cluster layer"""
        layers = list(range(7))  # Assuming layers 0-6

        track_counts = {}
        for track_type, df in track_data.items():
            if len(df) > 0:
                layer_counts = df["firstSharedLayer"].value_counts().reindex(layers, fill_value=0)
                track_counts[track_type] = layer_counts.tolist()
            else:
                track_counts[track_type] = [0] * len(layers)

        return self._create_stacked_bar_plot(
            [str(layer) for layer in layers], track_counts,
            "Shared Cluster Layer", "Shared Clusters: Track Types by Layer"
        )

    def _create_stacked_bar_plot(
            self,
            x_labels: List[str],
            track_counts: Dict[TrackType, List[int]],
            xlabel: str,
            title: str
        ) -> plt.Figure:
        """Create a stacked bar plot"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        x_pos = np.arange(len(x_labels))

        # Define stack order - use split doubly reco if available, otherwise use combined
        if (TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED in track_counts and
            TrackType.DOUBLY_RECO_ONLY_WITH_SHARED in track_counts):
            # Use detailed split for datasets with shared cluster analysis
            stack_order = [
                TrackType.FAKE,
                TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED,
                TrackType.DOUBLY_RECO_ONLY_WITH_SHARED,
                TrackType.GOOD
            ]
        else:
            # Use combined doubly reco for datasets without detailed analysis
            stack_order = [
                TrackType.FAKE,
                TrackType.DOUBLY_RECO,
                TrackType.GOOD
            ]

        # Only use track types that actually exist in the data
        stack_order = [t for t in stack_order if t in track_counts]
        bottoms = np.zeros(len(x_labels))

        for track_type in stack_order:
            counts = track_counts[track_type]
            config = self.TRACK_CONFIGS[track_type]
            total = sum(counts)

            # Skip if no data
            if total == 0:
                continue

            ax.bar(
                x_pos, counts, width=self.config.bar_width, alpha=self.config.alpha,
                color=config.color, edgecolor=self.config.edge_color,
                label=config.label_template.format(total), bottom=bottoms
            )

            # Add segment annotations
            self._add_segment_annotations(ax, counts, bottoms)
            bottoms += np.array(counts)

        # Add total annotations
        self._add_total_annotations(ax, bottoms)

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel("Counts", fontsize=14)
        ax.set_title(title)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=self.config.rotation, ha="right")
        ax.legend(fontsize=14)

        plt.tight_layout()
        return fig

    def _add_segment_annotations(self, ax, counts: List[int], bottoms: np.ndarray):
        """Add annotations to individual segments"""
        for i, count in enumerate(counts):
            if count > 0:
                y_pos = bottoms[i] + count / 2
                ax.text(i, y_pos, str(count), ha="center", va="center",
                       color=self.config.text_color, fontweight=self.config.font_weight)

    def _add_total_annotations(self, ax, totals: np.ndarray):
        """Add total count annotations above bars"""
        for i, total in enumerate(totals):
            if total > 0:
                ax.text(i, total + 0.1, str(int(total)), ha="center", va="bottom")


class Histogram2DPlotter:
    # pylint: disable=too-few-public-methods
    """Handles 2D histogram plotting"""

    COLORMAPS = {
        TrackType.FAKE: "Reds",
        TrackType.GOOD: "Blues",
        TrackType.DOUBLY_RECO: "Greens",
        TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED: "Greens",
        TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: "Greens",
    }

    TITLES = {
        TrackType.FAKE: '"Fake" Tracks',
        TrackType.GOOD: '"Good" Tracks',
        TrackType.DOUBLY_RECO: '"Doubly Reco" Tracks',
        TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED: '"Doubly Reco" (also w/o shared)',
        TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: '"Doubly Reco" (only w/ shared)',
    }

    def create_2d_histogram_grid(self, track_data: Dict[TrackType, pd.DataFrame]) -> plt.Figure:  # pylint: disable=too-many-locals
        """Create 2x2 grid of 2D histograms"""
        # Get unique labels across all data
        all_labels = []
        for df in track_data.values():
            if len(df) > 0:
                all_labels.extend(df["label"].tolist())

        if not all_labels:
            return None

        unique_labels = self._get_ordered_unique_labels(all_labels)
        label_to_index = {label: i for i, label in enumerate(unique_labels)}

        fig, axes = plt.subplots(2, 2, figsize=(20, 15))

        # Individual track type plots - only use types that exist in the data
        available_types = [t for t in [TrackType.FAKE, TrackType.GOOD, TrackType.DOUBLY_RECO]
                        if t in track_data and len(track_data[t]) > 0]

        positions = [(0, 0), (0, 1), (1, 0)]
        histograms = {}

        for i, track_type in enumerate(available_types):
            if i < len(positions):  # Don't exceed available subplot positions
                row, col = positions[i]
                hist = self._create_single_2d_histogram(
                    axes[row, col], track_data[track_type],
                    unique_labels, label_to_index, track_type
                )
                histograms[track_type] = hist

        # Hide unused subplots
        for i in range(len(available_types), 3):
            row, col = positions[i]
            axes[row, col].set_visible(False)

        # Combined histogram
        if histograms:  # Only create if we have data
            self._create_combined_2d_histogram(axes[1, 1], histograms, unique_labels)
        else:
            axes[1, 1].set_visible(False)

        # Set x-ticks for visible subplots
        self._set_common_ticks(axes, unique_labels)

        plt.tight_layout()
        return fig

    def _get_ordered_unique_labels(self, all_labels: List[str]) -> List[str]:
        """Get unique labels with 'others' moved to end"""
        unique_labels = sorted(set(all_labels))
        if "others" in unique_labels:
            unique_labels.remove("others")
            unique_labels.append("others")
        return unique_labels

    def _create_single_2d_histogram(  # pylint: disable=too-many-arguments, too-many-positional-arguments
            self,
            ax,
            data: pd.DataFrame,
            unique_labels: List[str],
            label_to_index: Dict[str, int],
            track_type: TrackType
        ) -> Optional[np.ndarray]:
        """Create a single 2D histogram"""
        if len(data) == 0:
            ax.set_title(f'{self.TITLES[track_type]} (No Data)')
            ax.set_xlabel("Particle Origin")
            ax.set_ylabel("Shared cluster layer")
            return None

        label_indices = [label_to_index[label] for label in data["label"]]
        layers = data["firstSharedLayer"].tolist()

        hist, _, _ = np.histogram2d(
            label_indices, layers,
            bins=[len(unique_labels), 7],
            range=[[0, len(unique_labels)], [0, 7]]
        )

        # Prepare for display (set zeros to NaN for transparency)
        hist_display = hist.copy().astype(float)
        hist_display[hist_display == 0] = np.nan

        cmap = plt.get_cmap(self.COLORMAPS[track_type])(np.linspace(0.3, 1.0, 256))
        cmap = mcolors.ListedColormap(cmap)

        im = ax.imshow(
            hist_display.T, origin="lower", aspect="auto",
            cmap=cmap,
            extent=[0, len(unique_labels), 0, 7], vmin=0
        )

        ax.set_title(self.TITLES[track_type])
        ax.set_xlabel("Particle Origin")
        ax.set_ylabel("Shared cluster layer")
        plt.colorbar(im, ax=ax)

        # Add annotations for non-zero bins
        self._add_2d_annotations(ax, hist, len(unique_labels))

        return hist

    def _create_combined_2d_histogram(self, ax, histograms: Dict[TrackType, Optional[np.ndarray]],
                                    unique_labels: List[str]):
        """Create combined 2D histogram"""
        combined_hist = np.zeros((len(unique_labels), 7))

        for hist in histograms.values():
            if hist is not None:
                combined_hist += hist

        # Prepare for display
        hist_display = combined_hist.copy().astype(float)
        hist_display[hist_display == 0] = np.nan

        im = ax.imshow(
            hist_display.T, origin="lower", aspect="auto", cmap="viridis",
            extent=[0, len(unique_labels), 0, 7]
        )

        ax.set_title("Combined (Fake + Good + Doubly Reco)")
        ax.set_xlabel("Particle Origin")
        ax.set_ylabel("Shared cluster layer")
        plt.colorbar(im, ax=ax)

        self._add_2d_annotations(ax, combined_hist, len(unique_labels))

    def _add_2d_annotations(self, ax, hist: np.ndarray, n_labels: int):
        """Add count annotations to 2D histogram"""
        for i in range(n_labels):
            for j in range(7):
                if hist[i, j] > 0:
                    ax.text(i + 0.5, j + 0.5, int(hist[i, j]),
                           ha="center", va="center", color="white", fontweight="bold")

    def _set_common_ticks(self, axes, unique_labels: List[str]):
        """Set consistent ticks across all subplots"""
        for ax in axes.flatten():
            ax.set_xticks(np.arange(len(unique_labels)) + 0.5)
            ax.set_xticklabels(unique_labels, rotation=90, ha="right")
            ax.set_yticks(np.arange(7) + 0.5)
            ax.set_yticklabels(range(7))


class SharedClustersAnalyzer:
    """Main analyzer class that orchestrates the analysis and visualization"""

    def __init__(self, plot_config: PlotConfig = None):
        self.classifier = TrackClassifier()
        self.hist_plotter = HistogramPlotter(plot_config)
        self.hist2d_plotter = Histogram2DPlotter()

    def get_folders_to_process(self, input_folder):
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
                        "with_shared_clusters" in subf.name or "without_shared_clusters" in subf.name  # pylint: disable=line-too-long
                    ):
                        folders_to_keep.append(f)
                        break
            folders = folders_to_keep
        else:
            folders = [input_folder]
        return folders


    def analyze_and_plot(
            self,
            simulation_dir: str,
            output_file: str,
            all_tracks: bool = False
        ):
        """
        Main analysis and plotting function

        Args:
            input_file_with_shared (str): Input parquet file with shared clusters
            input_file_without_shared (str): Input parquet file without shared clusters
            output_file (str): Output PDF file for plots
            all_tracks (bool): Whether to include all tracks or only shared ones
        """
        # Load and filter data
        folders = self.get_folders_to_process(simulation_dir)
        df_with = []
        df_with_delta_rof = []
        df_without = []
        for folder in folders:
            df_with.append(pd.read_parquet(os.path.join(
                    folder,
                    "outputs",
                    "with_shared_clusters",
                    "CheckTracksCA.parquet"
                )))
            df_with_delta_rof.append(pd.read_parquet(os.path.join(
                    folder,
                    "outputs",
                    "with_shared_clusters_delta_rof",
                    "CheckTracksCA.parquet"
                )))
            df_without.append(pd.read_parquet(os.path.join(
                    folder,
                    "outputs",
                    "without_shared_clusters",
                    "CheckTracksCA.parquet"
                )))
        df_with = pd.concat(df_with, ignore_index=True)
        df_with_delta_rof = pd.concat(df_with_delta_rof, ignore_index=True)
        df_without = pd.concat(df_without, ignore_index=True)

        if not all_tracks:
            df_with = df_with.query("isShared == 1")
            df_with_delta_rof = df_with_delta_rof.query("isShared == 1")

        # Classify tracks
        track_data_with = self.classifier.classify_tracks(df_with)
        track_data_with_delta_rof = self.classifier.classify_tracks(df_with_delta_rof)
        track_data_without = self.classifier.classify_tracks(df_without)

        # Refine doubly reco into subcategories
        doubly_split = self.classifier.classify_doubly_reco_details(
            track_data_with[TrackType.DOUBLY_RECO],
            track_data_without[TrackType.DOUBLY_RECO]
        )
        track_data_with.update(doubly_split)
        
        doubly_split_delta_rof = self.classifier.classify_doubly_reco_details(
            track_data_with_delta_rof[TrackType.DOUBLY_RECO],
            track_data_without[TrackType.DOUBLY_RECO]
        )
        track_data_with_delta_rof.update(doubly_split_delta_rof)

        doubly_split_without = self.classifier.classify_doubly_reco_details(
            track_data_with[TrackType.DOUBLY_RECO],
            track_data_without[TrackType.DOUBLY_RECO]
        )
        track_data_without.update(doubly_split_without)

        # Create PDF with plots
        with PdfPages(output_file) as pdf:
            print("Generating plots with shared clusters...")
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.5, 'With Shared Clusters', 
                    ha='center', va='center', fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
            self._save_plots_to_pdf(pdf, track_data_with, cluster_sharing=True, delta_rof=False)

            self._save_efficiency(pdf, track_data_with, track_data_without)

            print("Generating plots with shared clusters and delta ROF...")
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.5, 'With Shared Clusters and Delta ROF', 
                    ha='center', va='center', fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
            self._save_plots_to_pdf(pdf, track_data_with_delta_rof, cluster_sharing=True, delta_rof=True)
            self._save_efficiency(pdf, track_data_with_delta_rof, track_data_without)

            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.5, 'Compare efficiency with and without delta ROF', 
                    ha='center', va='center', fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
            self._save_efficiency_comparison(pdf, track_data_with, track_data_with_delta_rof, track_data_without)

            print("Generating plots without shared clusters...")
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.5, 'Without Shared Clusters', 
                    ha='center', va='center', fontsize=12)
            pdf.savefig(fig)
            plt.close(fig)
            self._save_plots_to_pdf(pdf, track_data_without, cluster_sharing=False)

    def _save_plots_to_pdf(
            self,
            pdf: PdfPages,
            track_data: Dict[TrackType, pd.DataFrame],
            cluster_sharing: bool,
            delta_rof: bool = False
        ):
        """Generate and save all plots to PDF"""
        # Stacked histogram by particle origin
        fig = self.hist_plotter.create_stacked_histogram_by_origin(track_data)
        if fig:
            pdf.savefig(fig)
            plt.close(fig)

        # Individual histograms for each track type
        # pylint: disable=line-too-long
        if cluster_sharing:
            track_titles = {
                TrackType.FAKE: 'Shared Clusters: "Fake" Tracks Only',
                TrackType.DOUBLY_RECO: "Shared Clusters: Doubly Reco Tracks Only",
                TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED: 'Shared Clusters: "Doubly Reco" (also w/o shared)',
                TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: 'Shared Clusters: "Doubly Reco" (only w/ shared)',
                TrackType.GOOD: 'Shared Clusters: "Good" Tracks Only'
            }
        else:
            track_titles = {
                TrackType.FAKE: 'No Shared Clusters: "Fake" Tracks Only',
                TrackType.DOUBLY_RECO: "No Shared Clusters: Doubly Reco Tracks Only",
                TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED: 'No Shared Clusters: "Doubly Reco" (also w/o shared)',
                TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: 'No Shared Clusters: "Doubly Reco" (only w/ shared)',
                TrackType.GOOD: 'No Shared Clusters: "Good" Tracks Only'
            }

        track_colors = {
            TrackType.FAKE: "red",
            TrackType.DOUBLY_RECO: "green",
            TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED: "darkgreen",
            TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: "limegreen",
            TrackType.GOOD: "blue"
        }

        for track_type in [
            TrackType.FAKE,
            TrackType.DOUBLY_RECO,
            TrackType.GOOD,
            TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED,
            TrackType.DOUBLY_RECO_ONLY_WITH_SHARED
        ]:
            df = track_data[track_type]
            # if len(df) > 0:
            labels = df["label"].tolist()
            fig = self.hist_plotter.create_single_histogram(
                labels, track_titles[track_type], track_colors[track_type]
            )
            if fig:
                pdf.savefig(fig)
                plt.close(fig)

        # 2D histogram grid
        fig = self.hist2d_plotter.create_2d_histogram_grid(track_data)
        if fig:
            pdf.savefig(fig)
            plt.close(fig)

        # Stacked histogram by layer
        fig = self.hist_plotter.create_stacked_histogram_by_layer(track_data)
        if fig:
            pdf.savefig(fig)
            plt.close(fig)

    def _save_efficiency(self, pdf, track_data_with, track_data_without):
        data_types = list(track_data_with.keys())
        if TrackType.DOUBLY_RECO in data_types:
            data_types.pop(data_types.index(TrackType.DOUBLY_RECO))
        
        all_labels = []
        for typ in data_types:
            df_without = track_data_without[typ]
            if len(df_without) > 0:
                all_labels.extend(df_without["label"].tolist())

        if not all_labels:
            return None

        plot_labels, _ = self.hist_plotter.prepare_labels(all_labels)
        x_pos = np.arange(len(plot_labels))
        
        # Define markers for different track types
        markers = ['o', 'x']

        track_counts = {True: {}, False: {}}
        for shared in [True, False]:
            data_source = track_data_with if shared else track_data_without
            for track_type, df in data_source.items():
                labels = df["label"].tolist() if len(df) > 0 else []
                label_counter = Counter(labels)
                track_counts[shared][track_type] = np.array([label_counter.get(label, 0) for label in plot_labels])

        fig, ax = plt.subplots(figsize=self.hist_plotter.config.figure_size)
        ax.set_yscale('log')
        
        order = [TrackType.FAKE, TrackType.GOOD]
        
        total_counts = np.zeros(len(plot_labels))
        for t in data_types:
            if t in track_counts[False]:
                total_counts += track_counts[False][t]

        safe_total = np.where(total_counts == 0, np.nan, total_counts)
        bottoms = np.zeros(len(plot_labels))
        legends = {
            TrackType.FAKE: "Fake",
            TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: "Multiply reco only w/ shared",
            TrackType.GOOD: "Good"
        }

        for i, track_type in enumerate(order):
            if track_type not in track_counts[True]:
                continue
                
            counts = track_counts[True][track_type]
            efficiencies = counts/safe_total
            # efficiencies = np.divide(counts, safe_total, out=np.zeros_like(counts, dtype=float), where=total_counts != 0)

            current_tops = bottoms + efficiencies
            color = f"C{i}"
            ax.errorbar(
                x_pos, current_tops, 
                xerr=0.5, 
                fmt=markers[i], 
                color=color,
                markersize=5,
                capsize=0,
                label=legends[track_type]
            )

            for x, eff_val, total_height in zip(x_pos, efficiencies, current_tops):
                if eff_val > 0:
                    ax.text(
                        x, total_height * 1.1, f'{eff_val:.1e}', 
                        ha='center', va='bottom', fontsize=8, color=color
                    )


        ax.set_xlabel("Origin", fontsize=14)
        ax.set_ylabel("Tracks with shared clusters / Total tracks", fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_labels, rotation=90, ha="right")
        
        ax.set_ylim(5e-6, 2.0) 
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, which="both", ls="--", linewidth=0.5)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=self.hist_plotter.config.figure_size)
        ax.set_yscale('log')
        
        order = [TrackType.DOUBLY_RECO_ONLY_WITH_SHARED, TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED]
        
        total_counts = np.zeros(len(plot_labels))
        for t in data_types:
            if t in track_counts[False]:
                total_counts += track_counts[False][t]

        safe_total = np.where(total_counts == 0, np.nan, total_counts)
        bottoms = np.zeros(len(plot_labels))
        legends = {
            TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: "Multiply reco only w/ shared",
            TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED: "Multiply reco also w/o shared"
        }

        for i, track_type in enumerate(order):
            if track_type not in track_counts[True]:
                continue
                
            counts = track_counts[True][track_type]
            efficiencies = counts/safe_total
            # efficiencies = np.divide(counts, safe_total, out=np.zeros_like(counts, dtype=float), where=total_counts != 0)

            current_tops = bottoms + efficiencies
            color = f"C{3-i}"

            ax.errorbar(
                x_pos, current_tops, 
                xerr=0.5, 
                fmt=markers[i], 
                color=color,
                markersize=5,
                capsize=0,
                label=legends[track_type]
            )

            for x, eff_val, total_height in zip(x_pos, efficiencies, current_tops):
                if eff_val > 0:
                    ax.text(
                        x, total_height * 1.1, f'{eff_val:.1e}', 
                        ha='center', va='bottom', fontsize=8, color=color
                    )

        ax.set_xlabel("Origin", fontsize=14)
        ax.set_ylabel("Tracks with shared clusters / Total tracks", fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(plot_labels, rotation=90, ha="right")
        
        ax.set_ylim(5e-6, 2.0) 
        ax.legend(fontsize=14)
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(True, which="both", ls="--", linewidth=0.5)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _save_efficiency_comparison(self, pdf, track_data_with, track_data_with_delta_rof, track_data_without):
        data_types = list(track_data_with.keys())
        if TrackType.DOUBLY_RECO in data_types:
            data_types.pop(data_types.index(TrackType.DOUBLY_RECO))
        
        all_labels = []
        for typ in data_types:
            df_without = track_data_without[typ]
            if len(df_without) > 0:
                all_labels.extend(df_without["label"].tolist())

        if not all_labels:
            return None

        plot_labels, _ = self.hist_plotter.prepare_labels(all_labels)
        x_pos = np.arange(len(plot_labels))
        
        # Define markers for different track types
        markers = ['o', 'x', 'd', 'P']

        track_counts = {"with_shared": {}, "with_shared_delta_rof": {}, "without_shared": {}}
        for data_type in ["with_shared", "with_shared_delta_rof", "without_shared"]:
            if data_type == "with_shared":
                data_source = track_data_with
            elif data_type == "with_shared_delta_rof":
                data_source = track_data_with_delta_rof
            else:
                data_source = track_data_without

            for track_type, df in data_source.items():
                labels = df["label"].tolist() if len(df) > 0 else []
                label_counter = Counter(labels)
                track_counts[data_type][track_type] = np.array([label_counter.get(label, 0) for label in plot_labels])

        fig, (ax, ax_ratio) = plt.subplots(
            2, 1, 
            figsize=self.hist_plotter.config.figure_size, 
            sharex=True, 
            gridspec_kw={'height_ratios': [3, 1]}
        )
        
        ax.set_yscale('log')
        
        order = [TrackType.FAKE, TrackType.GOOD]
        
        total_counts = np.zeros(len(plot_labels))
        for t in data_types:
            if t in track_counts["without_shared"]:
                total_counts += track_counts["without_shared"][t]

        safe_total = np.where(total_counts == 0, np.nan, total_counts)
        bottoms = np.zeros(len(plot_labels))
        legends = {
            TrackType.FAKE: "Fake",
            TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: "Multiply reco only w/ shared",
            TrackType.GOOD: "Good"
        }

        for i, track_type in enumerate(order):
            if track_type not in track_counts["with_shared"]:
                continue
                
            counts = track_counts["with_shared"][track_type]
            efficiencies = counts/safe_total
                
            counts_delta_rof = track_counts["with_shared_delta_rof"][track_type]
            efficiencies_delta_rof = counts_delta_rof/safe_total
            # efficiencies = np.divide(counts, safe_total, out=np.zeros_like(counts, dtype=float), where=total_counts != 0)

            current_tops = bottoms + efficiencies
            color = f"C{i}"
            ax.errorbar(
                x_pos, current_tops, 
                xerr=0.5, 
                fmt=markers[i], 
                color=f"C{i}",
                markersize=5,
                capsize=0,
                label=f"{legends[track_type]} (no delta ROF)"
            )

            current_tops = bottoms + efficiencies_delta_rof
            
            ax.errorbar(
                x_pos, current_tops, 
                xerr=0.5, 
                fmt=markers[i+2], 
                color=f"C{i+2}",
                markersize=5,
                capsize=0,
                label=f"{legends[track_type]} (delta ROF)"
            )

            ratio = np.where(efficiencies != 0, efficiencies_delta_rof / efficiencies, np.nan)
            
            ax_ratio.errorbar(
                x_pos, ratio, 
                xerr=0.5, fmt=markers[i+2], color=f"C{i+2}",
                markersize=4, capsize=0, label=f"{legends[track_type]} ratio"
            )

        ax.set_ylabel("Tracks with shared clusters / Total tracks")
        ax.set_xticks(x_pos)
        
        ax_ratio.set_xlabel("Origin")
        ax_ratio.set_ylabel("Ratio\n(delta ROF/no delta ROF)")
        ax_ratio.axhline(1.0, color='black', lw=1, ls='-')
        ax_ratio.set_ylim(0, 4)
        ax_ratio.legend()
        ax_ratio.grid(True, which="both", ls="--", alpha=0.5)
        ax_ratio.set_xticklabels(plot_labels, rotation=90, ha="right")

        ax.set_ylim(5e-6, 2.0) 
        ax.legend()
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, which="both", ls="--", linewidth=0.5)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        fig, (ax, ax_ratio) = plt.subplots(
            2, 1, 
            figsize=self.hist_plotter.config.figure_size, 
            sharex=True, 
            gridspec_kw={'height_ratios': [3, 1]}
        )
        
        ax.set_yscale('log')
        
        order = [TrackType.DOUBLY_RECO_ONLY_WITH_SHARED, TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED]
        
        total_counts = np.zeros(len(plot_labels))
        for t in data_types:
            if t in track_counts["without_shared"]:
                total_counts += track_counts["without_shared"][t]

        safe_total = np.where(total_counts == 0, np.nan, total_counts)
        bottoms = np.zeros(len(plot_labels))
        legends = {
            TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: "Multiply reco only w/ shared",
            TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED: "Multiply reco also w/o shared"
        }

        for i, track_type in enumerate(order):
            if track_type not in track_counts["with_shared"]:
                continue
                
            counts = track_counts["with_shared"][track_type]
            efficiencies = counts/safe_total
                
            counts_delta_rof = track_counts["with_shared_delta_rof"][track_type]
            efficiencies_delta_rof = counts_delta_rof/safe_total
            # efficiencies = np.divide(counts, safe_total, out=np.zeros_like(counts, dtype=float), where=total_counts != 0)

            current_tops = bottoms + efficiencies
            color = f"C{i}"
            
            ax.errorbar(
                x_pos, current_tops, 
                xerr=0.5, 
                fmt=markers[i], 
                color=f"C{i}",
                markersize=5,
                capsize=0,
                label=f"{legends[track_type]} (no delta ROF)"
            )

            current_tops = bottoms + efficiencies_delta_rof
            ax.errorbar(
                x_pos, current_tops, 
                xerr=0.5, 
                fmt=markers[i+2], 
                color=f"C{i+2}",
                markersize=5,
                capsize=0,
                label=f"{legends[track_type]} (delta ROF)"
            )

            ratio = np.where(efficiencies != 0, efficiencies_delta_rof / efficiencies, np.nan)
            ax_ratio.errorbar(
                x_pos, ratio, 
                xerr=0.5, fmt=markers[i+2], color=f"C{i}",
                markersize=4, capsize=0, label=f"{legends[track_type]} ratio"
            )

        ax.set_ylabel("Tracks with shared clusters / Total tracks")
        ax.set_xticks(x_pos)
        
        ax_ratio.set_xlabel("Origin")
        ax_ratio.set_ylabel("Ratio\n(delta ROF/no delta ROF)")
        ax_ratio.axhline(1.0, color='black', lw=1, ls='-') # Reference line at 1
        ax_ratio.set_ylim(0, 4)
        ax_ratio.legend()
        ax_ratio.grid(True, which="both", ls="--", alpha=0.5)
        ax_ratio.set_xticklabels(plot_labels, rotation=90, ha="right")
        
        ax.set_ylim(5e-6, 2.0) 
        ax.legend()
        ax.tick_params(axis='both', labelsize=12)
        ax.grid(True, which="both", ls="--", linewidth=0.5)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Draw shared clusters analysis")
    parser.add_argument(
        "simulation_dir", type=str,
        help="Input simulation directory"
    )
    parser.add_argument(
        "output", type=str,
        help="Output PDF file"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Include all tracks, not only shared ones"
    )
    parser.add_argument(
        "--figure-size", type=int, nargs=2, default=[15, 8],
        help="Figure size (width height)"
    )

    args = parser.parse_args()

    # Create analyzer with custom config
    plot_config = PlotConfig(figure_size=tuple(args.figure_size))
    analyzer = SharedClustersAnalyzer(plot_config)

    # Run analysis
    print("Starting shared clusters analysis...")
    analyzer.analyze_and_plot(
        args.simulation_dir,
        args.output,
        args.all
    )
    print(f"Analysis complete. Results saved to: {args.output}")



if __name__ == "__main__":
    main()
