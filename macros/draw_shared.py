"""Refactored script to draw shared clusters analysis results"""

import argparse
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


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
    bar_width: float = 0.8
    alpha: float = 0.7
    edge_color: str = "black"
    rotation: int = 45
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

        print(f"Track classification:")
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
        ids_with = set(df_doubly_with["mcTrackID"])
        ids_without = set(df_doubly_without["mcTrackID"])

        # Tracks that are doubly reco in both cases (intersection)
        ids_also_without = ids_with.intersection(ids_without)
        also_without = df_doubly_with[df_doubly_with["mcTrackID"].isin(ids_also_without)]

        # Tracks that are doubly reco only due to sharing (difference)
        ids_only_with = ids_with - ids_without
        only_with = df_doubly_with[df_doubly_with["mcTrackID"].isin(ids_only_with)]
        
        print(f"Found {len(only_with)} doubly reconstructed tracks only due to shared clusters.")

        return {
            TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED: also_without,
            TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: only_with,
        }


class HistogramPlotter:
    """Handles histogram plotting functionality"""

    TRACK_CONFIGS = {
        TrackType.FAKE: TrackTypeConfig("red", "Fake (Total: {})"),
        TrackType.GOOD: TrackTypeConfig("blue", "Good (Total: {})"),
        TrackType.DOUBLY_RECO: TrackTypeConfig("green", "Doubly Reco (Total: {})"),
        TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED: TrackTypeConfig("darkgreen", "Doubly Reco Also w/o Shared (Total: {})"),
        TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: TrackTypeConfig("limegreen", "Doubly Reco Only w/ Shared (Total: {})"),
    }

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()
    
    def _prepare_labels(self, all_labels: List[str]) -> Tuple[List[str], Dict[str, int]]:
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
        for bar, count in zip(bars, counts):
            if count > 0:
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + y_offset,
                    str(count),
                    ha="center", va="bottom"
                )
    
    def create_single_histogram(self, labels: List[str], title: str, color: str) -> plt.Figure:
        """Create a single histogram with ordered bars"""
        if not labels:
            return None
            
        plot_labels, _ = self._prepare_labels(labels)
        label_counts = Counter(labels)
        counts = [label_counts.get(label, 0) for label in plot_labels]
        
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        bars = ax.bar(
            range(len(plot_labels)), counts,
            width=1.0, alpha=self.config.alpha, color=color,
            edgecolor=self.config.edge_color
        )
        
        ax.set_xlabel("Particle Origin")
        ax.set_ylabel("Counts")
        ax.set_title(title)
        ax.set_xticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels, rotation=self.config.rotation, ha="right")
        
        self._add_bar_annotations(bars, counts)
        plt.tight_layout()
        return fig
    

 
    def create_stacked_histogram_by_origin(self, track_data: Dict[TrackType, pd.DataFrame]) -> plt.Figure:
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
            
        # Use _prepare_labels to get properly sorted labels
        plot_labels, label_to_index = self._prepare_labels(all_labels)
        
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

    def create_stacked_histogram_by_layer(self, track_data: Dict[TrackType, pd.DataFrame]) -> plt.Figure:
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

    def _create_stacked_bar_plot(self, x_labels: List[str], track_counts: Dict[TrackType, List[int]], 
                                xlabel: str, title: str) -> plt.Figure:
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
            self._add_segment_annotations(ax, x_pos, counts, bottoms)
            bottoms += np.array(counts)
        
        # Add total annotations
        self._add_total_annotations(ax, x_pos, bottoms)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts")
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=self.config.rotation, ha="right")
        ax.legend()
        
        plt.tight_layout()
        return fig
        
    def _add_segment_annotations(self, ax, x_pos: np.ndarray, counts: List[int], bottoms: np.ndarray):
        """Add annotations to individual segments"""
        for i, count in enumerate(counts):
            if count > 0:
                y_pos = bottoms[i] + count / 2
                ax.text(i, y_pos, str(count), ha="center", va="center",
                       color=self.config.text_color, fontweight=self.config.font_weight)
    
    def _add_total_annotations(self, ax, x_pos: np.ndarray, totals: np.ndarray):
        """Add total count annotations above bars"""
        for i, total in enumerate(totals):
            if total > 0:
                ax.text(i, total + 0.1, str(int(total)), ha="center", va="bottom")


class Histogram2DPlotter:
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
    
    def create_2d_histogram_grid(self, track_data: Dict[TrackType, pd.DataFrame]) -> plt.Figure:
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
    
    def _create_single_2d_histogram(self, ax, data: pd.DataFrame, unique_labels: List[str], 
                                  label_to_index: Dict[str, int], track_type: TrackType) -> Optional[np.ndarray]:
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
        
        im = ax.imshow(
            hist_display.T, origin="lower", aspect="auto",
            cmap=self.COLORMAPS[track_type],
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
            ax.set_xticklabels(unique_labels, rotation=45, ha="right")
            ax.set_yticks(np.arange(7) + 0.5)
            ax.set_yticklabels(range(7))


class SharedClustersAnalyzer:
    """Main analyzer class that orchestrates the analysis and visualization"""
    
    def __init__(self, plot_config: PlotConfig = None):
        self.classifier = TrackClassifier()
        self.hist_plotter = HistogramPlotter(plot_config)
        self.hist2d_plotter = Histogram2DPlotter()


    def analyze_and_plot(
            self,
            input_file_with_shared: str,
            input_file_without_shared: str,
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
        df_with = pd.read_parquet(input_file_with_shared)
        df_without = pd.read_parquet(input_file_without_shared)

        if not all_tracks:
            df_with = df_with.query("isShared == 1")
        
        # Classify tracks
        track_data_with = self.classifier.classify_tracks(df_with)
        track_data_without = self.classifier.classify_tracks(df_without)

        # Refine doubly reco into subcategories
        doubly_split = self.classifier.classify_doubly_reco_details(
            track_data_with[TrackType.DOUBLY_RECO],
            track_data_without[TrackType.DOUBLY_RECO]
        )
        track_data_with.update(doubly_split)

        doubly_split_without = self.classifier.classify_doubly_reco_details(
            track_data_with[TrackType.DOUBLY_RECO],
            track_data_without[TrackType.DOUBLY_RECO]
        )
        track_data_without.update(doubly_split_without)
        
        # Create PDF with plots
        with PdfPages(output_file) as pdf:
            print("Generating plots with shared clusters...")
            self._save_plots_to_pdf(pdf, track_data_with, cluster_sharing=True)
            
            print("Generating plots without shared clusters...")
            self._save_plots_to_pdf(pdf, track_data_without, cluster_sharing=False)

    def _save_plots_to_pdf(self, pdf: PdfPages, track_data: Dict[TrackType, pd.DataFrame], cluster_sharing: bool):
        """Generate and save all plots to PDF"""
        # Stacked histogram by particle origin
        fig = self.hist_plotter.create_stacked_histogram_by_origin(track_data)
        if fig:
            pdf.savefig(fig)
            plt.close(fig)
        
        # Individual histograms for each track type
        if (cluster_sharing):
            track_titles = {
                TrackType.FAKE: 'Shared Clusters: "Fake" Tracks Only',
                TrackType.DOUBLY_RECO: "Shared Clusters: Doubly Reco Tracks Only",
                TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED: 'No Shared Clusters: "Doubly Reco" (also w/o shared)',
                TrackType.DOUBLY_RECO_ONLY_WITH_SHARED: 'No Shared Clusters: "Doubly Reco" (only w/ shared)',
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
        
        for track_type in [TrackType.FAKE, TrackType.DOUBLY_RECO, TrackType.GOOD, TrackType.DOUBLY_RECO_ALSO_WITHOUT_SHARED, TrackType.DOUBLY_RECO_ONLY_WITH_SHARED]:
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

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Draw shared clusters analysis")
    parser.add_argument(
        "input_with_shared", type=str, 
        help="Input CheckTracksCA parquet file (with shared clusters)"
    )
    parser.add_argument(
        "input_without_shared", type=str, 
        help="Input CheckTracksCA parquet file (without shared clusters)"
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
        args.input_with_shared, 
        args.input_without_shared, 
        args.output, 
        args.all
    )
    print(f"Analysis complete. Results saved to: {args.output}")



if __name__ == "__main__":
    main()