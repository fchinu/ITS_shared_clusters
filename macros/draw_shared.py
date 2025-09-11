import argparse
import uproot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
from matplotlib.backends.backend_pdf import PdfPages

PDG_LABELS = {
    11: "e",
    13: r"\mu",
    22: r"\gamma",
    111: r"\pi^{0}",
    113: r"\rho^{0}",
    211: r"\pi^{\pm}",
    213: r"\rho^{\pm}",
    221: r"\eta",
    223: r"\omega",
    310: r"K_{s}^{0}",
    313: r"K*^{0}",
    321: r"K",
    331: r"\eta'",
    411: r"D^{\pm}",
    1114: r"\Delta^{\pm}",
    2212: "p",
    2224: r"\Delta^{++}",
    3112: r"\Sigma^{\pm}",
    1000822080: "Pb"
}

def create_single_histogram(labels, title, color):
    """Create a single histogram with ordered bars"""
    label_counts = Counter(labels)
    
    altro_count = label_counts.pop("altro", 0)
    
    sorted_items = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    
    if altro_count > 0:
        sorted_items.append(("altro", altro_count))
    
    plot_labels = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(plot_labels)), counts, width=1.0, alpha=0.7, 
                   color=color, edgecolor='black')
    
    plt.xlabel('Particle Origin')
    plt.ylabel('Counts')
    plt.title(title)
    plt.xticks(range(len(plot_labels)), plot_labels, rotation=45, ha='right')
    
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    return sorted_items


def classify_and_create_data(df_shared):
    """Classify particles and return all necessary data for plotting"""
    
    df = df_shared.copy()

    return df[~df["isGoodMother"]], df[(df["isGoodMother"]) & (~df["same_mc_track_id"])], df[df["same_mc_track_id"]] # fake_data, good_data

def create_2d_histogram_classified(fake_data, good_data, doubly_reco_data):
    """Create 2D histogram showing fake vs good particles"""
    # Combine all data to get consistent label ordering
    all_labels = list(fake_data['label']) + list(good_data['label']) + list(doubly_reco_data['label'])
    unique_labels = sorted(set(all_labels))
    
    # Move "altro" to end if present
    if "altro" in unique_labels:
        unique_labels.remove("altro")
        unique_labels.append("altro")
    
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    
    # Prepare data for histograms
    fake_label_indices = [label_to_index[label] for label in fake_data['label']]
    fake_layers = list(fake_data['firstSharedLayer'])
    
    good_label_indices = [label_to_index[label] for label in good_data['label']]
    good_layers = list(good_data['firstSharedLayer'])

    doubly_reco_label_indices = [label_to_index[label] for label in doubly_reco_data['label']]
    doubly_reco_layers = list(doubly_reco_data['firstSharedLayer'])
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # Fake particles histogram
    if len(fake_data) > 0:
        hist_fake, _, _ = np.histogram2d(
            fake_label_indices, fake_layers,
            bins=[len(unique_labels), 7],
            range=[[0, len(unique_labels)], [0, 7]]
        )

        # Set zeros to NaN for transparent display
        hist_fake_display = hist_fake.copy().astype(float)
        hist_fake_display[hist_fake_display == 0] = np.nan

        im1 = axes[0, 0].imshow(hist_fake_display.T, origin='lower', aspect='auto', cmap='Reds',
                            extent=[0, len(unique_labels), 0, 7], vmin=0)
        axes[0, 0].set_title('"Fake" Tracks')
        axes[0, 0].set_xlabel('Particle Origin')
        axes[0, 0].set_ylabel('Shared cluster layer')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Annotate non-zero bins
        for i in range(len(unique_labels)):
            for j in range(7):
                if hist_fake[i, j] > 0:
                    axes[0, 0].text(i + 0.5, j + 0.5, int(hist_fake[i, j]), 
                                ha='center', va='center', color='white', fontweight='bold')
    else:
        axes[0, 0].set_title('"Fake" Tracks (No Data)')
        axes[0, 0].set_xlabel('Particle Origin')
        axes[0, 0].set_ylabel('Shared cluster layer')
    
    # Good particles histogram
    if len(good_data) > 0:
        hist_good, _, _ = np.histogram2d(
            good_label_indices, good_layers,
            bins=[len(unique_labels), 7],
            range=[[0, len(unique_labels)], [0, 7]]
        )

        # Set zeros to NaN for transparent display
        hist_good_display = hist_good.copy().astype(float)
        hist_good_display[hist_good_display == 0] = np.nan

        im2 = axes[0, 1].imshow(hist_good_display.T, origin='lower', aspect='auto', cmap='Blues',
                            extent=[0, len(unique_labels), 0, 7], vmin=0)
        axes[0, 1].set_title('"Good" Tracks')
        axes[0, 1].set_xlabel('Particle Origin')
        axes[0, 1].set_ylabel('Shared cluster layer')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Annotate non-zero bins
        for i in range(len(unique_labels)):
            for j in range(7):
                if hist_good[i, j] > 0:
                    axes[0, 1].text(i + 0.5, j + 0.5, int(hist_good[i, j]), 
                                ha='center', va='center', color='white', fontweight='bold')
    else:
        axes[0, 1].set_title('"Good" Tracks (No Data)')
        axes[0, 1].set_xlabel('Particle Origin')
        axes[0, 1].set_ylabel('Shared cluster layer')
    
    # Doubly reco histogram
    if len(doubly_reco_data) > 0:
        hist_doubly_reco, _, _ = np.histogram2d(
            doubly_reco_label_indices, doubly_reco_layers,
            bins=[len(unique_labels), 7],
            range=[[0, len(unique_labels)], [0, 7]]
        )

        # Set zeros to NaN for transparent display
        hist_doubly_reco_display = hist_doubly_reco.copy().astype(float)
        hist_doubly_reco_display[hist_doubly_reco_display == 0] = np.nan

        im2 = axes[1, 0].imshow(hist_doubly_reco_display.T, origin='lower', aspect='auto', cmap='Greens',
                            extent=[0, len(unique_labels), 0, 7], vmin=0)
        axes[1, 0].set_title('"Doubly Reco" Tracks')
        axes[1, 0].set_xlabel('Particle Origin')
        axes[1, 0].set_ylabel('Shared cluster layer')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Annotate non-zero bins
        for i in range(len(unique_labels)):
            for j in range(7):
                if hist_doubly_reco[i, j] > 0:
                    axes[1, 0].text(i + 0.5, j + 0.5, int(hist_doubly_reco[i, j]), 
                                ha='center', va='center', color='white', fontweight='bold')
    else:
        axes[1, 0].set_title('"Good" Tracks (No Data)')
        axes[1, 0].set_xlabel('Particle Origin')
        axes[1, 0].set_ylabel('Shared cluster layer')
    
    # Combined histogram
    hist_combined = np.zeros((len(unique_labels), 7))
    if len(fake_data) > 0:
        hist_combined += hist_fake
    if len(good_data) > 0:
        hist_combined += hist_good
    if len(doubly_reco_data) > 0:
        hist_combined += hist_doubly_reco

    # Set zeros to NaN for transparent display
    hist_combined_display = hist_combined.copy().astype(float)
    hist_combined_display[hist_combined_display == 0] = np.nan
    
    im3 = axes[1, 1].imshow(hist_combined_display.T, origin='lower', aspect='auto', cmap='viridis',
                        extent=[0, len(unique_labels), 0, 7])
    axes[1, 1].set_title('Combined (Fake + Good + Doubly Reco)')
    axes[1, 1].set_xlabel('Particle Origin')
    axes[1, 1].set_ylabel('Shared cluster layer')
    plt.colorbar(im3, ax=axes[1, 1])
    
    # Annotate non-zero bins for combined (only show counts > 0)
    for i in range(len(unique_labels)):
        for j in range(7):
            if hist_combined[i, j] > 0:
                axes[1, 1].text(i + 0.5, j + 0.5, int(hist_combined[i, j]), 
                            ha='center', va='center', color='white', fontweight='bold')
    
    # Set x-ticks for all subplots
    for ax in axes.flatten():
        ax.set_xticks(np.arange(len(unique_labels)) + 0.5)
        ax.set_xticklabels(unique_labels, rotation=45, ha='right')
        ax.set_yticks(np.arange(7) + 0.5)
        ax.set_yticklabels(range(7))
    
    plt.tight_layout()

def draw_shared_clusters(input_file, output_file):
    df = pd.read_parquet(input_file)
    
    df_shared = df.copy() #.query("isShared == 1")
        
    fake_data, good_data, doubly_reco_data = classify_and_create_data(df_shared)

    # Extract labels and layers from DataFrames
    fake_labels = list(fake_data['label']) if len(fake_data) > 0 else []
    good_labels = list(good_data['label']) if len(good_data) > 0 else []
    doubly_reco_labels = list(doubly_reco_data['label']) if len(doubly_reco_data) > 0 else []

    fake_first_shared_layers = list(fake_data['firstSharedLayer']) if len(fake_data) > 0 else []
    good_first_shared_layers = list(good_data['firstSharedLayer']) if len(good_data) > 0 else []
    doubly_reco_first_shared_layers = list(doubly_reco_data['firstSharedLayer']) if len(doubly_reco_data) > 0 else []

    # Create output PDF
    with PdfPages(output_file) as pdf:
        all_labels = fake_labels + good_labels + doubly_reco_labels
        all_first_shared_layers = fake_first_shared_layers + good_first_shared_layers + doubly_reco_first_shared_layers
        all_label_counts = Counter(all_labels)
        all_first_shared_layer_counts = Counter(all_first_shared_layers)

        altro_count = all_label_counts.pop("altro", 0)
        
        all_sorted_items = sorted(all_label_counts.items(), key=lambda x: x[1], reverse=True)
        
        if altro_count > 0:
            all_sorted_items.append(("altro", altro_count))
        
        all_plot_labels = [item[0] for item in all_sorted_items]
        
        fake_counts_dict = Counter(fake_labels)
        good_counts_dict = Counter(good_labels)
        doubly_reco_counts_dict = Counter(doubly_reco_labels)

        fake_counts = [fake_counts_dict.get(label, 0) for label in all_plot_labels]
        good_counts = [good_counts_dict.get(label, 0) for label in all_plot_labels]
        doubly_reco_counts = [doubly_reco_counts_dict.get(label, 0) for label in all_plot_labels]

        plt.figure(figsize=(15, 8))
        
        x_pos = range(len(all_plot_labels))
        width = 0.8
        
        bars1 = plt.bar(x_pos, fake_counts, width=width, alpha=0.7, 
                       color='red', edgecolor='black', label='Fake')
        bars3 = plt.bar(x_pos, doubly_reco_counts, width=width, alpha=0.7,
                          color='green', edgecolor='black', label='Doubly Reco', bottom=fake_counts)
        bars2 = plt.bar(x_pos, good_counts, width=width, alpha=0.7, 
                       color='blue', edgecolor='black', label='Good', bottom=fake_counts + np.array(doubly_reco_counts))
        
        plt.xlabel('Particle Origin')
        plt.ylabel('Counts')
        plt.title('Shared Clusters: "Fake" vs "Good" vs Doubly Reco Tracks')
        plt.xticks(x_pos, all_plot_labels, rotation=45, ha='right')
        plt.legend()
        
        # Add count labels on bars
        for i, (fake_count, good_count, doubly_reco_count) in enumerate(zip(fake_counts, good_counts, doubly_reco_counts)):
            total = fake_count + good_count + doubly_reco_count
            if total > 0:
                plt.text(i, total + 0.1, str(total), ha='center', va='bottom')
                if fake_count > 0:
                    plt.text(i, fake_count/2, str(fake_count), ha='center', va='center', 
                            color='white', fontweight='bold')
                if doubly_reco_count > 0:
                    plt.text(i, fake_count + doubly_reco_count/2, str(doubly_reco_count), ha='center', va='center', 
                            color='white', fontweight='bold')
                if good_count > 0:
                    plt.text(i, fake_count + doubly_reco_count + good_count/2, str(good_count), ha='center', va='center', 
                            color='white', fontweight='bold')
                    
        
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        
        if fake_labels:
            fake_items = create_single_histogram(fake_labels, 'Shared Clusters: "Fake" Tracks Only', 'red')
            pdf.savefig()
            plt.close()
        
        if doubly_reco_labels:
            doubly_reco_items = create_single_histogram(doubly_reco_labels, 'Shared Clusters: Doubly Reco Tracks Only', 'green')
            pdf.savefig()
            plt.close()
        
        if good_labels:
            good_items = create_single_histogram(good_labels, 'Shared Clusters: "Good" Tracks Only', 'blue')
            pdf.savefig()
            plt.close()

        create_2d_histogram_classified(fake_data, good_data, doubly_reco_data)
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(15, 8))
        x_pos = np.arange(7)  # if layers are 0..6
        width = 0.8

        # Count per layer
        fake_counts = df_shared.query("not isGoodMother")['firstSharedLayer'].value_counts().reindex(x_pos, fill_value=0)
        doubly_reco_counts = df_shared.query("same_mc_track_id")['firstSharedLayer'].value_counts().reindex(x_pos, fill_value=0)
        good_counts = df_shared.query("isGoodMother and not same_mc_track_id")['firstSharedLayer'].value_counts().reindex(x_pos, fill_value=0)

        # Plot stacked bars
        plt.bar(x_pos, fake_counts, width=width, alpha=0.7, color='red', edgecolor='black', label='Fake')
        plt.bar(x_pos, doubly_reco_counts, width=width, alpha=0.7, color='green', edgecolor='black', label='Doubly Reco', bottom=fake_counts)
        plt.bar(x_pos, good_counts, width=width, alpha=0.7, color='blue', edgecolor='black', label='Good', bottom=fake_counts+np.array(doubly_reco_counts))

        plt.xlabel('Shared Cluster Layer')
        plt.ylabel('Counts')
        plt.title('Shared Clusters: "Fake" vs "Good" vs Doubly Reco Tracks')

        all_plot_labels = [str(i) for i in x_pos]  # example layer labels
        plt.xticks(x_pos, all_plot_labels, rotation=45, ha='right')

        # Add counts on top of each segment
        for i in x_pos:
            if fake_counts[i] > 0:
                plt.text(i, fake_counts[i]/2, str(fake_counts[i]), ha='center', va='center', color='black', fontsize=10)
            if doubly_reco_counts[i] > 0:
                plt.text(i, fake_counts[i] + doubly_reco_counts[i]/2, str(doubly_reco_counts[i]), ha='center', va='center', color='black', fontsize=10)
            if good_counts[i] > 0:
                plt.text(i, fake_counts[i] + doubly_reco_counts[i] + good_counts[i]/2, str(good_counts[i]), ha='center', va='center', color='black', fontsize=10)

        
        plt.tight_layout()
        pdf.savefig()
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw shared clusters")
    parser.add_argument("--input", type=str, default="CheckTracksCA.parquet", help="Input CheckTracksCA parquet file")
    parser.add_argument("--output", type=str, default="shared_clusters.pdf", help="Output file")
    args = parser.parse_args()

    draw_shared_clusters(args.input, args.output)