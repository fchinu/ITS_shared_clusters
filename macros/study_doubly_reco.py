import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from matplotlib.backends.backend_pdf import PdfPages

def draw_phi_distribution(df_all, df_doubly, df_not_doubly):
    fig = plt.figure(figsize=(10,6))
    plt.hist(df_all["phi"].explode(), bins=100, alpha=0.5, log=True, label='All Tracks')
    plt.hist(df_not_doubly["phi"].explode(), bins=100, alpha=0.7, log=True, label='Not Doubly Reco Tracks')
    plt.hist(df_doubly["phi"].explode(), bins=100, alpha=0.9, log=True, label='Doubly Reco Tracks')
    plt.xlabel('Phi (radians)')
    plt.ylabel('Counts')
    plt.title('Distribution of Track Phi')
    plt.legend()
    
    return fig

def draw_dr_dphi_dz(df_doubly, df_not_doubly):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    not_shared_tracks = df_not_doubly.groupby(['event'])
    for event, group in not_shared_tracks:
        if len(group) < 2:
            continue

    dr_list, dphi_list, dz_list = [], [], []

    for event, group in df_not_doubly.groupby("event"):
        if len(group) < 2:
            continue

        r = np.stack(group["clusterR[7]"].to_numpy())
        phi = np.stack(group["clusterPhi[7]"].to_numpy())
        z = np.stack(group["clusterZ[7]"].to_numpy())

        # Replace ~0 values with NaN
        for arr in (r, phi, z):
            arr[np.isclose(arr, 0, atol=1e-8)] = np.nan

        dr = np.abs(r[:, None, :] - r[None, :, :])
        dphi = np.abs(phi[:, None, :] - phi[None, :, :])
        dz = np.abs(z[:, None, :] - z[None, :, :])

        iu = np.triu_indices(len(group), k=1)

        dr_list.append(dr[iu].ravel())
        dphi_list.append(dphi[iu].ravel())
        dz_list.append(dz[iu].ravel())

    dr = np.concatenate(dr_list)
    dphi = np.concatenate(dphi_list)
    dz = np.concatenate(dz_list)

    axs[0, 0].hist(dr, bins=100, alpha=0.5, label='Not Doubly Reco Tracks', color='orange', density=True, log=True)
    axs[0, 1].hist(dphi, bins=100, alpha=0.5, label='Not Doubly Reco Tracks', color='orange', density=True, log=True)
    axs[0, 2].hist(dz, bins=100, alpha=0.5, label='Not Doubly Reco Tracks', color='orange', density=True, log=True)
    axs[1, 0].hist(df_doubly['n_hits'], bins=100, alpha=0.5, label='Not Doubly Reco Tracks', color='orange', density=True, log=True)
    axs[1, 1].hist(df_doubly['pt'], bins=100, alpha=0.5, label='Not Doubly Reco Tracks', color='orange', density=True, log=True)

    shared_tracks = df_doubly.groupby(['mcTrackID', 'event'])
    dr_list, dphi_list, dz_list = [], [], []
    for (mc_id, event), group in shared_tracks:
        if len(group) < 2:
            continue

        r = np.stack(group["clusterR[7]"].to_numpy())
        phi = np.stack(group["clusterPhi[7]"].to_numpy())
        z = np.stack(group["clusterZ[7]"].to_numpy())
        
        for arr in (r, phi, z):
            arr[np.isclose(arr, 0, atol=1e-8)] = np.nan

        dr_list.append(np.abs(r[1] - r[0]))
        dphi_list.append(np.abs(phi[1] - phi[0]))
        dz_list.append(np.abs(z[1] - z[0]))
    
    dr = np.concatenate(dr_list)
    dphi = np.concatenate(dphi_list)
    dz = np.concatenate(dz_list)

    axs[0, 0].hist(dr, bins=100, alpha=0.5, label='Doubly Reco Tracks', color='blue', density=True, log=True)
    axs[0, 1].hist(dphi, bins=100, alpha=0.5, label='Doubly Reco Tracks', color='blue', density=True, log=True)
    axs[0, 2].hist(dz, bins=100, alpha=0.5, label='Doubly Reco Tracks', color='blue', density=True, log=True)
    axs[1, 0].hist(df_not_doubly['n_hits'], bins=100, alpha=0.5, label='Doubly Reco Tracks', color='blue', density=True, log=True)
    axs[1, 1].hist(df_not_doubly['pt'], bins=100, alpha=0.5, label='Doubly Reco Tracks', color='blue', density=True, log=True)


    axs[0, 0].set_xlabel(r'$\Delta$ r (cm)')
    axs[0, 0].set_ylabel('Counts')
    axs[0, 1].set_xlabel(r'$\Delta \phi$ (radians)')
    axs[0, 1].set_ylabel('Counts')
    axs[0, 2].set_xlabel(r'$\Delta$ z (cm)')
    axs[0, 2].set_ylabel('Counts')
    axs[1, 0].set_xlabel('n clusters')
    axs[1, 0].set_ylabel('Counts')
    axs[1, 1].set_xlabel(r'$p_\mathrm{T}$ (GeV/c)')
    axs[1, 1].set_ylabel('Counts')
    axs[1, 1].legend()
    plt.tight_layout()
    return fig

def study_doubly_reco(input_parquet, output_file):
    df = pd.read_parquet(input_parquet)

    df_doubly_reco = df.query("same_mc_track_id")
    df_not_doubly_reco = df.query("not same_mc_track_id")

    with PdfPages(output_file) as pdf:
        fig_phi = draw_phi_distribution(df, df_doubly_reco, df_not_doubly_reco)
        pdf.savefig(fig_phi)
        plt.close(fig_phi)

        fig_dr_dphi_dz = draw_dr_dphi_dz(df_doubly_reco, df_not_doubly_reco)
        pdf.savefig(fig_dr_dphi_dz)
        plt.close(fig_dr_dphi_dz)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Study doubly reconstructed tracks")
    parser.add_argument("input", type=str, help="Input Parquet file")
    parser.add_argument("output", type=str, help="Output file")
    args = parser.parse_args()

    study_doubly_reco(args.input, args.output)
