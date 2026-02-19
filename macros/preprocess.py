"""Convert CheckTracksCA output ROOT file to Parquet format for easier analysis."""

import argparse
import os
import sys
from pathlib import Path
import uproot
import numpy as np
import pandas as pd

sys.path.append(".")
from utils.data_matcher import (  # pylint: disable=wrong-import-position, import-error  # noqa: E402
    DataMatcher,
)

MC_COLS_TO_ADD = [
    "pdg",
    "pt",
    "eta",
    "phi",
    "motherTrackId",
    "motherTrackPdg",
    "process",
    "firstSharedLayer",
    "clusters",
    "isReco",
    "isFake",
    "isPrimary",
    "mcTrack.mStartVertexCoordinatesX",
    "mcTrack.mStartVertexCoordinatesY",
    "mcTrack.mStartVertexCoordinatesZ",
    "mcTrack.mStartVertexCoordinatesT",
]

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
    3122: r"\Lambda",
    3112: r"\Sigma^{\pm}",
    1000822080: "Pb",
}


def get_df_from_output(input_root):
    """Read the CheckTracksCA output ROOT file and return a pandas DataFrame with relevant info."""
    with uproot.open(input_root) as f:
        df = f["ParticleInfoReco"].arrays(  # pylint: disable=redefined-outer-name
            library="pd"
        )
        df_mc = f["ParticleInfo"].arrays(library="pd")
    data_matcher = DataMatcher(df, df_mc)

    df = data_matcher.add_mc_info(MC_COLS_TO_ADD)
    for col in df.columns:
        if df[col].dtype == "awkward":
            df[col] = df[col].apply(lambda x: x.tolist())

    return df

def get_df_matching(input_root):
    """Get the DataFrame after matching with AO2D."""
    with uproot.open(input_root) as f:
        df_matching = f["MatchingIndex"].arrays(  # pylint: disable=redefined-outer-name
            library="pd"
        )
    return df_matching

def make_label(row):
    """Create a label for the particle based on its PDG and mother's PDG."""
    pdg, mother = abs(row.pdg), abs(row.motherTrackPdg)
    if pdg in PDG_LABELS and mother in PDG_LABELS:
        return rf"${PDG_LABELS[pdg]} \leftarrow {PDG_LABELS[mother]}$"

    #print(f"Unknown PDG: {row.pdg}, Mother PDG: {row.motherTrackPdg}")
    return "others"


def get_cylindrical(df):  # pylint: disable=redefined-outer-name
    """Add cylindrical coordinates to the dataframe."""
    x = np.stack(df["clusterX[7]"].to_numpy())
    y = np.stack(df["clusterY[7]"].to_numpy())

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)

    df["clusterR[7]"] = list(r)
    df["clusterPhi[7]"] = list(phi)


def main(input_folder, output_file):  # pylint: disable=too-many-locals
    """Main function to convert ROOT to Parquet."""
    # Look for all ROOT files in the input folder
    input_folder = Path(input_folder)

    dfs = []
    for iteration_dir in input_folder.iterdir():
        if not iteration_dir.is_dir():
            continue

        files = list(iteration_dir.glob("*.root"))
        tfs = [int(f.name.split("tf")[-1].split(".root")[0]) for f in files]

        for file, tf in zip(files, tfs):
            dfs.append(get_df_from_output(file))

            outfile_root = output_file.replace(".parquet", f"tf{tf}.root")
            cols_matching = ["event", "pdg", "pt", "eta", "phi"]
            with uproot.recreate(outfile_root) as f:
                f["RecoTracks"] = dfs[-1][cols_matching]

            base_dir = input_folder.parent.parent.parent
            iteration = iteration_dir.name
            variant = input_folder.name
            dfs[-1]["tf"] = tf
            dfs[-1]["iteration"] = iteration

            ao2d_file = base_dir / variant / iteration / f"tf{tf}" / "AO2D.root"
            if not os.path.exists(ao2d_file):
                ao2d_file = base_dir / variant / iteration / f"tf{tf}" / "AO2D_with_tracksel.root"
            os.system(
                f"root -l -b -q 'macros/matchItsAO2DTracks.cxx+(\"{outfile_root}\", \"{ao2d_file}\", {tf})'"  # pylint: disable=line-too-long
            )
            dfs[-1] = pd.concat([dfs[-1], get_df_matching(outfile_root)], axis=1)
            os.remove(outfile_root)


    df = pd.concat(dfs, ignore_index=True)
    get_cylindrical(df)

    df["label"] = df[["pdg", "motherTrackPdg"]].apply(make_label, axis=1)

    df["isGoodMother"] = False

    # Only apply the logic to shared clusters
    shared_mask = (
        df["isShared"] == True  # pylint: disable=singleton-comparison  # noqa: E712
    )
    df.loc[shared_mask, "isGoodMother"] = (
        df[shared_mask]
        .groupby(["event", "motherTrackId", "tf", "iteration"])["motherTrackId"]
        .transform("count")
        > 1
    )

    df.loc[:, "same_mc_track_id"] = (
        df.groupby(["event", "mcTrackID", "tf", "iteration"])["mcTrackID"].transform("count") > 1
    )

    n_layers = 7
    df["layers_hits"] = df["clusters"].apply(
        lambda x: [(x >> i) & 1 == 1 for i in range(n_layers)]
    )

    df["n_hits"] = df["layers_hits"].apply(sum)

    df.to_parquet(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CheckTracksCA output ROOT file to Parquet format."
    )
    parser.add_argument(
        "input", type=str, help="Input folder with CheckTracksCA output ROOT files"
    )
    parser.add_argument("output", type=str, help="Output Parquet file")
    args = parser.parse_args()

    main(args.input, args.output)
