"""Convert CheckTracksCA output ROOT file to Parquet format for easier analysis."""
import argparse
import uproot
import numpy as np
from data_matcher import DataMatcher

MC_COLS_TO_ADD = [
    "pdg", "pt", "eta", "phi", "motherTrackId", "motherTrackPdg",
    "process", "firstSharedLayer", "clusters", "isReco", "isFake", "isPrimary"
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
    3112: r"\Sigma^{\pm}",
    1000822080: "Pb"
}


def get_df_from_output(input_root):

    with uproot.open(input_root) as f:
        df = f["ParticleInfoReco"].arrays(library="pd")
        df_mc = f["ParticleInfo"].arrays(library="pd")
    data_matcher = DataMatcher(df, df_mc)

    df = data_matcher.add_mc_info(MC_COLS_TO_ADD)
    for col in df.columns:
        if df[col].dtype == "awkward":
            df[col] = df[col].apply(lambda x: x.tolist())

    return df


def make_label(row):
    pdg, mother = abs(row.pdg), abs(row.motherTrackPdg)
    if pdg in PDG_LABELS and mother in PDG_LABELS:
        return fr"${PDG_LABELS[pdg]} \leftarrow {PDG_LABELS[mother]}$"
    else:
        print(f"Unknown PDG: {row.pdg}, Mother PDG: {row.motherTrackPdg}")
        return "altro"
    
def get_cylindrical(df):
    print(df.columns)
    X = np.stack(df['clusterX[7]'].to_numpy())
    Y = np.stack(df['clusterY[7]'].to_numpy())
    Z = np.stack(df['clusterZ[7]'].to_numpy())

    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)

    df['r'] = list(R)
    df['phi'] = list(PHI)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CheckTracksCA output ROOT file to Parquet format.")
    parser.add_argument("input", type=str, help="Input ROOT file from CheckTracksCA")
    parser.add_argument("output", type=str, help="Output Parquet file")
    args = parser.parse_args()

    df = get_df_from_output(args.input)
    get_cylindrical(df)

    df["label"] = df[["pdg", "motherTrackPdg"]].apply(make_label, axis=1)

    df["isGoodMother"] = False

    # Only apply the logic to shared clusters
    shared_mask = df["isShared"] == True
    df.loc[shared_mask, "isGoodMother"] = (
        df[shared_mask].groupby(["event", "motherTrackId"])["motherTrackId"]
        .transform("count") > 1
    )

    df.loc[:, "same_mc_track_id"] = (
        df.groupby(["event", "mcTrackID"])["mcTrackID"]
        .transform("count") > 1
    )

    df.to_parquet(args.output)