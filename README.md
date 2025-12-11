# ITS Shared Clusters Studies

## Simulation

Simulation (`make simulate-without` / `make simulate-with`)

Script: `run_simulations.sh`

| Target | Description | Key Mechanism |
|--------|-------------|----------------|
| `simulate-without` | Runs the full MC → RECO workflow without the cluster sharing feature. | Standard execution of `o2dpg_sim_workflow.py` |
| `copy-output` | Uses rsync to duplicate the MC output from the 'without' run to the 'with' run directory. | Ensures identical MC data for both RECO runs |
| `simulate-with` | Reruns the reconstruction (RECO) step only on the copied data, but with the shared clusters feature enabled. | Uses `--rerun-from itsreco_1` and the `--its-first-cluster-sharing` flag |

## Check
Check (`make check`)

Script: `run_check.sh`

Purpose: Performs the initial, per-time-frame analysis of tracking performance.

Action: Copies the `CheckTracksCA.C` ROOT macro into each time-frame (tf*) folder and executes it.

Output: Intermediate .root and .pdf files are staged in the `outputs/partial/` directory.

## Pre-processing

Pre-processing (`make preprocess`)

Script: `run_preprocess.sh`

Purpose: Preprocesses and converts the intermediate data into a parquet format.

Action: Runs the `preprocess.py` script which:
- Merges the multiple partial ROOT files for each variant.
- Applies some logic
    - `isGoodMother` (the shared cluster belongs to tracks that share the same MC mother particle)
    - `same_mc_track_id` (the shared cluster belongs to tracks reconstructed from the same MC particle)
- Runs the `matchItsAO2DTracks.cxx` macro to match ITS tracks to AOD tracks.
- Converts the merged data into a single parquet file.

## Analysis
Analysis (`make analysis`)

Script: `run_analysis.sh`

Purpose: Generates the final high-level comparison plots.

Action: Executes Python scripts:
- `draw_shared.py`: Study tracking metrics between the two runs.
- `study_doubly_reco.py`: Study tracks that are reconstructed multiple times.
