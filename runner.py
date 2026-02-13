import subprocess
import argparse
import os
from pathlib import Path
import shutil
import sys
import time
import json
import logging

class Runner():
    """Class to handle simulations."""

    def __init__(self, **kwargs):
        self.base_name = kwargs["name"]
        self.iterations = kwargs["n"]
        self.n_workers = kwargs["nworkers"]
        self.n_timeframes = kwargs["ntimeframes"]
        self.n_sigevents = kwargs["nsigevents"]
        self.start_splitid = kwargs["splitid"]
        self.sim_engine = kwargs["simengine"]
        self.is_pp = kwargs["is_pp"]
        self.low_field = kwargs["low_field"]
        self.start_from = kwargs["start_from"]
        self.rerun_failed = kwargs["rerun_failed"]
        self.run_iterations = kwargs["run_iterations"]

        self.base_dir = Path("simulations") / self.base_name
        self.sim_dir = self.base_dir / "without_shared_clusters"
        self.delta_rof_dir = self.base_dir / "without_shared_clusters_delta_rof"
        self.w_shared_cl_dir = self.base_dir / "with_shared_clusters"
        self.w_shared_cl_delta_rof_dir = self.base_dir / "with_shared_clusters_delta_rof"
        self.output_dir = self.base_dir / "outputs"
        self.partial_output_dir = self.output_dir / "partial"

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)


        self.setup()
        self.run()
        # We try to rerun failed simulations
        # if not self.rerun_failed:
        #     self.rerun_failed = True
        #     self.run()

    def setup(self):
        """Setup function to prepare the environment for running simulations."""

        # Create simulation directory
        self.sim_dir.mkdir(parents=True, exist_ok=True)
        self.w_shared_cl_dir.mkdir(parents=True, exist_ok=True)
        self.delta_rof_dir.mkdir(parents=True, exist_ok=True)
        self.w_shared_cl_delta_rof_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / self.sim_dir.name).mkdir(parents=True, exist_ok=True)
        (self.output_dir / self.w_shared_cl_dir.name).mkdir(parents=True, exist_ok=True)
        (self.output_dir / self.delta_rof_dir.name).mkdir(parents=True, exist_ok=True)
        (self.output_dir / self.w_shared_cl_delta_rof_dir.name).mkdir(parents=True, exist_ok=True)
        (self.partial_output_dir / self.sim_dir.name).mkdir(parents=True, exist_ok=True)
        (self.partial_output_dir / self.w_shared_cl_dir.name).mkdir(parents=True, exist_ok=True)
        (self.partial_output_dir / self.delta_rof_dir.name).mkdir(parents=True, exist_ok=True)
        (self.partial_output_dir / self.w_shared_cl_delta_rof_dir.name).mkdir(parents=True, exist_ok=True)
        
        iterations_to_run = self.run_iterations if self.run_iterations is not None else range(self.iterations)
        for i in iterations_to_run:
            iter_dir = self.sim_dir / str(i)
            iter_dir.mkdir(parents=True, exist_ok=True)
            (self.partial_output_dir / self.sim_dir.name / str(i)).mkdir(parents=True, exist_ok=True)
            (self.partial_output_dir / self.w_shared_cl_dir.name / str(i)).mkdir(parents=True, exist_ok=True)
            (self.partial_output_dir / self.delta_rof_dir.name / str(i)).mkdir(parents=True, exist_ok=True)
            (self.partial_output_dir / self.w_shared_cl_delta_rof_dir.name / str(i)).mkdir(parents=True, exist_ok=True)

        # Create source file with environment variables for reproducibility
        source_file = self.base_dir / "source.sh"
        with open(source_file, "w") as f:
            f.write("#!/bin/bash\n\n")
            f.write("export NAME=" + self.base_name + "\n")
            f.write("export SPLITID=" + str(self.start_splitid) + "\n")
            f.write("export NITERATIONS=" + str(self.iterations) + "\n")
            f.write("export NWORKERS=" + str(self.n_workers) + "\n")
            f.write("export NTIMEFRAMES=" + str(self.n_timeframes) + "\n")
            f.write("export NSIGEVENTS=" + str(self.n_sigevents) + "\n")
            f.write("export SIMENGINE=" + self.sim_engine + "\n")
            f.write("export IS_PP=" + str(int(self.is_pp)) + "\n")
            f.write("export LOW_FIELD=" + str(int(self.low_field)) + "\n")

    def run(self):
        """Run the simulations for the specified number of iterations."""
        stages = ["sim", "copy", "reco_shared", "tracksel", "check", "preprocess", "analysis"]
        start_idx = stages.index(self.start_from) if not self.rerun_failed else 0

        iterations_to_run = self.run_iterations if self.run_iterations is not None else range(1, self.iterations)
        for i in iterations_to_run:
            rerun_from = None
            if self.rerun_failed:
                with open(self.sim_dir / str(i) / "simulation.log", "r") as log_file:
                    log_content = log_file.read()
                    lines = log_content.split('\n')
                    failed_line = next((line for line in lines if "failed ... checking retry" in line), None)
                    if failed_line:
                        failed_task = failed_line.split("failed ... checking retry")[0].strip().split("_")[0]
                        self.logger.info(failed_line)
                        rerun_from = f"{failed_task}_*"
                        self.logger.info(f"Rerunning failed simulation for iteration {i} from {failed_task}*...")
                        shutil.rmtree((self.w_shared_cl_dir / str(i)).absolute())
                    else:
                        self.logger.info(f"No failed tasks found for iteration {i}. Skipping rerun.")
                        continue

            try:
                if start_idx <= 0:
                    self.run_simulation(i, self.start_splitid + i, rerun_from=rerun_from)
                if start_idx <= 1:
                    self.copy_simulations(self.w_shared_cl_dir, i)
                    self.copy_simulations(self.delta_rof_dir, i)
                    self.copy_simulations(self.w_shared_cl_delta_rof_dir, i)
                if start_idx <= 2:
                    self.run_its_reco(i, shared_cl=True, delta_rof=False)
                    self.run_its_reco(i, shared_cl=False, delta_rof=True)
                    self.run_its_reco(i, shared_cl=True, delta_rof=True)
                # if start_idx <= 3:
                #     self.extend_with_track_selection(i)
                if start_idx <= 4:
                    self.run_checktracksca(i)  # TODO: parallelize
            except Exception as e:
                self.logger.error(f"An error occurred in iteration {i}: {e}")

        if start_idx <= 5:
            self.preprocess()
        if start_idx <= 6:
            self.draw_shared()

    def _get_all_tf_dirs(self, base_directory: Path):
        """Get all timeframe directories in the base directory."""
        tf_dirs = []
        for item in base_directory.iterdir():
            if item.is_dir() and "tf" in item.name:
                tf_dirs.append(item)
        return tf_dirs

    def run_simulation(
        self,
        iteration: int,
        seed_value,
        rerun_from=None 
    ):
        """
        Executes the O2DPG simulation workflow (MC->RECO->AOD).
        """
        o2dpg_root = os.environ.get("O2DPG_ROOT")
        o2_root = os.environ.get("O2_ROOT")

        if not o2dpg_root or not o2_root:
            self.logger.error("Error: O2DPG_ROOT and O2_ROOT must be set in the environment.")
            return False

        # load utility functions from O2
        try:
            subprocess.run(["bash", "-c", f". {o2_root}/share/scripts/jobutils.sh"], check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Could not source O2 jobutils.sh: {e}")
            return False
        
        dir = self.sim_dir / str(iteration)

        self.logger.info("=== Simulation Parameters ===")
        self.logger.info(f"OUTPUT_DIR: {dir}")
        self.logger.info(f"NWORKERS: {self.n_workers}")
        self.logger.info(f"NSIGEVENTS: {self.n_sigevents}")
        self.logger.info(f"NTIMEFRAMES: {self.n_timeframes}")
        self.logger.info(f"SIMENGINE: {self.sim_engine}")
        self.logger.info(f"LOW_FIELD: {self.low_field}")
        self.logger.info(f"SEED: {seed_value}")
        self.logger.info("==========================")

        workflow_gen_cmd = [
            f"{o2dpg_root}/MC/bin/o2dpg_sim_workflow.py",
            "-j", str(self.n_workers),
            "-ns", str(self.n_sigevents),
            "-tf", str(self.n_timeframes),
            "-interactionRate", "500000",
            "-confKey", "Diamond.width[2]=6.",
            "-e", self.sim_engine,
            "-seed", str(seed_value),
            # "-mod", "--skipModules ZDC"  # Note: Passed as a single string argument for argparse
        ]

        # Collision Logic
        if self.is_pp:
            workflow_gen_cmd.extend(["-eCM", "13600", "-col", "pp", "-gen", "pythia8pp"])
        else:
            workflow_gen_cmd.extend(["-eCM", "5360", "-col", "PbPb", "-gen", "pythia8"])

        # Low Field Logic
        if self.low_field:
            workflow_gen_cmd.extend(["-field", "2"])

        try:
            self.logger.info("Generating simulation workflow...")
            # We run this inside work_dir so workflow.json is created there
            with open(dir / "simulation.log", "w") as log_file:
                subprocess.run(workflow_gen_cmd, cwd=dir, check=True, stdout=log_file, stderr=log_file)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Workflow generation failed with exit code {e.returncode}")
            return False

        # Check if workflow.json exists
        if not (dir / "workflow.json").exists():
            self.logger.error("Error: workflow.json not generated")
            return False

        # Construct Workflow Execution Command
        runner_cmd = [
            f"{o2dpg_root}/MC/bin/o2_dpg_workflow_runner.py",
            "-f", "workflow.json",
            "-tt", "aod",
            "--cpu-limit", "32"
        ]

        if rerun_from:
            workflow_gen_cmd.extend(["--rerun-from", rerun_from])

        # if shared_clusters:
        #     # If shared clusters, we rely on the specific rerun logic
        #     runner_cmd.extend(["--rerun-from", "itsreco_*"])

        # Run Simulation
        try:
            self.logger.info("Running simulation workflow...")
            with open(dir / "simulation.log", "a") as log_file:
                subprocess.run(runner_cmd, cwd=dir, check=True, stdout=log_file, stderr=log_file)
            self.logger.info(f"Simulation script completed successfully in: {dir}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Workflow execution failed with exit code {e.returncode}")
            return False

    def copy_simulations(self, to: Path, iteration: int = -1):
        """Copy simulation results to another location."""
        try:
            if iteration == -1:
                subprocess.run(["cp", "-r", f"{self.sim_dir}/.", str(to)], check=True)
                self.logger.info(f"Copied simulations to {to}")
            else:
                subprocess.run(["cp", "-r", f"{self.sim_dir}/{iteration}/.", str(to / str(iteration))], check=True)
                self.logger.info(f"Copied simulations to {to / str(iteration)}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to copy simulations: {e}")

    def _get_last_modified_time(self, directory: Path) -> float:
        """Get the last modified time of files in a directory."""
        latest_time = 0.0
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                mod_time = file_path.stat().st_mtime
                if mod_time > latest_time:
                    latest_time = mod_time
        return latest_time

    def _link_missing_files(self, source_dir: Path, target_dir: Path):
        source_path = source_dir.resolve()
        target_path = target_dir.resolve()

        for item in source_path.rglob("*"):
            if item.is_dir():
                continue
            
            relative_path = item.relative_to(source_path)
            link_destination = target_path / relative_path

            if not link_destination.exists():
                try:
                    # Create the parent directory structure (e.g., tf1/) 
                    # as real folders so we can put links inside them.
                    link_destination.parent.mkdir(parents=True, exist_ok=True)
                    link_destination.symlink_to(item)
                except Exception as e:
                    self.logger.error(f"Failed to link {relative_path}: {e}")
                    return False
        return True

    def run_its_reco(self, iteration: int, shared_cl: bool = True, delta_rof: bool = False):
        """Run ITS reconstruction"""

        if not (shared_cl or delta_rof):
            self.logger.info("Please specify at least one of shared_cl or delta_rof to True for running ITS reconstruction.")
            sys.exit()

        o2dpg_root = os.environ.get("O2DPG_ROOT")
        o2_root = os.environ.get("O2_ROOT")

        if not o2dpg_root or not o2_root:
            self.logger.error("Error: O2DPG_ROOT and O2_ROOT must be set in the environment.")
            return False

        log_text = "=== Running ITS Reconstruction "
        if shared_cl:
            log_text += "with Shared Clusters "
        if delta_rof:
            log_text += "with deltaRof parameters "
        log_text += "==="
        self.logger.info(log_text)

        dir = None
        if shared_cl and delta_rof:
            dir = self.w_shared_cl_delta_rof_dir / str(iteration)
        elif shared_cl:
            dir = self.w_shared_cl_dir / str(iteration)
        elif delta_rof:
            dir = self.delta_rof_dir / str(iteration)
        else:
            dir = self.sim_dir / str(iteration)

        latest_mod_time = self._get_last_modified_time(dir)

        # Modify workflow to enable shared clusters
        workflow_file = dir / "workflow.json"
        with open(workflow_file, "r") as f:
            workflow = json.load(f)
        for task in workflow["stages"]:
            if "itsreco" in task["name"]:
                # add
                new_params = []
                if shared_cl:
                    new_params.append("ITSCATrackerParam.allowSharingFirstCluster=true")
                if delta_rof:
                    new_params.append("ITSCATrackerParam.deltaRof=1")
                    new_params.append("ITSVertexerParam.deltaRof=1")
                new_param = ";".join(new_params)
                parts = task["cmd"].split('\"')

                # parts[0] is the start, parts[1] is the content of configKeyValues
                if new_param not in parts[1]:
                    # Add a semicolon if the string isn't empty
                    separator = ";" if parts[1] and not parts[1].endswith(";") else ""
                    parts[1] = f"{parts[1]}{separator}{new_param}"

                    # Reconstruct the command
                    task["cmd"] = '\"'.join(parts)

        with open(workflow_file, "w") as f:
            json.dump(workflow, f, indent=4)

        # Construct Workflow Execution Command
        runner_cmd = [
            f"{o2dpg_root}/MC/bin/o2_dpg_workflow_runner.py",
            "-f", "workflow.json",
            "-tt", "itsreco",
            "--cpu-limit", "32",
            "--rerun-from", "itsreco_*"
        ]

        # Run ITS Reconstruction with Shared Clusters
        try:
            self.logger.info("Running ITS reconstruction with shared clusters...")
            with open(dir / "its_reco_shared_clusters.log", "w") as log_file:
                subprocess.run(runner_cmd, cwd=dir, check=True, stdout=log_file, stderr=log_file)
            log_text = "=== ITS Reconstruction "
            if shared_cl:
                log_text += "with Shared Clusters "
            if delta_rof:
                log_text += "with deltaRof parameters "
            log_text += f"completed successfully in: {dir} ==="
            self.logger.info(log_text)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"ITS reconstruction failed with exit code {e.returncode}")
            return False

        # Remove old files
        for root, _, files in os.walk(dir):
            for file in files:
                file_path = Path(root) / file
                if file_path.stat(follow_symlinks=False).st_mtime <= latest_mod_time:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        self.logger.error(f"Failed to remove file {file_path}: {e}")
                        return False

        return self._link_missing_files(self.sim_dir / str(iteration), dir)

    def extend_with_track_selection(self, iteration: int):
        """Extend AOD with track selection information using native Python sub-processes."""
        for dir_sim in self._get_all_tf_dirs(self.sim_dir / str(iteration)) + self._get_all_tf_dirs(self.w_shared_cl_dir / str(iteration)):
            ao2d_root = "AO2D.root"
            ao2d_old = "AO2D_old.root"
            
            self.logger.info(f"Processing track selection in: {dir_sim}")

            if (dir_sim / ao2d_root).exists() and not (dir_sim / ao2d_old).exists():
                (dir_sim / ao2d_root).rename(dir_sim / ao2d_old)

            config_path = Path(__file__).parent / "configs" / "configuration_track_sel.json"
            output_director_path = Path(__file__).parent / "configs" / "OutputDirector.json"
            log_path = dir_sim / "log_track_selection.txt"
            config_param = f"json://{str(config_path)}"
            
            common_args = f"-b --configuration {config_param}"
            
            cmd = f"""o2-analysis-propagationservice {common_args} | \
            o2-analysis-event-selection-service {common_args} | \
            o2-analysis-trackselection {common_args} \
                --aod-file AO2D_old.root \
                --aod-writer-json {output_director_path} \
                --shm-segment-size 3000000000 \
                --aod-parent-access-level 1
            """
            try:
                with open(log_path, "w") as log_file:
                    subprocess.run(cmd, shell=True, stdout=log_file, stderr=log_file, cwd=dir_sim)

                # Merge results with hadd
                merged_file = "AO2D_with_tracksel.root"
                hadd_cmd = ["hadd", "-f", str(merged_file), str(ao2d_old), str(ao2d_root)]
                
                self.logger.info("Merging AOD files...")
                subprocess.run(hadd_cmd, check=True, cwd=dir_sim)

                (dir_sim / ao2d_root).unlink(missing_ok=True)
                (dir_sim / "AnalysisResults.root").unlink(missing_ok=True)

                self.logger.info("Track selection extension completed successfully.")
                return True

            except subprocess.CalledProcessError as e:
                self.logger.error(f"Command failed during execution: {e}")
                return False
            except Exception as e:
                self.logger.error(f"An unexpected error occurred: {e}")
                return False

    def run_checktracksca(self, iteration: int):
        """Run checkTrackSca on the AOD files."""
        for dir in (self.sim_dir, self.w_shared_cl_dir):
            for dir_sim in self._get_all_tf_dirs(dir / str(iteration)):
                tf = dir_sim.name.split("tf")[-1]
                ao2d_root = dir_sim / "AO2D.root"
                log_path = dir_sim / "log_checkTracksCA.txt"
                script = Path("CheckTracksCA.C")

                if not ao2d_root.exists():
                    ao2d_root = dir_sim / "AO2D_with_tracksel.root"
                    if not ao2d_root.exists():
                        self.logger.error(f"No AOD file found for CheckTracksCA in {dir_sim}")
                        return False

                cmd = [
                    "root", "-l", "-b", "-q",
                    f"{script.absolute()}(true, true, false, true, true, \"o2trac_its.root\", \"o2sim\", \"o2clus_its.root\", \"sgn_Kine.root\")"
                ]

                try:
                    with open(log_path, "w") as log_file:
                        subprocess.run(cmd, check=True, stdout=log_file, stderr=log_file, cwd=dir_sim)
                    self.logger.info(f"CheckTracksCA completed successfully in: {dir_sim}")

                except subprocess.CalledProcessError as e:
                    self.logger.error(f"CheckTracksCA failed with exit code {e.returncode} in {dir_sim}")
                    return False

                # Move output files to the outputs directory
                output_subdir = self.partial_output_dir / dir.name / str(iteration)
                try:
                    subprocess.run(["mv", str(dir_sim / "fakeClusters.pdf"), str(output_subdir / f"fakeClusters_tf{tf}.pdf")], check=True)
                    subprocess.run(["mv", str(dir_sim / "CheckTracksCA.root"), str(output_subdir / f"CheckTracksCA_tf{tf}.root")], check=True)
                    self.logger.info(f"Moved CheckTracksCA output files from {dir_sim} to {output_subdir}")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"Failed to move CheckTracksCA output files: {e}")
                    return False

        return True
    
    def preprocess(self):
        """
        Python version of the bash preprocess function.
        Handles venv execution by isolating environment variables.
        """
        venv_path = Path(f"{os.getenv('HOME')}/.virtualenv/ml312")
        python_bin = venv_path / "bin" / "python3"
        script_path = Path(__file__).parent / "macros" / "preprocess.py"

        # We copy the current env and remove variables that interfere with venvs
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)

        for dir in (self.sim_dir, self.w_shared_cl_dir):
            input_subdir = self.partial_output_dir / dir.name
            output_file = self.output_dir / dir.name / f"CheckTracksCA.parquet"

            try:
                result = subprocess.run(
                    [str(python_bin), str(script_path), str(input_subdir), str(output_file)],
                    env=env,
                    check=True,
                    text=True
                )
            except subprocess.CalledProcessError as e:
                self.logger.error(f"Failed to move preprocess files in {input_subdir}: {e}")
                return False

        return True

    def draw_shared(self):
        """
        Python version of the bash preprocess function.
        Handles venv execution by isolating environment variables.
        """
        venv_path = Path(f"{os.getenv('HOME')}/.virtualenv/ml312")
        python_bin = venv_path / "bin" / "python3"
        script_path = Path(__file__).parent / "macros" / "draw_shared.py"

        # We copy the current env and remove variables that interfere with venvs
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)

        output_file = self.output_dir / "analysis_output.pdf"

        try:
            result = subprocess.run(
                [str(python_bin), str(script_path), str(self.base_dir), str(output_file)],
                env=env,
                check=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to run draw_shared.py : {e}")
            return False

        return True

if __name__ == "__main__":
    name = os.getenv("NAME", "shared_clusters_test")
    splitid = int(os.getenv("SPLITID", 42))
    n_iterations = int(os.getenv("NITERATIONS", 4))
    n_workers = int(os.getenv("NWORKERS", 4))
    n_timeframes = int(os.getenv("NTIMEFRAMES", 10))
    n_sigevents = int(os.getenv("NSIGEVENTS", 1000))
    sim_engine = os.getenv("SIMENGINE", "TGeant4")
    is_pp = bool(int(os.getenv("IS_PP", 1)))
    low_field = bool(int(os.getenv("LOW_FIELD", 0)))

    parser = argparse.ArgumentParser(description="Batch run Makefile simulations using Env Vars.")
    parser.add_argument("--name", type=str, default=name,
                        help=f"Base name (Current Env: {name})")
    parser.add_argument("--splitid", type=int, default=splitid,
                        help=f"Starting SplitID (Current Env: {splitid})")
    parser.add_argument("--n", type=int, default=n_iterations,
                        help="Number of iterations to run")
    parser.add_argument("--nworkers", type=int, default=n_workers,
                        help=f"Number of workers (Current Env: {n_workers})")
    parser.add_argument("--ntimeframes", type=int, default=n_timeframes,
                        help=f"Number of timeframes (Current Env: {n_timeframes})")
    parser.add_argument("--nsigevents", type=int, default=n_sigevents,
                        help=f"Number of signal events (Current Env: {n_sigevents})")
    parser.add_argument("--simengine", type=str, default=sim_engine,
                        help=f"Simulation engine (Current Env: {sim_engine})")
    parser.add_argument("--is-pp", type=int, default=int(is_pp),
                        help=f"Is pp collision (Current Env: {is_pp})")
    parser.add_argument("--low-field", type=int, default=int(low_field),
                        help=f"Low magnetic field (Current Env: {low_field})")
    parser.add_argument("--start-from", type=str, 
                        choices=["sim", "copy", "reco_shared", "tracksel", "check", "preprocess", "analysis"], 
                        default="reco_shared",
                        help="Stage to start from: sim, copy, reco_shared, tracksel, check, preprocess, or analysis")
    parser.add_argument("--rerun-failed", action="store_true",
                        help="Rerun only failed stages. If set, overrides --start-from to 'sim'.")
    parser.add_argument("--run-iterations", nargs="+", type=int, default=None,
                        help="Run only selected iterations. Provide a list of iteration numbers (e.g., --run-iterations 0 2 4)")
    args = parser.parse_args()

    runner = Runner(
        name=args.name,
        n=args.n,
        nworkers=args.nworkers,
        ntimeframes=args.ntimeframes,
        nsigevents=args.nsigevents,
        splitid=args.splitid,
        simengine=args.simengine,
        is_pp=args.is_pp,
        low_field=args.low_field,
        start_from=args.start_from,
        rerun_failed=args.rerun_failed,
        run_iterations=args.run_iterations
    )