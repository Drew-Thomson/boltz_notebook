import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


NOTEBOOK = Path(__file__).resolve().parents[1] / "Boltz_Colab.ipynb"


def load_cells():
    notebook = json.loads(NOTEBOOK.read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.cells = load_cells()
        self.namespace = {}
        exec(self.cells[3], self.namespace)  # form
        exec(self.cells[5], self.namespace)  # command builder

    def test_cells_are_valid_python_and_clean(self):
        notebook = json.loads(NOTEBOOK.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(NOTEBOOK), "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])

    def test_cli_options_and_safe_arguments(self):
        options = dict(self.namespace)
        options.update(
            checkpoint="/content/a file;echo bad.ckpt",
            seed="0",
            write_full_pae=True,
            no_kernels=True,
            preprocessing_threads="4",
        )
        command = self.namespace["build_command"]("/content/input with spaces.yaml", options)
        self.assertEqual(command[:3], ["boltz", "predict", "/content/input with spaces.yaml"])
        self.assertEqual(command[command.index("--checkpoint") + 1], "/content/a file;echo bad.ckpt")
        self.assertEqual(command[command.index("--seed") + 1], "0")
        self.assertEqual(command[command.index("--preprocessing-threads") + 1], "4")
        self.assertIn("--write_full_pae", command)
        self.assertIn("--no_kernels", command)
        self.assertNotIn("--cache", command)
        self.assertNotIn("--write_full_pde", command)

    def test_all_predict_options_are_exposed(self):
        options = dict(self.namespace)
        for name in (
            "cache", "checkpoint", "affinity_checkpoint", "step_scale", "seed",
            "msa_server_username", "api_key_header", "method", "preprocessing_threads",
        ):
            options[name] = "value"
        for name in (
            "write_full_pae", "write_full_pde", "override", "use_msa_server",
            "use_potentials", "affinity_mw_correction", "subsample_msa", "no_kernels",
            "write_embeddings",
        ):
            options[name] = True
        command = self.namespace["build_command"]("input.yaml", options)
        flags = {part for part in command if part.startswith("--")}
        expected = {
            "--out_dir", "--cache", "--checkpoint", "--devices", "--accelerator",
            "--recycling_steps", "--sampling_steps", "--diffusion_samples",
            "--max_parallel_samples", "--step_scale", "--write_full_pae", "--write_full_pde",
            "--output_format", "--num_workers", "--override", "--seed", "--use_msa_server",
            "--msa_server_url", "--msa_pairing_strategy", "--msa_server_username",
            "--api_key_header", "--use_potentials", "--model", "--method",
            "--preprocessing-threads", "--affinity_mw_correction", "--sampling_steps_affinity",
            "--diffusion_samples_affinity", "--affinity_checkpoint", "--max_msa_seqs",
            "--subsample_msa", "--num_subsampled_msa", "--no_kernels", "--write_embeddings",
        }
        self.assertEqual(flags, expected)

    def run_prediction(self, root, log, returncode=0, auth="none"):
        root = Path(root)
        input_path = root / "input.yaml"
        input_path.write_text("sequences: []\n")
        self.namespace.update(
            input_path=input_path,
            out_dir=str(root),
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
            subprocess=subprocess,
            Path=Path,
            msa_auth=auth,
        )
        if auth == "basic":
            self.namespace["msa_server_username"] = "user"
        process = SimpleNamespace(stdout=iter(log), wait=lambda: returncode)
        with patch("subprocess.Popen", return_value=process) as popen, \
             patch("getpass.getpass", return_value="secret"), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            exec(self.cells[6], self.namespace)
        return popen.call_args, output.getvalue()

    def test_success_and_secret_handling(self):
        with tempfile.TemporaryDirectory() as root:
            prediction_dir = Path(root) / "boltz_results_input" / "predictions" / "input"
            prediction_dir.mkdir(parents=True)
            (prediction_dir / "input_model_0.cif").write_text("structure")
            args, output = self.run_prediction(root, ["Number of failed examples: 0\n"], auth="basic")
            self.assertEqual(args.kwargs["env"]["BOLTZ_MSA_PASSWORD"], "secret")
            self.assertNotIn("secret", args.args[0])
            self.assertNotIn("secret", output)
            self.assertEqual(len(self.namespace["structures"]), 1)

    def test_api_key_is_passed_through_environment(self):
        with tempfile.TemporaryDirectory() as root:
            prediction_dir = Path(root) / "boltz_results_input" / "predictions" / "input"
            prediction_dir.mkdir(parents=True)
            (prediction_dir / "input_model_0.cif").write_text("structure")
            args, output = self.run_prediction(root, [], auth="api_key")
            self.assertEqual(args.kwargs["env"]["MSA_API_KEY_VALUE"], "secret")
            self.assertNotIn("secret", args.args[0])
            self.assertNotIn("secret", output)

    def test_preprocessing_failure_does_not_accept_old_results(self):
        with tempfile.TemporaryDirectory() as root:
            prediction_dir = Path(root) / "boltz_results_input" / "predictions" / "input"
            prediction_dir.mkdir(parents=True)
            (prediction_dir / "input_model_0.cif").write_text("old")
            with self.assertRaisesRegex(RuntimeError, "skipped the input"):
                self.run_prediction(root, ["Failed to process input.yaml. Skipping.\n"])
            self.assertEqual(self.namespace["structures"], [])

    def test_prediction_failure_does_not_accept_old_results(self):
        with tempfile.TemporaryDirectory() as root:
            prediction_dir = Path(root) / "boltz_results_input" / "predictions" / "input"
            prediction_dir.mkdir(parents=True)
            (prediction_dir / "input_model_0.cif").write_text("old")
            with self.assertRaisesRegex(RuntimeError, "failed predictions"):
                self.run_prediction(root, ["Number of failed examples: 1\n"])
            self.assertEqual(self.namespace["structures"], [])

    def test_missing_structure_and_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "produced no structure"):
                self.run_prediction(root, [])
            with self.assertRaises(subprocess.CalledProcessError):
                self.run_prediction(root, [], returncode=1)


if __name__ == "__main__":
    unittest.main()
