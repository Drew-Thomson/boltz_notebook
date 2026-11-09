# Boltz Colab notebooks

Run [Boltz](https://github.com/jwohlwend/boltz) structure predictions in Google Colab using a standard Boltz YAML input.

[Open the Boltz notebook in Colab](https://colab.research.google.com/github/Drew-Thomson/boltz_notebook/blob/main/Boltz_Colab.ipynb)

## Getting started

1. Open the notebook, select **Runtime → Change runtime type → GPU**, and run the cells in order. Colab availability, GPU memory, and runtime limits vary.
2. Upload one `.yaml` or `.yml` file in the upload cell. Follow the [Boltz input specification](https://github.com/jwohlwend/boltz/blob/v2.2.1/docs/prediction.md) for proteins, ligands, constraints, templates, and optional affinity prediction.
3. Configure the prediction form and run the prediction. The notebook exposes every `boltz predict` option in Boltz 2.2.1; blank optional fields use Boltz defaults. Use the help cell for descriptions.
4. Download the results ZIP, or set `out_dir` to a mounted Google Drive directory to retain results. Colab's local files disappear when the runtime is deleted.

The notebook enables the public MMseqs2 MSA server by default for proteins without an MSA. This sends sequences to an external service; disable it and supply your own MSA if that is inappropriate. Input files that reference local MSAs or templates must use paths accessible in the Colab runtime; relative paths are resolved from `/content`. Upload these supporting files through Colab's Files pane or mount Drive, and adjust paths in the YAML as needed.

The default model is Boltz-2; choose Boltz-1 in the form if needed. Affinity prediction also requires a `properties` entry in the YAML. A GPU is strongly recommended; large jobs can exceed Colab's memory or session limits. See the upstream [prediction guide](https://github.com/jwohlwend/boltz/blob/v2.2.1/docs/prediction.md) for input details, output interpretation, and troubleshooting.

Boltz reuses preprocessed inputs by name. When changing a YAML, MSA, template, or preprocessing settings, use a new output directory or a new YAML filename, even if `override` is enabled. `override` recomputes predictions but does not clear Boltz's preprocessing cache.

Predictions and input sequences may be sensitive. Colab, the selected MSA server, notebook output, downloaded archives, and any mounted Drive storage are subject to their respective privacy policies. Do not save credentials in the notebook: authenticated MSA server secrets are requested interactively instead.

The notebook installs `boltz[cuda]==2.2.1` for a reproducible CLI. It requires Python 3.10–3.12 and downloads model weights on first use. If Colab changes its runtime or dependencies, select a compatible runtime or update the notebook and re-check the options against the new Boltz version.
