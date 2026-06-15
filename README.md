# More with LESS – Local Scene Representations for Tactile Imaging

**Zohar Rimon, Elisei Shafer, Tal Tepper, Daniel Kozin, Alon Malka, Roy Holland, Aviv Tamar**  
Technion – Israel Institute of Technology &nbsp;|&nbsp; Robotics: Science and Systems 2026

[![Website](https://img.shields.io/badge/Website-zoharri.github.io/LESS-blue)](https://zoharri.github.io/LESS/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.14344-b31b1b.svg)](https://arxiv.org/abs/2606.14344)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](assets/more_with_less.pdf)
[![Data](https://img.shields.io/badge/Data-Zenodo-green)](https://zenodo.org/communities/artificial-palpation)

<p align="center">
  <img src="assets/teaser.gif" alt="LESS teaser" width="300">
</p>

> Data Driven, 3D, Hand-Held, Generalizable Tactile Imaging.

---

## Installation

```bash
conda create -n less python=3.9 "pip<25" -y
conda activate less
pip install -r requirements.txt
```

## Data

All datasets are open-sourced on [Zenodo](https://zenodo.org/communities/artificial-palpation). Use the provided [download_data.sh](download_data.sh) script:

```bash
chmod +x download_data.sh

bash download_data.sh                       # download all
bash download_data.sh poke_primitive        # primary training set only
bash download_data.sh big                   # big-phantom set only
bash download_data.sh multi                 # multi-tumor set only
bash download_data.sh handheld              # hand-held set only
```

### data-poke-primitive ~213 GB — [Zenodo record (poke)](https://zenodo.org/records/20367204) | [Zenodo record (primitive)](https://zenodo.org/records/20367198)

The primary training set used in the paper. Contains palpation trajectories over a set of breast phantoms with spherical inserts, collected with a robot arm at controlled poke speeds. Both *poke* and *primitive* subsets are merged into a single directory (`data/data_poke_primitive/`).

### data-poke-big, data-poke-multi, data-handheld — [Zenodo record](https://zenodo.org/records/20367501)

All three generalization datasets are bundled in a single Zenodo record:

- **data-poke-big** — palpation data on larger phantoms (out-of-distribution sizes). Saved to `data/data_poke_big/`.
- **data-poke-multi** — palpation data on phantoms with multiple simultaneous inserts. Saved to `data/data_poke_multi/`.
- **data-handheld** — palpation data collected with a hand-held sensor (no robot arm). Includes estimated sensor locations and a pose calibration file (`est_to_robot_calibration.npz`). Saved to `data/data_handheld/`.

## Reproducing Paper Results

All experiments from the paper are defined as WandB sweeps in the [sweeps/](sweeps/) directory, organized by experiment group. To reproduce:

1. **Download the data** (see the Data section below) and update every `dataset.data_folder` value in the sweep YAML files to point to your local dataset path.

2. **Run Stage 1 sweeps** (rep training — files named `particle_gru_*` or `global_gru_*`) to train the tactile representation:
   ```bash
   wandb sweep sweeps/<group>/<sweep_file>.yaml
   wandb agent <entity>/<project>/<sweep_id>
   ```

3. **Update Stage 2 sweep checkpoints** — the downstream sweeps (files named `*TCNN*`) have `checkpoints/TBD/best_rep_model.pt` placeholder values. Replace `TBD` with the actual WandB run IDs from the Stage 1 sweeps.

4. **Run Stage 2 sweeps** in the same way.

5. **Evaluate** — after Stage 2 training, open `test_rep_and_downstream.py` and copy the run IDs (folder names created under `checkpoints/` during training) into the `checkpoints` list of the matching `TESTS` entry. Each entry has a comment pointing to its corresponding sweep. Then run:
   ```bash
   python3 test_rep_and_downstream.py --test <TEST_NAME>
   ```

## Training LESS

Training is two-stage: first learn the tactile representation, then train the downstream imaging model with the representation frozen.

### Stage 1 — Representation Learning

```bash
python3 train_rep_and_downstream.py \
  --config_path configs/GRU_reconstruction_model_real.yaml \
  --dataset.data_folder data/palpation_dataset_0 \
  --models.rep_learning_model.name ParticlesReconstructionModel \
  --optimizer.disable_downstream_model true 
```

At the end of training the path to the best checkpoint is printed. Pass it to Stage 2 via `--models.rep_learning_model_checkpoint`.

### Stage 2 — 2D Tactile Imaging

```bash
python3 train_rep_and_downstream.py \
  --config_path configs/GRU_reconstruction_model_real.yaml \
  --dataset.data_folder data/palpation_dataset_0 \
  --models.downstream_model.name TransposedConvParRepPred \
  --models.downstream_model.transposed_conv_par_rep_pred.num_channels 128 \
  --models.downstream_model.transposed_conv_par_rep_pred.particle_image_pred_size 32 \
  --models.downstream_model.transposed_conv_par_rep_pred.loss dice_focal \
  --models.rep_learning_model_checkpoint /path/to/best_rep_model.pt \
  --optimizer.freeze_rep_learning_model true \
  --num_epochs 500
```

### Stage 2 — 3D Tactile Imaging

```bash
python3 train_rep_and_downstream.py \
  --config_path configs/GRU_reconstruction_model_real_3d.yaml \
  --dataset.data_folder data/palpation_dataset_0 \
  --models.rep_learning_model_checkpoint /path/to/best_rep_model.pt \
  --optimizer.freeze_rep_learning_model true \
  --num_epochs 500
```

Use `--help` to see all available configuration options.

## Citation

```bibtex
@inproceedings{rimon2026less,
  title     = {More with {LESS}: Local Scene Representations for Tactile Imaging},
  author    = {Rimon, Zohar and Shafer, Elisei and Tepper, Tal and Kozin, Daniel
               and Malka, Alon and Holland, Roy and Tamar, Aviv},
  booktitle = {Robotics: Science and Systems},
  year      = {2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
