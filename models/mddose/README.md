# MD-Dose

Code from [LinjieFu-U/mamba_dose](https://github.com/LinjieFu-U/mamba_dose) (Fu et al., "MD-Dose: A
diffusion model based on the Mamba for radiation dose prediction", BIBM 2024). It builds on [Vim](https://github.com/hustvl/Vim)
(`vim/`), whose modified copies of [mamba](https://github.com/state-spaces/mamba) (`mamba/`) and
[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) (`causal-conv1d/`) are included.

Ours: `Dataset/datasets_openkbp.py` reads DoseDiff's preprocessed 2D slices; `pred_openkbp.py` predicts and scores the
test cases; `ddim_struc.py` adds a deterministic DDIM sampler; `train_struc.py` keeps the epoch with the lowest validation
MAE (`model_best_mae.pkl`).
