# DoseSteer

Code for *DoseSteer: Steering diffusion-based dose prediction models towards clinical targets via Dual-Response Guidance*.

Dual-Response Guidance (DRG) steers a frozen diffusion dose predictor at test time towards structure-level clinical
targets (OAR mean dose, PTV D95). This repository covers DoseDiff, MD-Dose and DoseDiT on OpenKBP, and DoseDiff on
GDP-HMM.

```
setup/           1. Setup: environment files
preprocessing/   2. Data download and preprocessing (OpenKBP, GDP-HMM)
models/          3. Model training and evaluation (DoseDiff, MD-Dose, DoseDiT)
drg/             4. Steering with DRG
third_party/     DINOv3 ViT code used by DoseDiT
```

Run all commands from the repository root.

## 1. Setup

Two conda environments: `dosediff` for the data preparation and for DoseDiff and DoseDiT on OpenKBP, `mddose` for MD-Dose
and for DoseDiff on GDP-HMM.

```bash
conda create -n dosediff python=3.11 -y && conda activate dosediff     # PyTorch 2.4.1, CUDA 12.1
pip install -r setup/requirements_dosediff.txt

conda create -n mddose python=3.10 -y && conda activate mddose         # PyTorch 2.6.0, CUDA 12.4
pip install -r setup/requirements_mddose.txt
# MD-Dose only: build its two CUDA extensions (needs nvcc 12.4; on Blackwell GPUs, PyTorch 2.7.0 and CUDA 12.8)
CAUSAL_CONV1D_FORCE_BUILD=TRUE pip install -e models/mddose/causal-conv1d --no-build-isolation
MAMBA_FORCE_BUILD=TRUE pip install -e models/mddose/mamba --no-build-isolation
```

## 2. Data download and preprocessing

**OpenKBP.** Download the [OpenKBP](https://github.com/ababier/open-kbp) data (200 training, 40 validation and 100 test
cases) and preprocess it with DoseDiff's pipeline (env `dosediff`):

```bash
git clone https://github.com/ababier/open-kbp && mkdir -p data/openkbp && cp -r open-kbp/provided-data data/openkbp/
python preprocessing/openkbp/csv2nii.py        # CSV -> NIfTI volumes
python preprocessing/openkbp/PSDM_OpenKBP.py   # structure masks and signed distance maps
python preprocessing/openkbp/NII_to_npy.py     # 2D slices for training
```

The three OpenKBP predictors share the result, `data/openkbp/preprocessed/`: `{train,validation,test}-pats/` (reference
doses for evaluation), `{train,validation,test}-pats_preprocess/` (model inputs) and `NPY/` (training slices).

**GDP-HMM.** Clone the [challenge repository](https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge) (its `meta_files/` list
the plans and splits) to `data/gdp_hmm`, download the NumPy data from
[Hugging Face](https://huggingface.co/datasets/Jungle15/GDP-HMM_Challenge) into `data/gdp_hmm/data/{HaN,Lung}/{train,valid}/`,
make the paths in `meta_data.csv` relative and build the slice cache for training (~100 GB, env `dosediff`):

```bash
git clone https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge data/gdp_hmm
sed -i 's#/pct_ids/users/z004b27b/data/GDP-HMM_Challenge/#data/#' data/gdp_hmm/meta_files/meta_data.csv
python preprocessing/gdp_hmm/gdp_data.py    # -> data/gdp_hmm/cache
```

The splits follow the `dev_split` column of `meta_data.csv`: `train` (2727 plans), `valid` (148 plans, model
selection) and `test` (356 plans: the challenge's validation set).

## 3. Model training and evaluation

The scripts' defaults are the paper's settings. Each training keeps the checkpoint with the lowest validation MAE
(`*_best_mae.*`). The test scripts sample the test cases with 8 deterministic DDIM steps and write the doses and official
scores under `runs/<model>/results/`. Scripts run on GPU 0 unless `--gpu` is given (`train_struc.py`: set
`CUDA_VISIBLE_DEVICES`).

```bash
# DoseDiff on OpenKBP (env dosediff)
torchrun --nproc_per_node=4 models/dosediff/train.py --gpu 0,1,2,3
python models/dosediff/pred.py --run_dir runs/dosediff/<run>

# MD-Dose on OpenKBP (env mddose)
python models/mddose/train_struc.py
python models/mddose/pred_openkbp.py

# DoseDiT on OpenKBP (env dosediff)
torchrun --nproc_per_node=1 models/dosedit/train.py
python models/dosedit/pred.py

# DoseDiff on GDP-HMM (env mddose)
torchrun --nproc_per_node=1 models/dosediff/train_gdp.py
python models/dosediff/pred_gdp.py --ckpt runs/gdp_hmm/<run>/model_best_mae.pth
```

## 4. Steering with DRG

`drg/core.py` implements DRG with the paper's hyperparameters; `drg/specs_openkbp.py` and `drg/specs_gdp.py` wrap the
predictors for it. For every test case the steering scripts take the structure metrics of the reference dose as targets, run
the unsteered prediction and DRG from the same initial noise, and score both. Use the environments of section 3 (`mddose`
for MD-Dose and GDP-HMM):

```bash
python drg/steer_openkbp.py --model dosediff --ckpt runs/dosediff/<run>/model_best_mae.pth
python drg/steer_openkbp.py --model mddose
python drg/steer_openkbp.py --model dosedit
python drg/steer_gdp.py --ckpt runs/gdp_hmm/<run>/model_best_mae.pth
```

Without `--ckpt`, MD-Dose and DoseDiT load the best checkpoints of their default training runs. Each run writes
`results/<setting>/metrics.json` (per case: official score, DVH score, target error, voxel MAE inside and outside the
structures; `--save_dose` adds the doses) and prints the cohort means and DRG's improvement of the official score (paper:
DoseDiff ~3 %, MD-Dose ~5 %, DoseDiT ~5 %, GDP-HMM ~11 %). `steer_gdp.py --shard i --nshards n` splits the plans over
GPUs.

To steer another diffusion predictor, wrap it like `drg/specs_openkbp.py` (the interface is in the docstring of
`drg/core.py`).

## Acknowledgments

[DoseDiff](https://github.com/whisney/DoseDiff) (model, training, OpenKBP preprocessing and scorer; built on OpenAI's
[guided-diffusion](https://github.com/openai/guided-diffusion)), [MD-Dose](https://github.com/LinjieFu-U/mamba_dose) (with
[Vim](https://github.com/hustvl/Vim), [mamba](https://github.com/state-spaces/mamba) and
[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)), [DINOv3](https://github.com/facebookresearch/dinov3)
(DoseDiT's encoders are adapted from its ViT code), [OpenKBP](https://github.com/ababier/open-kbp) and
[GDP-HMM](https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge) (data and evaluation). Third-party code keeps its license
files.

This repository is for research use, not clinical deployment.
