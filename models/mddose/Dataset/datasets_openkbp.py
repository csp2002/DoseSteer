"""MD-Dose's slice dataset for OpenKBP.

Reads the 2D slices already produced by DoseDiff's preprocessing
(<npy_root>/{train,validation,test}/<field>/<pt>_<z>.npy, default
data/openkbp/preprocessed/NPY, see preprocessing/openkbp) and converts them to
MD-Dose conventions: image-valued channels live in [0, 1] (the sampler clamps
x_0 to [0, 1]); dose in Gy is recovered via `pred * 80`.

Condition layout (10 channels):
  [0] CT                        (DoseDiff npy is [-1,1] -> rescaled to [0,1])
  [1] fused PTV mask            (70/70*PTV70 + 63/70*PTV63 + 56/70*PTV56)
  [2] Mask_Brainstem
  [3] Mask_Esophagus
  [4] Mask_Larynx
  [5] Mask_LeftParotid
  [6] Mask_Mandible
  [7] Mask_possible_dose_mask   (body / feasible-dose region)
  [8] Mask_RightParotid
  [9] Mask_SpinalCord
"""

import os

import numpy as np
import torch
import torch.utils.data as data

DOSEDIFF_NPY_ROOT = 'data/openkbp/preprocessed/NPY'

NON_PTV_MASKS = ['Mask_Brainstem', 'Mask_Esophagus', 'Mask_Larynx',
                 'Mask_LeftParotid', 'Mask_Mandible', 'Mask_possible_dose_mask',
                 'Mask_RightParotid', 'Mask_SpinalCord']

CONDITION_CHANNELS = 1 + 1 + len(NON_PTV_MASKS)  # CT + fused PTV + 8 masks = 10


class MyDataset(data.Dataset):
    def __init__(self, phase, aug=False, npy_root=DOSEDIFF_NPY_ROOT):
        self.phase = phase
        self.aug = aug
        phase_dir = {'train': 'train', 'val': 'validation', 'test': 'test'}[phase]
        self.data_root = os.path.join(npy_root, phase_dir)
        self.file_name_list = sorted(os.listdir(os.path.join(self.data_root, 'ct')))
        self.sum_case = len(self.file_name_list)

    def _load(self, field, file_name):
        return np.load(os.path.join(self.data_root, field, file_name))

    def __getitem__(self, index_):
        file_name = self.file_name_list[index_ % self.sum_case]

        ct = (self._load('ct', file_name) + 1) / 2      # [-1,1] -> [0,1]
        dose = (self._load('dose', file_name) + 1) / 2  # [-1,1] -> [0,1]; Gy = *80

        ptvs = (70.0 / 70.0) * self._load('Mask_PTV70', file_name) \
             + (63.0 / 70.0) * self._load('Mask_PTV63', file_name) \
             + (56.0 / 70.0) * self._load('Mask_PTV56', file_name)
        masks = [self._load(f, file_name) for f in NON_PTV_MASKS]

        condition = np.stack([ct, ptvs] + masks, axis=0)
        possible_mask = masks[NON_PTV_MASKS.index('Mask_possible_dose_mask')]

        image_ = torch.from_numpy(condition).float()
        dose_ = torch.from_numpy(dose.copy()).float()
        mask_ = torch.from_numpy(possible_mask.copy()).float()
        ct_path = os.path.join(self.data_root, 'ct', file_name)
        dose_path = os.path.join(self.data_root, 'dose', file_name)
        return image_, dose_, mask_, ct_path, dose_path

    def __len__(self):
        return self.sum_case
