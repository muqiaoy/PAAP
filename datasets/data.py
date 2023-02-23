import json
import logging
import os
import re
import torch
import numpy as np

from .audio import Audioset

logger = logging.getLogger(__name__)


def match_dns(noisy, clean):
    """match_dns.
    Match noisy and clean DNS dataset filenames.

    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    """
    logger.debug("Matching noisy and clean for dns dataset")
    noisydict = {}
    extra_noisy = []
    for path, size in noisy:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            # maybe we are mixing some other dataset in
            extra_noisy.append((path, size))
        else:
            noisydict[match.group(1)] = (path, size)
    noisy[:] = []
    extra_clean = []
    copied = list(clean)
    clean[:] = []
    for path, size in copied:
        match = re.search(r'fileid_(\d+)\.wav$', path)
        if match is None:
            extra_clean.append((path, size))
        else:
            noisy.append(noisydict[match.group(1)])
            clean.append((path, size))
    extra_noisy.sort()
    extra_clean.sort()
    clean += extra_clean
    noisy += extra_noisy


def match_files(noisy, clean, matching="sort"):
    """match_files.
    Sort files to match noisy and clean filenames.
    :param noisy: list of the noisy filenames
    :param clean: list of the clean filenames
    :param matching: the matching function, at this point only sort is supported
    """
    if matching == "dns":
        # dns dataset filenames don't match when sorted, we have to manually match them
        match_dns(noisy, clean)
    elif matching == "sort":
        noisy.sort()
        clean.sort()
    elif matching == "sort_fileid":
        
        noisy = sorted(noisy ,key=lambda x: int(x[0].split("fileid_")[1].split(".wav")[0]))
        clean = sorted(clean ,key=lambda x: int(x[0].split("fileid_")[1].split(".wav")[0]))
    else:
        raise ValueError(f"Invalid value for matching {matching}")

class NoisyCleanSet:
    def __init__(self, json_dir, matching="sort", length=None, stride=None,
                 pad=True, sample_rate=None, acoustic_path=None, ph_logits_path=None):
        """__init__.

        :param json_dir: directory containing both clean.json and noisy.json
        :param matching: matching function for the files
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        """
        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')
        with open(noisy_json, 'r') as f:
            noisy = json.load(f)
        with open(clean_json, 'r') as f:
            clean = json.load(f)

        self.noisy_dir_list = match_files(noisy, clean, matching)
        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
        kw_data = {'acoustic_path': acoustic_path, 'ph_logits_path': ph_logits_path}
        self.clean_set = Audioset(clean, **kw, **kw_data)
        self.noisy_set = Audioset(noisy, **kw)

        self.MU = torch.from_numpy(np.array(
            [ 2.31615782e-01, -5.02114248e+00, 7.16793156e+00, 1.40047576e-02,
            -1.44424592e-03, 1.18291244e-01, 7.16937304e+00, 5.01161051e+00,
            7.38044071e+00, 1.30544746e+00, 7.16783571e+00, 7.72617990e-03,
            3.78611624e-01, 1.80594587e+00, 2.74223471e+00, 7.16790104e+00,
            2.29371735e+02, 2.61031281e+02, -2.86713428e+01, 4.58741486e+02,
            2.72984955e+02, -2.86713428e+01, 4.58874390e+02, 2.71175812e+02,
            -2.86713428e+01], dtype=np.float32))
        self.STD = torch.from_numpy(np.array(
            [ 4.24716711e-01, 1.09750290e+01, 1.51086359e+01, 2.98775751e-02,
            1.85245797e-02, 2.39421308e-01, 1.63376312e+01, 1.22261524e+01,
            1.53735695e+01, 1.42613926e+01, 1.21981163e+01, 2.58955006e-02,
            8.05543840e-01, 3.83967781e+00, 6.79308844e+00, 1.41308403e+01,
            3.49271667e+02, 6.28384338e+02, 6.05799637e+01, 6.89079407e+02,
            5.62089905e+02, 6.05799637e+01, 1.09140088e+03, 5.42341919e+02,
            6.05799637e+01], dtype=np.float32))

        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        noisy, _, _, _, offset = self.noisy_set[index]
        clean, acoustics, ph_logits, _ = self.clean_set.__getitem__(index, offset)
        acoustics = (acoustics - self.MU) / self.STD
        
        return noisy, clean, acoustics, ph_logits

    def __len__(self):
        return len(self.noisy_set)
    
    
