# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

from collections import namedtuple
import json
from pathlib import Path
import math
import os
import sys
import multiprocessing
from tqdm import tqdm
from glob import glob
import numpy as np

import torchaudio
import torch
from torch.nn import functional as F

from .dsp import convert_audio

Info = namedtuple("Info", ["length", "sample_rate", "channels"])



def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def find_audio_files(path, exts=[".wav"], progress=True):
    audio_files = []
    for root, folders, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    for idx, file in enumerate(audio_files):
        info = get_info(file)
        meta.append((file, info.length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta


def get_egemap(file, file_length, examples, output_path, smile, length, stride, sample_rate, level):

    num_frames = 0
    offset = 0
    if level == 'func':
        egemaps = np.zeros((examples, len(smile.feature_names)))
    elif level == 'lld':
        if length is not None:
            egemaps = np.zeros((examples, length // 160 - 4, len(smile.feature_names)))
        else:
            egemaps = np.zeros((examples, file_length // 160 - 4, len(smile.feature_names)))
    else:
        raise NotImplementedError
    for seg_idx in range(examples):
        if length is not None:
            offset = stride * seg_idx
            num_frames = length
        if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
            seg, sr = torchaudio.load(str(file),
                                    frame_offset=offset,
                                    num_frames=num_frames or -1)
        else:
            seg, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
        egemaps[seg_idx] = smile.process_signal(seg, sampling_rate=sample_rate).values
    egemaps = egemaps.squeeze(0)  # squeeze in case examples == 1
    np.save(os.path.join(output_path, os.path.basename(file.replace(".wav", ".npy"))), egemaps)


def get_spec(file, file_length, examples, output_path, spectrogram, length, stride, sample_rate):

    num_frames = 0
    offset = 0
    if length is not None:
        spec = np.zeros((examples, spectrogram.win_length // 2 + 1, length // spectrogram.hop_length + 1))
    else:
        spec = np.zeros((examples, spectrogram.win_length // 2 + 1, file_length // spectrogram.hop_length + 1))
    for seg_idx in range(examples):
        if length is not None:
            offset = stride * seg_idx
            num_frames = length
        if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
            seg, sr = torchaudio.load(str(file),
                                    frame_offset=offset,
                                    num_frames=num_frames or -1)
        else:
            seg, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
        spec[seg_idx] = spectrogram(seg)
    spec = spec.squeeze(0)  # squeeze in case examples == 1
    np.save(os.path.join(output_path, os.path.basename(file.replace(".wav", ".npy"))), spec)


def get_phoneme_logits(file, file_length, examples, output_path, length, stride, sample_rate, aligner):

    num_frames = 0
    offset = 0
    if length is not None:
        logits = np.zeros((examples, length // 160 - 2, 42))
    else:
        logits = np.zeros((examples, file_length // 160 - 2, 42))
    for seg_idx in range(examples):
        if length is not None:
            offset = stride * seg_idx
            num_frames = length
        if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
            seg, sr = torchaudio.load(str(file),
                                    frame_offset=offset,
                                    num_frames=num_frames or -1)
        else:
            seg, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
        logits[seg_idx] = aligner.align(audio=seg)
    logits = logits.squeeze(0)  # squeeze in case examples == 1
    np.save(os.path.join(output_path, os.path.basename(file.replace(".wav", ".npy"))), logits)

def execute_multiprocess(files, num_examples, output_path, smile, length, stride, sample_rate, level):
    
    PROCESSES = 32
    
    with multiprocessing.Pool(PROCESSES) as pool:
        
        in_args = [(file, file_length, examples, output_path, smile, length, stride, sample_rate, level) 
                for (file, file_length), examples in zip(files, num_examples) if not os.path.exists(os.path.join(output_path, os.path.basename(file.replace(".wav", ".npy"))))]
        
        jobs = [pool.apply_async(get_egemap, in_arg) for in_arg in in_args]
        
        for j in tqdm(jobs):
            j.get()
            
    return None

def execute_singleprocess(files, num_examples, output_path, smile, length, stride, sample_rate, level):
    
    
    for (file, file_length), examples in tqdm(zip(files, num_examples)): 
        if not os.path.exists(os.path.join(output_path, os.path.basename(file.replace(".wav", ".npy")))):
            get_egemap(file, file_length, examples, output_path, smile, length, stride, sample_rate, level) 
            
    return None


def execute_singleprocess_ph_logits(files, num_examples, output_path, length, stride, sample_rate):

    import os,sys,inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0,parentdir)
    sys.path.append(os.path.join(parentdir, 'charsiu'))
    from charsiu.Charsiu import charsiu_predictive_aligner
    aligner = charsiu_predictive_aligner(aligner='charsiu/en_w2v2_fc_10ms')
    for (file, file_length), examples in tqdm(zip(files, num_examples)):
        if not os.path.exists(os.path.join(output_path, os.path.basename(file.replace(".wav", ".npy")))):
            get_phoneme_logits(file, file_length, examples, output_path, length, stride, sample_rate, aligner) 
    
    return None
                

   
def execute_multiprocess_spec(files, num_examples, output_path, spectrogram, length, stride, sample_rate):
    
    PROCESSES = 32
    
    with multiprocessing.Pool(PROCESSES) as pool:
        
        in_args = [(file, file_length, examples, output_path, spectrogram, length, stride, sample_rate) 
                for (file, file_length), examples in zip(files, num_examples) if not os.path.exists(os.path.join(output_path, os.path.basename(file.replace(".wav", ".npy"))))]
        
        jobs = [pool.apply_async(get_spec, in_arg) for in_arg in in_args]
        
        for j in tqdm(jobs):
            j.get()
            
    return None


class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None,
                 channels=None, convert=False, acoustic_path=None, ph_logits_path=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []

        self.length = length
        self.stride = stride or length
        if self.stride is not None:
            self.is_random_sampling = (self.stride >= self.files[0][1])  # TODO: Here it assumes all of the files are of the same length
        else:
            self.is_random_sampling = False
        assert self.is_random_sampling ^ pad
        self.sample_rate = sample_rate
        self.channels = channels
        self.convert = convert
        self.with_path = with_path

        # examples will be the number of windows in one file
        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length <= length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

        
        self.acoustic_path = acoustic_path
        self.ph_logits_path = ph_logits_path
        # generate the egemaps features for the 1st time if it does not exist

        if acoustic_path is not None:
            if not os.path.exists(acoustic_path):
                os.makedirs(acoustic_path, exist_ok=True)
            if len(glob(os.path.join(acoustic_path, "*.npy"))) < len(self.files):
                print("eGeMAPS LLDs do not exist (%d/%d). Generating... This might take a while" % (len(glob(os.path.join(acoustic_path, "*.npy"))), len(files)))
                import opensmile
                smile_lld = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
                execute_multiprocess(self.files, self.num_examples, acoustic_path, smile_lld, self.length, self.stride, self.sample_rate, level='lld')

        if ph_logits_path is not None:
            if not os.path.exists(ph_logits_path):
                os.makedirs(ph_logits_path, exist_ok=True)
            if len(glob(os.path.join(ph_logits_path, "*.npy"))) < len(self.files):
                print("phoneme logits do not exist (%d/%d). Generating... This might take a while" % (len(glob(os.path.join(ph_logits_path, "*.npy"))), len(files)))
                execute_singleprocess_ph_logits(self.files, self.num_examples, ph_logits_path, self.length, self.stride, self.sample_rate)
        return


    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index, frame_offset=None):
        for (file, file_length), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            if not self.is_random_sampling:
                assert frame_offset is None
                offset = 0
                if self.length is not None:
                    offset = self.stride * index
                    num_frames = self.length
            else:
                if frame_offset is None:
                    offset = 0
                    if self.length is not None:
                        offset = np.random.randint(file_length - self.length - 1)
                        num_frames = self.length
                    frame_offset = offset
                else:
                    assert (self.stride >= file_length)
                    offset = frame_offset
                    num_frames = self.length
            if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
                out, sr = torchaudio.load(str(file),
                                          frame_offset=offset,
                                          num_frames=num_frames or -1)
            else:
                out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
            target_sr = self.sample_rate or sr
            target_channels = self.channels or out.shape[0]
            if self.convert:
                out = convert_audio(out, sr, target_sr, target_channels)
            else:
                if sr != target_sr:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                       f"{target_sr}, but got {sr}")
                if out.shape[0] != target_channels:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                       f"{target_channels}, but got {sr}")
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))

            if self.acoustic_path is not None:
                acoustics = torch.from_numpy(np.load(os.path.join(self.acoustic_path, os.path.basename(file.replace(".wav", ".npy"))))).float()
                acoustics = F.normalize(acoustics)
            else:
                acoustics = torch.Tensor([-1])
            if self.ph_logits_path is not None:
                ph_logits = torch.from_numpy(np.load(os.path.join(self.ph_logits_path, os.path.basename(file.replace(".wav", ".npy"))))).float()
            else:
                ph_logits = torch.Tensor([-1])
            if self.with_path:
                return out, file
            else:
                return out, acoustics, ph_logits, frame_offset


if __name__ == "__main__":
    meta = []
    for path in sys.argv[1:]:
        meta += find_audio_files(path)
    json.dump(meta, sys.stdout, indent=4)