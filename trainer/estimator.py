import torch
import torch.nn as nn
import numpy as np
import os
import torch.nn.functional as F

# import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
# sys.path.append(os.path.join(parentdir, 'charsiu'))
# from charsiu.Charsiu import charsiu_predictive_aligner

from trainer.utils import capture_init


class PhonemeWeightEstimator(torch.nn.Module):
    
    @capture_init
    def __init__(self):
        
        super(PhonemeWeightEstimator, self).__init__()

        self.lstm = torch.nn.LSTM(25, 64, 3, bidirectional=True, batch_first=True)
        
        self.nn = nn.Sequential(
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 42),
        )
        
        # self.lstm = torch.nn.LSTM(514, 256, 3, bidirectional=True, batch_first=True)
        
        # self.nn = nn.Sequential(
        #             nn.Linear(512, 256),
        #             nn.ReLU(),
        #             nn.Linear(256, 42),
        # )
        
    def forward(self, estimated_acoustics):
        # estimated_acoustics = F.normalize(estimated_acoustics, dim=1)
        hidden, _ = self.lstm(estimated_acoustics)
        estimated_ph_logits = self.nn(hidden)
        
        return estimated_ph_logits

    # def forward(self, wav):
    #     wav = wav.squeeze(dim=1)
    #     spec = self.get_stft(wav)
    #     hidden, _ = self.lstm(spec)
    #     estimated_lld = self.nn(hidden)
        
    #     return estimated_lld

    def get_stft(self, wav, return_short_time_energy = False):
        self.nfft = 512
        self.hop_length = 160
        spec = torch.stft(wav, n_fft=self.nfft, hop_length=self.hop_length, return_complex=False)
        
        
        spec_real = spec[..., 0]
        spec_imag = spec[..., 1]
             
                
        spec = spec.permute(0, 2, 1, 3).reshape(spec.size(dim=0), -1, 514)
        
        
        
        if return_short_time_energy:
            st_energy = torch.mul(torch.sum(spec_real**2 + spec_imag**2, dim = 1), 2/self.nfft)
            assert spec.size(dim=1) == st_energy.size(dim=1)
            return spec.float(), st_energy.float()
        else: 
            return spec.float()



class AcousticEstimator(torch.nn.Module):
    
    def __init__(self):
        
        super(AcousticEstimator, self).__init__()
        
        self.lstm = torch.nn.LSTM(514, 256, 3, bidirectional=True, batch_first=True)
        
        self.nn = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, 25),
        )
        
    def forward(self, wav):
        wav = wav.squeeze(dim=1)
        spec = self.get_stft(wav)
        hidden, _ = self.lstm(spec)
        estimated_lld = self.nn(hidden)
        
        return estimated_lld

    def get_stft(self, wav, return_short_time_energy = False):
        self.nfft = 512
        self.hop_length = 160
        spec = torch.stft(wav, n_fft=self.nfft, hop_length=self.hop_length, return_complex=False)
        
        
        spec_real = spec[..., 0]
        spec_imag = spec[..., 1]
             
                
        spec = spec.permute(0, 2, 1, 3).reshape(spec.size(dim=0), -1, 514)
        
        
        
        if return_short_time_energy:
            st_energy = torch.mul(torch.sum(spec_real**2 + spec_imag**2, dim = 1), 2/self.nfft)
            assert spec.size(dim=1) == st_energy.size(dim=1)
            return spec.float(), st_energy.float()
        else: 
            return spec.float()