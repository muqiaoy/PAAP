import torch
import numpy as np
import os
import torch.nn.functional as F

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
sys.path.append(os.path.join(parentdir, 'charsiu'))
from charsiu.Charsiu import charsiu_predictive_aligner

from trainer.estimator import PhonemeWeightEstimator, AcousticEstimator



class PAAPLoss(torch.nn.Module):
    
    def __init__(self, args, device = 'cuda'):
        
        super(PAAPLoss, self).__init__()
        model_state_dict = torch.load(os.path.join(parentdir, args.acoustic_model_path), map_location=device)['model_state_dict']
        self.args = args
        self.estimate_acoustics = AcousticEstimator()
        if self.args is not None:
            if self.args.ac_loss_type == "l2":
                self.l2 = torch.nn.MSELoss()
            if self.args.ac_loss_type == "l1":
                self.l1 = torch.nn.L1Loss()
        self.estimate_acoustics.load_state_dict(model_state_dict)
        self.estimate_acoustics.to(device)
        self.estimate_acoustics.train()

        self.is_phoneme_weighted = self.args.is_phoneme_weighted
        if self.is_phoneme_weighted:
            if args.phoneme_weight_path is not None:
                self.weight = torch.from_numpy(np.load(os.path.join(parentdir, args.phoneme_weight_path))).to(device)
            else:
                self.weight = PhonemeWeightEstimator().to(device)
                self.weight.load_state_dict(torch.load(os.path.join(parentdir, args.phoneme_model_path))['state'])
            self.num_phonemes = 42  #TODO: avoid hardcode
        
    def __call__(self, clean_waveform, enhan_waveform):
        
        return self.forward(clean_waveform, enhan_waveform)

    def forward(self, clean_waveform, enhan_waveform, noisy_waveform = None):
        
        clean_spectrogram = self.get_stft(clean_waveform)
        enhan_spectrogram, enhan_st_energy = self.get_stft(enhan_waveform, return_short_time_energy = True)
        
        
        clean_acoustics = self.estimate_acoustics(clean_spectrogram)
        enhan_acoustics = self.estimate_acoustics(enhan_spectrogram)
        
        
        if noisy_waveform is not None:
            noisy_spectrogram = self.get_stft(noisy_waveform)
            noisy_acoustics = self.estimate_acoustics(noisy_spectrogram)
            
            
        if self.args is None:
            if noisy_waveform is not None:
                return {"clean_acoustics": clean_acoustics, "enhan_acoustics": enhan_acoustics, "noisy_acoustics": noisy_acoustics}
            else:
                return {"clean_acoustics": clean_acoustics, "enhan_acoustics": enhan_acoustics}
        else:
            if self.is_phoneme_weighted:
                if self.args.phoneme_weight_path is not None:
                    clean_acoustics = torch.cat((clean_acoustics, torch.ones(clean_acoustics.size(0), clean_acoustics.size(1), 1).cuda()), dim=-1)
                    enhan_acoustics = torch.cat((enhan_acoustics, torch.ones(enhan_acoustics.size(0), enhan_acoustics.size(1), 1).cuda()), dim=-1)
                    clean_acoustics = clean_acoustics @ self.weight
                    enhan_acoustics = enhan_acoustics @ self.weight
                else:
                    clean_acoustics = self.weight(clean_acoustics)
                    enhan_acoustics = self.weight(enhan_acoustics)

            if self.args.ac_loss_type == "vector_l2":
                loss = torch.linalg.vector_norm(
                    enhan_acoustics - clean_acoustics,
                    ord     = 2, 
                    dim     = 1)
                return torch.mean(loss)
            elif self.args.ac_loss_type == "l2":
                loss   = self.l2(enhan_acoustics, clean_acoustics)
                return loss
            elif self.args.ac_loss_type == "l1":
                loss   = self.l1(enhan_acoustics, clean_acoustics)
                return loss
            elif self.args.ac_loss_type == "frame_energy_weighted_l2":
                factor = 1/(enhan_acoustics.size(dim=0) * enhan_acoustics.size(dim=1) * enhan_acoustics.size(dim=2))
                loss   = factor * torch.sum(((torch.sigmoid(enhan_st_energy) ** 0.5).unsqueeze(dim = -1) \
                * (enhan_acoustics - clean_acoustics)) ** 2)
                return loss                                        
            
           
    
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
        
        self.linear1 = torch.nn.Linear(512, 256)
        self.linear2 = torch.nn.Linear(256, 25)
        
        self.act = torch.nn.ReLU()
        
    def forward(self, A0):
        A1, _ = self.lstm(A0)
        Z1    = self.linear1(A1)
        A2    = self.act(Z1)
        Z2    = self.linear2(A2)
        
        return Z2

