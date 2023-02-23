import torch
from torch.nn import functional
from torch.cuda.amp import autocast
from functools import partial

from .feature import drop_band, stft, istft
from .base_model import BaseModel
from .sequence_model import SequenceModel
from .mask import build_complex_ideal_ratio_mask, decompress_cIRM


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from trainer.utils import capture_init


class FullSubNet(BaseModel):
    @capture_init
    def __init__(self,
                 num_freqs,
                 look_ahead,
                 sequence_model,
                 fb_num_neighbors,
                 sb_num_neighbors,
                 fb_output_activate_function,
                 sb_output_activate_function,
                 fb_model_hidden_size,
                 sb_model_hidden_size,
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=2,
                 weight_init=True,
                 sample_rate=16000,
                 ):
        """
        FullSubNet model (cIRM mask)

        Args:
            num_freqs: Frequency dim of the input
            look_ahead: Number of use of the future frames
            fb_num_neighbors: How much neighbor frequencies at each side from fullband model's output
            sb_num_neighbors: How much neighbor frequencies at each side from noisy spectrogram
            sequence_model: Chose one sequence model as the basic model e.g., GRU, LSTM
            fb_output_activate_function: fullband model's activation function
            sb_output_activate_function: subband model's activation function
            norm_type: type of normalization, see more details in "BaseModel" class
        """
        super().__init__()
        assert sequence_model in ("GRU", "LSTM"), f"{self.__class__.__name__} only support GRU and LSTM."

        self.fb_model = SequenceModel(
            input_size=num_freqs,
            output_size=num_freqs,
            hidden_size=fb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=fb_output_activate_function
        )

        self.sb_model = SequenceModel(
            input_size=(sb_num_neighbors * 2 + 1) + (fb_num_neighbors * 2 + 1),
            output_size=2,
            hidden_size=sb_model_hidden_size,
            num_layers=2,
            bidirectional=False,
            sequence_model=sequence_model,
            output_activate_function=sb_output_activate_function
        )

        self.torch_stft = partial(stft)
        self.torch_istft = partial(istft)

        self.sb_num_neighbors = sb_num_neighbors
        self.fb_num_neighbors = fb_num_neighbors
        self.look_ahead = look_ahead
        self.norm = self.norm_wrapper(norm_type)
        self.num_groups_in_drop_band = num_groups_in_drop_band
        self.sample_rate = sample_rate

        if weight_init:
            self.apply(self.weight_init)

    def forward(self, noisy, clean=None):
        noisy = noisy.squeeze(dim=1)
        noisy_mag, noisy_phase, noisy_real, noisy_imag = self.torch_stft(noisy)
        if clean is not None:
            clean = clean.squeeze(dim=1)
            _, _, clean_real, clean_imag = self.torch_stft(clean)
            cIRM = build_complex_ideal_ratio_mask(noisy_real, noisy_imag, clean_real, clean_imag)  # [B, F, T, 2]
        
            if noisy.size(0) > 1:
                cIRM = drop_band(
                    cIRM.permute(0, 3, 1, 2),  # [B, 2, F ,T]
                    self.num_groups_in_drop_band
                ).permute(0, 2, 3, 1)

        # [B, F, T] => [B, 1, F, T] => model => [B, 2, F, T] => [B, F, T, 2]
        noisy_mag = noisy_mag.unsqueeze(1)
        cRM = self.forward_noisy_mag(noisy_mag)
        cRM = cRM.permute(0, 2, 3, 1)

        if clean is not None:
            enh_loss = functional.mse_loss(cIRM, cRM)
        
        cRM = decompress_cIRM(cRM)

        enhanced_real = cRM[..., 0] * noisy_real - cRM[..., 1] * noisy_imag
        enhanced_imag = cRM[..., 1] * noisy_real + cRM[..., 0] * noisy_imag

        enhanced = self.torch_istft((enhanced_real, enhanced_imag), length=noisy.size(-1), input_type="real_imag")

        if clean is not None:  # During training
            return enhanced, enh_loss
        else:
            return enhanced


    def forward_noisy_mag(self, noisy_mag):
        """
        Args:
            noisy_mag: noisy magnitude spectrogram

        Returns:
            The real part and imag part of the enhanced spectrogram

        Shapes:
            noisy_mag: [B, 1, F, T]
            return: [B, 2, F, T]
        """
        assert noisy_mag.dim() == 4
        noisy_mag = functional.pad(noisy_mag, [0, self.look_ahead])  # Pad the look ahead
        batch_size, num_channels, num_freqs, num_frames = noisy_mag.size()
        assert num_channels == 1, f"{self.__class__.__name__} takes the mag feature as inputs."

        # Fullband model
        fb_input = self.norm(noisy_mag).reshape(batch_size, num_channels * num_freqs, num_frames)
        fb_output = self.fb_model(fb_input).reshape(batch_size, 1, num_freqs, num_frames)

        # Unfold fullband model's output, [B, N=F, C, F_f, T]. N is the number of sub-band units
        fb_output_unfolded = self.unfold(fb_output, num_neighbors=self.fb_num_neighbors)
        fb_output_unfolded = fb_output_unfolded.reshape(batch_size, num_freqs, self.fb_num_neighbors * 2 + 1, num_frames)

        # Unfold noisy spectrogram, [B, N=F, C, F_s, T]
        noisy_mag_unfolded = self.unfold(noisy_mag, num_neighbors=self.sb_num_neighbors)
        noisy_mag_unfolded = noisy_mag_unfolded.reshape(batch_size, num_freqs, self.sb_num_neighbors * 2 + 1, num_frames)

        # Concatenation, [B, F, (F_s + F_f), T]
        sb_input = torch.cat([noisy_mag_unfolded, fb_output_unfolded], dim=2)
        sb_input = self.norm(sb_input)

        # Speeding up training without significant performance degradation.
        # These will be updated to the paper later.
        if batch_size > 1:
            sb_input = drop_band(sb_input.permute(0, 2, 1, 3), num_groups=self.num_groups_in_drop_band)  # [B, (F_s + F_f), F//num_groups, T]
            num_freqs = sb_input.shape[2]
            sb_input = sb_input.permute(0, 2, 1, 3)  # [B, F//num_groups, (F_s + F_f), T]

        sb_input = sb_input.reshape(
            batch_size * num_freqs,
            (self.sb_num_neighbors * 2 + 1) + (self.fb_num_neighbors * 2 + 1),
            num_frames
        )

        # [B * F, (F_s + F_f), T] => [B * F, 2, T] => [B, F, 2, T]
        sb_mask = self.sb_model(sb_input)
        sb_mask = sb_mask.reshape(batch_size, num_freqs, 2, num_frames).permute(0, 2, 1, 3).contiguous()

        output = sb_mask[:, :, :, self.look_ahead:]
        return output