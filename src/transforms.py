"""
Data augmentation transforms for mel spectrograms.
"""

import numpy as np
import torch


class SpecAugment:
    """
    SpecAugment: frequency and time masking for spectrograms.
    
    Reference: https://arxiv.org/abs/1904.08779
    """
    
    def __init__(
        self,
        freq_mask_param: int = 15,
        time_mask_param: int = 25,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
    ):
        """
        Args:
            freq_mask_param: Max width of frequency mask
            time_mask_param: Max width of time mask
            n_freq_masks: Number of frequency masks to apply
            n_time_masks: Number of time masks to apply
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
    
    def __call__(self, spec):
        """
        Apply SpecAugment to a spectrogram.
        
        Args:
            spec: Tensor of shape (n_mels, time) or (batch, n_mels, time)
        
        Returns:
            Augmented spectrogram
        """
        spec = spec.clone() if isinstance(spec, torch.Tensor) else spec.copy()
        
        n_mels = spec.shape[-2]
        n_frames = spec.shape[-1]
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param + 1)
            f0 = np.random.randint(0, max(1, n_mels - f))
            spec[..., f0:f0 + f, :] = 0
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param + 1)
            t0 = np.random.randint(0, max(1, n_frames - t))
            spec[..., :, t0:t0 + t] = 0
        
        return spec


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class RandomApply:
    """Apply a transform with a given probability."""
    
    def __init__(self, transform, p: float = 0.5):
        self.transform = transform
        self.p = p
    
    def __call__(self, x):
        if np.random.random() < self.p:
            return self.transform(x)
        return x


class AddNoise:
    """Add random Gaussian noise."""
    
    def __init__(self, noise_level: float = 0.005):
        self.noise_level = noise_level
    
    def __call__(self, spec):
        if isinstance(spec, torch.Tensor):
            noise = torch.randn_like(spec) * self.noise_level
        else:
            noise = np.random.randn(*spec.shape) * self.noise_level
        return spec + noise


class TimeShift:
    """Randomly shift spectrogram in time (circular)."""
    
    def __init__(self, max_shift: int = 20):
        self.max_shift = max_shift
    
    def __call__(self, spec):
        shift = np.random.randint(-self.max_shift, self.max_shift + 1)
        if isinstance(spec, torch.Tensor):
            return torch.roll(spec, shifts=shift, dims=-1)
        else:
            return np.roll(spec, shift, axis=-1)

