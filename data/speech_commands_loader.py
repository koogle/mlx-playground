import os
import re
import hashlib
import wave
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union

import mlx.core as mx
import numpy as np


class SpeechCommandsDataset:
    """
    Data loader for Google Speech Commands v2 dataset.
    
    Loads 1-second audio files and converts them to spectrograms or raw waveforms
    for training state space models on speech recognition/transcription tasks.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "training",
        validation_percentage: float = 10.0,
        testing_percentage: float = 10.0,
        sample_rate: int = 16000,
        use_spectrogram: bool = True,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        background_noise_prob: float = 0.1,
        background_noise_volume: float = 0.1,
    ):
        """
        Args:
            data_dir: Path to speech_commands_v2 directory
            split: 'training', 'validation', or 'testing'  
            validation_percentage: Percentage of data for validation
            testing_percentage: Percentage of data for testing
            sample_rate: Audio sample rate (should be 16000 for this dataset)
            use_spectrogram: If True, convert audio to mel spectrograms
            n_mels: Number of mel frequency bands
            n_fft: FFT size for spectrogram
            hop_length: Hop length for STFT
            background_noise_prob: Probability of adding background noise
            background_noise_volume: Volume of background noise to add
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.validation_percentage = validation_percentage
        self.testing_percentage = testing_percentage
        self.sample_rate = sample_rate
        self.use_spectrogram = use_spectrogram
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.background_noise_prob = background_noise_prob
        self.background_noise_volume = background_noise_volume
        
        # Load class labels and file paths
        self.classes, self.class_to_idx = self._load_classes()
        self.file_paths, self.labels = self._load_file_list()
        self.background_noise_files = self._load_background_noise_files()
        
        print(f"Loaded {len(self.file_paths)} files for {split} split")
        print(f"Classes: {self.classes}")
        
    def _load_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """Load all available word classes from directory names"""
        classes = []
        for item in os.listdir(self.data_dir):
            item_path = self.data_dir / item
            if (item_path.is_dir() and 
                not item.startswith('_') and 
                item != 'LICENSE'):
                classes.append(item)
        
        classes.sort()  # Ensure consistent ordering
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return classes, class_to_idx
        
    def _load_background_noise_files(self) -> List[Path]:
        """Load background noise files for data augmentation"""
        noise_dir = self.data_dir / "_background_noise_"
        if not noise_dir.exists():
            return []
            
        noise_files = []
        for file in noise_dir.glob("*.wav"):
            noise_files.append(file)
        return noise_files
    
    def _which_set(self, filename: str) -> str:
        """
        Determines which data partition the file should belong to using hash function.
        This is the same function provided in the dataset README.
        """
        MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
        
        base_name = os.path.basename(filename)
        # Ignore anything after '_nohash_' for set determination
        hash_name = re.sub(r'_nohash_.*$', '', base_name)
        
        # Generate stable hash-based assignment
        hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
        percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) *
                          (100.0 / MAX_NUM_WAVS_PER_CLASS))
        
        if percentage_hash < self.validation_percentage:
            return 'validation'
        elif percentage_hash < (self.testing_percentage + self.validation_percentage):
            return 'testing'
        else:
            return 'training'
    
    def _load_file_list(self) -> Tuple[List[Path], List[int]]:
        """Load file paths and labels for the current split"""
        file_paths = []
        labels = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if not class_dir.is_dir():
                continue
                
            class_idx = self.class_to_idx[class_name]
            
            for wav_file in class_dir.glob("*.wav"):
                file_set = self._which_set(str(wav_file))
                if file_set == self.split:
                    file_paths.append(wav_file)
                    labels.append(class_idx)
        
        return file_paths, labels
    
    def _load_audio(self, file_path: Path) -> np.ndarray:
        """Load WAV file as numpy array"""
        with wave.open(str(file_path), 'rb') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            audio = audio / 32768.0  # Normalize to [-1, 1]
            return audio
    
    def _add_background_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add background noise to audio with specified probability"""
        if (random.random() > self.background_noise_prob or 
            not self.background_noise_files):
            return audio
            
        # Select random noise file
        noise_file = random.choice(self.background_noise_files)
        noise_audio = self._load_audio(noise_file)
        
        # Select random segment from noise (longer files)
        if len(noise_audio) > len(audio):
            start_idx = random.randint(0, len(noise_audio) - len(audio))
            noise_segment = noise_audio[start_idx:start_idx + len(audio)]
        else:
            # Tile noise if it's shorter than audio
            repeats = (len(audio) // len(noise_audio)) + 1
            noise_segment = np.tile(noise_audio, repeats)[:len(audio)]
        
        # Mix with background noise
        mixed_audio = audio + self.background_noise_volume * noise_segment
        return np.clip(mixed_audio, -1.0, 1.0)
    
    def _audio_to_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Convert audio waveform to mel spectrogram"""
        # Simple STFT implementation - in practice you'd use librosa or similar
        # This is a basic implementation for MLX compatibility
        
        # Zero-pad audio to handle edge cases
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        padded_audio = np.pad(audio, (self.n_fft//2, self.n_fft//2), mode='constant')
        
        # Compute STFT
        stft = []
        for i in range(n_frames):
            start = i * self.hop_length
            frame = padded_audio[start:start + self.n_fft]
            
            # Apply Hann window
            windowed = frame * np.hanning(self.n_fft)
            
            # FFT and compute magnitude spectrogram  
            fft = np.fft.rfft(windowed)
            magnitude = np.abs(fft)
            stft.append(magnitude)
        
        spectrogram = np.array(stft).T  # Shape: (n_freq_bins, n_time_frames)
        
        # Convert to mel scale (simplified - normally would use mel filter bank)
        # For now, just take first n_mels frequency bins as approximation
        mel_spec = spectrogram[:self.n_mels, :]
        
        # Log scale
        mel_spec = np.log(mel_spec + 1e-8)
        
        return mel_spec
    
    def __len__(self) -> int:
        return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Tuple[mx.array, mx.array]:
        """Get a single audio sample and its label"""
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        audio = self._load_audio(file_path)
        
        # Apply background noise augmentation for training
        if self.split == "training":
            audio = self._add_background_noise(audio)
        
        # Convert to spectrogram or use raw audio
        if self.use_spectrogram:
            features = self._audio_to_spectrogram(audio)
            # Shape: (n_mels, n_time_frames) -> (n_time_frames, n_mels)
            features = features.T
        else:
            # Use raw audio - reshape to (seq_len, 1)
            features = audio.reshape(-1, 1)
        
        return mx.array(features), mx.array([label])
    
    def create_batches(self, batch_size: int, shuffle: bool = True):
        """Create batches for training/evaluation"""
        indices = list(range(len(self)))
        if shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            batch_features = []
            batch_labels = []
            
            for idx in batch_indices:
                features, label = self[idx]
                batch_features.append(features)
                batch_labels.append(label)
            
            # Stack into batches
            features_batch = mx.stack(batch_features)
            labels_batch = mx.stack(batch_labels).squeeze(-1)
            
            yield features_batch, labels_batch


def create_speech_commands_loaders(
    data_dir: str,
    batch_size: int = 32,
    **kwargs
) -> Tuple[SpeechCommandsDataset, SpeechCommandsDataset, SpeechCommandsDataset]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_dir: Path to speech_commands_v2 directory
        batch_size: Batch size for data loading
        **kwargs: Additional arguments for SpeechCommandsDataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = SpeechCommandsDataset(data_dir, split="training", **kwargs)
    val_dataset = SpeechCommandsDataset(data_dir, split="validation", **kwargs)  
    test_dataset = SpeechCommandsDataset(data_dir, split="testing", **kwargs)
    
    return train_dataset, val_dataset, test_dataset


# Example usage
if __name__ == "__main__":
    # Create data loaders
    data_dir = "data/speech_commands_v2"
    train_loader, val_loader, test_loader = create_speech_commands_loaders(
        data_dir=data_dir,
        batch_size=32,
        use_spectrogram=True,
        background_noise_prob=0.1
    )
    
    # Test loading a batch
    for features, labels in train_loader.create_batches(batch_size=4, shuffle=True):
        print(f"Features shape: {features.shape}")  # (batch, time, features)
        print(f"Labels shape: {labels.shape}")      # (batch,)
        print(f"Sample labels: {labels}")
        break