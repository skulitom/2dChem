import pygame
import numpy as np
from typing import Dict, Optional
import os

class SoundManager:
    def __init__(self):
        """Initialize the sound system"""
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        
        # Set reasonable volume
        pygame.mixer.set_num_channels(32)  # Allow multiple sounds simultaneously
        self.master_volume = 0.3
        
        # Cache for generated sounds
        self._sound_cache: Dict[str, pygame.mixer.Sound] = {}
        self._last_played_time = 0
        self._min_interval = 50  # Minimum ms between sounds to prevent overwhelming
        
        # Generate base pop sounds
        self._generate_base_sounds()

    def _generate_base_sounds(self):
        """
        Generate base popping sounds for different elements.
        We'll give each element a slightly different pitch or duration,
        but keep the character gentle and ASMR-like.
        """
        self._generate_pop_sound('default', frequency=220.0, duration=0.15)
        self._generate_pop_sound('H', frequency=380.0, duration=0.12)  # Hydrogen - airy, slightly higher pitch
        self._generate_pop_sound('O', frequency=300.0, duration=0.18)  # Oxygen - medium pitch
        self._generate_pop_sound('N', frequency=270.0, duration=0.20)  # Nitrogen - slightly lower
        self._generate_pop_sound('C', frequency=240.0, duration=0.22)  # Carbon - lowest pitch

    def _generate_pop_sound(self, name: str, frequency: float, duration: float):
        """
        Generate a gentle, ASMR-like "pop" sound.
        
        We'll use:
        - A pure sine tone at the given frequency.
        - A soft exponential amplitude envelope that includes a quick fade-in and fade-out.
        - A tiny bit of very low-level noise to add a 'whispery' texture.
        """
        sample_rate = 44100
        samples = int(duration * sample_rate)
        
        # Time array
        t = np.linspace(0, duration, samples, endpoint=False)
        
        # Main sine wave
        wave = np.sin(2 * np.pi * frequency * t)
        
        # Add a tiny hint of soft noise to give a subtle texture (very low amplitude)
        noise = (np.random.normal(0, 0.01, samples))  # small random noise
        wave = wave * 0.95 + noise * 0.05
        
        # Create an envelope that fades in at the start and out at the end
        # Fade-in: first 10% of the sound, Fade-out: last 20% of the sound
        fade_in_duration = duration * 0.1
        fade_out_duration = duration * 0.2
        
        envelope = np.ones(samples)
        
        # Fade-in envelope
        fade_in_samples = int(fade_in_duration * sample_rate)
        if fade_in_samples > 0:
            envelope[:fade_in_samples] = np.linspace(0.0, 1.0, fade_in_samples)
        
        # Fade-out envelope
        fade_out_samples = int(fade_out_duration * sample_rate)
        if fade_out_samples > 0:
            envelope[-fade_out_samples:] = np.linspace(1.0, 0.0, fade_out_samples)
        
        # Apply envelope
        wave *= envelope
        
        # Normalize wave to prevent clipping and then convert to 16-bit PCM
        max_val = np.max(np.abs(wave))
        if max_val > 0:
            wave = wave / max_val  # normalize
        wave = np.int16(wave * 32767 * 0.3)  # scale down for softness
        
        # Make stereo by duplicating the channel
        stereo = np.column_stack((wave, wave))
        
        sound = pygame.mixer.Sound(stereo)
        sound.set_volume(self.master_volume)
        self._sound_cache[name] = sound

    def play_creation_sound(self, element_type: str):
        """Play the creation sound for a specific element"""
        current_time = pygame.time.get_ticks()
        if current_time - self._last_played_time < self._min_interval:
            return
            
        # Get element-specific sound or default
        sound = self._sound_cache.get(element_type, self._sound_cache['default'])
        
        # Slight random variation in volume to keep things organic
        volume_variation = np.random.uniform(0.9, 1.0)
        sound.set_volume(self.master_volume * volume_variation)
        
        # Play the sound
        sound.play()
        self._last_played_time = current_time

    def set_master_volume(self, volume: float):
        """Set the master volume (0.0 to 1.0)"""
        self.master_volume = max(0.0, min(1.0, volume))
        # Update all cached sounds
        for sound in self._sound_cache.values():
            sound.set_volume(self.master_volume)
