from pydub import AudioSegment
from scipy import signal
import numpy as np
import soundfile as sf

def apply_frequency_adjustments(audio_path, output_path):
    # Load the distorted and noisy audio using pydub
    audio = AudioSegment.from_file(audio_path)
    
    # Convert audio to numpy array
    audio_data = np.array(audio.get_array_of_samples())
    
    # Sampling parameters
    sample_rate = audio.frame_rate
    
    # Designing filters
    b_low, a_low = signal.butter(4, 500 / (sample_rate / 2), btype='low', analog=False)
    b_high, a_high = signal.butter(4, 6000 / (sample_rate / 2), btype='high', analog=False)
    b_band, a_band = signal.butter(4, [600 / (sample_rate / 2), 5000 / (sample_rate / 2)], btype='band', analog=False)
    
    # Apply filters
    filtered_low = signal.lfilter(b_low, a_low, audio_data)
    filtered_high = signal.lfilter(b_high, a_high, audio_data)
    filtered_band = signal.lfilter(b_band, a_band, audio_data)
    
    # Applying gain adjustments
    gain_low = 10 ** (-15 / 20)  # -15dB to gain factor
    gain_high = 10 ** (-15 / 20)  # -15dB to gain factor
    gain_band = 10 ** (2 / 20)    # 2dB to gain factor
    
    adjusted_low = filtered_low * gain_low
    adjusted_high = filtered_high * gain_high
    adjusted_band = filtered_band * gain_band
    
    # Combine the adjusted frequency bands
    final_audio_data = adjusted_low + adjusted_high + adjusted_band
    
    # Convert the adjusted numpy array back to an AudioSegment
    adjusted_audio = AudioSegment(
        final_audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    
    # Save the audio with frequency adjustments
    adjusted_audio.export(output_path, format="wav")

# Paths to input and output audio files
distorted_audio_path = "phone_microphone_distorted.wav"
output_audio_path = "frequency_adjusted_audio.wav"

apply_frequency_adjustments(distorted_audio_path, output_audio_path)
print("Frequency adjustments applied to audio.")



#---

from pydub import AudioSegment
from scipy import signal
import numpy as np

def apply_high_pass_filter(audio_path, output_path):
    # Load the distorted and noisy audio using pydub
    audio = AudioSegment.from_file(audio_path)
    
    # Convert audio to numpy array
    audio_data = np.array(audio.get_array_of_samples())
    
    # Sampling parameters
    original_sample_rate = audio.frame_rate
    target_sample_rate = 12000  # Adjust as needed
    
    # Downsample the audio
    audio_data = signal.resample(audio_data, int(len(audio_data) * target_sample_rate / original_sample_rate))
    
    # Designing the high-pass filter
    order = 4
    cutoff_freq = 6000
    normalized_cutoff_freq = cutoff_freq / (target_sample_rate / 2)
    b_high, a_high = signal.butter(order, normalized_cutoff_freq, btype='high', analog=False)
    
    # Apply the high-pass filter
    filtered_audio = signal.lfilter(b_high, a_high, audio_data)
    
    # Convert the filtered numpy array back to an AudioSegment
    filtered_audio_segment = AudioSegment(
        filtered_audio.tobytes(),
        frame_rate=target_sample_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    
    # Save the high-pass filtered audio
    filtered_audio_segment.export(output_path, format="wav")

# Paths to input and output audio files
distorted_audio_path = "phone_microphone_distorted.wav"
output_audio_path = "high_pass_filtered_audio.wav"

apply_high_pass_filter(distorted_audio_path, output_audio_path)
print("High-pass filter applied to audio.")
