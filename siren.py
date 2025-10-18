import numpy as np
import wave

def generate_siren(filename='alert.wav', duration=3, sample_rate=44100):
    """
    Generate siren sound without FFmpeg
    """
    print("🔊 Generating siren sound...")
    
    # Time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create siren effect (frequency sweeping)
    audio = np.array([])
    
    # 8 cycles of high-low siren
    for i in range(8):
        # High frequency (1200 Hz)
        high_freq = np.sin(2 * np.pi * 1200 * t[:sample_rate//4])
        # Low frequency (600 Hz)
        low_freq = np.sin(2 * np.pi * 600 * t[:sample_rate//4])
        
        audio = np.concatenate([audio, high_freq, low_freq])
    
    # Normalize to 16-bit range
    audio = np.int16(audio * 32767 * 0.8)  # 0.8 for volume control
    
    # Save as WAV file
    with wave.open(filename, 'w') as wav_file:
        # Set parameters: channels, sample width, frame rate, frames, compression
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    
    print(f"✅ Siren sound generated: {filename}")
    print(f"Duration: {duration} seconds")
    print(f"Format: WAV (works with playsound)")

if __name__ == '__main__':
    generate_siren('alert.wav', duration=3)