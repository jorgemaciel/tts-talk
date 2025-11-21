import torch
import subprocess
import numpy as np
import os
import shutil

def read_audio_ffmpeg(file_path):
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("Error: ffmpeg not found in PATH")
        return None
        
    print(f"Using ffmpeg at: {ffmpeg_path}")
    
    cmd = [
        ffmpeg_path,
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"Erro ffmpeg ao ler {file_path}: {e}")
        return None

    return torch.from_numpy(
        np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    )

# Test with the existing mp3 file
test_file = "test_audio.wav"
if os.path.exists(test_file):
    print(f"Testing with {test_file}")
    wav = read_audio_ffmpeg(test_file)
    if wav is not None:
        print(f"Success! Tensor shape: {wav.shape}")
    else:
        print("Failed to read audio.")
else:
    print(f"Test file {test_file} not found.")
