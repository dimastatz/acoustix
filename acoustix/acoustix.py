import librosa
import numpy as np
import json
import os
import sys
import soundfile as sf
import scipy.io.wavfile
import whisper  # Added whisper

# --- Global Model Cache ---
# No longer using a global variable for the model.
# It will be loaded in the main execution block and passed as an argument.

def get_speech_rate_wpm(model, wav_path, duration_sec):
    """
    Uses OpenAI's Whisper to perform ASR and calculate Words Per Minute (WPM).
    This is a pure function that depends only on its inputs.
    
    Args:
        model: The loaded Whisper model instance.
        wav_path (str): The file path to the audio file.
        duration_sec (float): The total duration of the audio in seconds.
        
    Returns:
        int: The calculated words per minute (WPM).
    """
    try:
        # Transcribe the audio
        # Using the file path is easier as Whisper handles resampling internally.
        result = model.transcribe(wav_path)
        
        text = result.get("text", "")
        words_count = len(text.split())
        
        # Calculate WPM
        duration_min = duration_sec / 60.0
        
        if duration_min > 0:
            wpm = words_count / duration_min
        else:
            wpm = 0
            
        return round(wpm)

    except Exception as e:
        # In a pure function, we avoid printing. We either re-raise or return a
        # fallback value, consistent with the original script's resilience.
        # print(f"Error during Whisper transcription: {e}") # Impure
        return 142  # Return placeholder on error

def get_emotion(y, sr):
    """
    Placeholder for Speech Emotion Recognition (SER).
    """
    # print("NOTE: 'emotion' requires a pre-trained SER model. Using placeholder.") # Impure
    # Placeholder values
    return {
        "calm": 0.52,
        "happy": 0.28,
        "angry": 0.12,
        "sad": 0.08
    }

def get_overlap_ratio(y, sr):
    """
    Placeholder for Speaker Diarization (identifying who speaks when).
    """
    # print("NOTE: 'overlap_ratio' requires a speaker diarization model. Using placeholder.") # Impure
    return 0.07  # Placeholder value

def get_engagement_index(pitch_variance, energy_variability, speech_rate, emotion_props):
    """
    Placeholder for a custom 'engagement' metric.
    """
    # print("NOTE: 'engagement_index' is a custom metric. Using placeholder logic.") # Impure
    # Example placeholder logic: 
    # (simple weighting of "happy" emotion and normalized pitch variance)
    engagement = emotion_props.get("happy", 0) + (pitch_variance / 50.0) # 50.0 is a magic number
    return min(max(engagement, 0.0), 1.0) * 0.81 # Scale and return placeholder

def analyze_audio(wav_path, whisper_model):
    """
    Analyzes a .wav file and extracts a dictionary of audio features.
    This function is pure and does not produce side effects.
    
    Args:
        wav_path (str): Path to the audio file.
        whisper_model: The loaded Whisper model instance.
        
    Returns:
        dict: A dictionary of extracted features.
        
    Raises:
        Exception: If the audio file cannot be loaded.
    """
    # print(f"Loading audio from {wav_path}...") # Impure
    try:
        # Load audio file. 'sr=None' preserves the original sample rate.
        y, sr = librosa.load(wav_path, sr=None)
    except Exception as e:
        # Propagate the error instead of printing and returning None.
        # The caller (main block) will handle the side effect (printing).
        raise e

    # --- 1. Duration ---
    duration_sec = librosa.get_duration(y=y, sr=sr)

    # --- 2. Speech Segments (VAD - Voice Activity Detection) ---
    # This is a simple energy-based VAD.
    # 'top_db=40' means segments quieter than 40dB below the max will be considered silence.
    # Adjust 40 to be more or less sensitive.
    segments = librosa.effects.split(y, top_db=40)
    speech_segments = len(segments)

    # --- 3. & 4. Pitch ---
    # Use PYIN algorithm to extract fundamental frequency (F0)
    # fmin and fmax define the expected pitch range for human speech
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        fill_na=np.nan  # Fill unvoiced frames with NaN
    )
    
    # Get only the F0 values from frames that were classified as "voiced"
    voiced_f0 = f0[voiced_flag]
    
    average_pitch_hz = float(np.nanmean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
    pitch_variance = float(np.nanstd(voiced_f0)) if len(voiced_f0) > 1 else 0.0

    # --- 5. & 6. Energy ---
    # Calculate Root-Mean-Square (RMS) energy
    rms = librosa.feature.rms(y=y)[0]
    # Convert to decibels (dB) for a more perceptually relevant scale
    rms_db = librosa.power_to_db(rms**2)
    
    mean_energy = float(np.mean(rms_db))
    energy_variability = float(np.std(rms_db))

    # --- 7. Speech Rate (Using Whisper) ---
    # We pass the path, duration, and the loaded model.
    speech_rate_wpm = get_speech_rate_wpm(whisper_model, wav_path, duration_sec)

    # --- 8. Emotion (Placeholder) ---
    emotion = get_emotion(y, sr)

    # --- 9. Overlap Ratio (Placeholder) ---
    overlap_ratio = get_overlap_ratio(y, sr)

    # --- 10. Engagement Index (Placeholder) ---
    engagement_index = get_engagement_index(
        pitch_variance, energy_variability, speech_rate_wpm, emotion
    )

    # --- Assemble Final Dictionary ---
    data = {
      "duration_sec": round(duration_sec, 2),
      "speech_segments": speech_segments,
      "average_pitch_hz": round(average_pitch_hz, 2),
      "pitch_variance": round(pitch_variance, 2),
      "mean_energy_db": round(mean_energy, 2), # Note: Changed to dB
      "energy_variability_db": round(energy_variability, 2), # Note: Changed to dB
      "speech_rate_wpm": speech_rate_wpm,
      "emotion": emotion,
      "overlap_ratio": overlap_ratio,
      "engagement_index": round(engagement_index, 2)
    }
    
    return data

def create_mock_file(filename="mock_audio.wav"):
    """
    Creates a simple, 5-second mock audio file with speech and silence.
    This function is inherently impure as its purpose is file I/O.
    """
    print(f"Creating mock file: {filename}...")
    sr = 16000  # 16kHz sample rate
    
    # 2 seconds of a 220Hz tone (speech)
    t_speech1 = np.linspace(0., 2., int(sr * 2), endpoint=False)
    speech1 = (np.sin(2. * np.pi * 220. * t_speech1) * 0.5)
    
    # 1 second of silence
    silence = np.zeros(int(sr * 1))
    
    # 1.5 seconds of a 300Hz tone (speech)
    t_speech2 = np.linspace(0., 1.5, int(sr * 1.5), endpoint=False)
    speech2 = (np.sin(2. * np.pi * 300. * t_speech2) * 0.4)
    
    # Concatenate parts
    final_signal = np.concatenate((speech1, silence, speech2))
    
    # Scale to 16-bit integer range
    final_signal_int16 = (final_signal * 32767).astype(np.int16)
    
    # Write to .wav file (side effect)
    scipy.io.wavfile.write(filename, sr, final_signal_int16)
    print(f"Mock file '{filename}' created.")


if __name__ == "__main__":
    # --- This block handles all side effects (I/O, printing, model loading) ---
    
    # Check if a file path was provided as a command-line argument
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if not os.path.exists(filepath):
            print(f"Error: File not found at '{filepath}'")
            sys.exit(1)
        elif not filepath.lower().endswith('.wav'):
             print(f"Error: File must be a .wav file. Got: '{filepath}'")
             sys.exit(1)
    else:
        # No file provided, use/create a mock file
        filepath = "mock_audio.wav"
        if not os.path.exists(filepath):
            create_mock_file(filepath) # Side effect

    try:
        # --- Load Model (Side Effect) ---
        print("Loading Whisper model (tiny) for analysis...")
        # Using "tiny" for speed. Other options: "base", "small", "medium", "large"
        whisper_model = whisper.load_model("tiny")
        print("Whisper model loaded.")
        
        # --- Run the pure analysis function ---
        print(f"Loading audio from {filepath}...")
        analysis_results = analyze_audio(filepath, whisper_model)
    
        # --- Print Results (Side Effect) ---
        if analysis_results:
            print("\n--- Analysis Results ---")
            print(json.dumps(analysis_results, indent=2))
            print("\n--------------------------")
            print("Note: 'mean_energy' and 'energy_variability' are in dB.")
            print("Note: 'emotion', 'overlap_ratio', and 'engagement_index' are placeholders.")
            print("Note: 'speech_rate_wpm' is calculated by Whisper.")

    except ImportError:
        print("\n--- ERROR ---")
        print("Required library 'openai-whisper' not found.")
        print("Please install it with: pip install openai-whisper")
        print("Whisper also requires 'ffmpeg' to be installed on your system.")
        sys.exit(1)
    except Exception as e:
        print(f"\n--- An error occurred ---")
        print(f"Error details: {e}")
        sys.exit(1)

