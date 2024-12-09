#Dataset (Single Word Audio Generation):
from gtts import gTTS
import os
from pydub import AudioSegment
import random
# Define the list of keywords
keywords = ["hello", "sorry", "right", "left", "start", "stop"]
# Directory to save the audio files
output_dir = "samples_5"
os.makedirs(output_dir, exist_ok=True)
# Generate and save audio files for each keyword
num_samples_per_keyword = 10  # Define the number of samples per keyword
def add_noise(audio_segment):
    """Adds random noise to the audio segment."""
    noise_level = random.randint(-3, 3)  # Very subtle noise level range
    return audio_segment + noise_level
def change_speed(audio_segment, speed=1.0):
    """Changes the speed of the audio."""
    return audio_segment.speedup(playback_speed=speed)
def change_pitch(audio_segment, semitones=0):
    """Changes the pitch of the audio."""
    return audio_segment._spawn(audio_segment.raw_data, overrides={
         "frame_rate": int(audio_segment.frame_rate * (2.0 ** (semitones / 12.0)))
    }).set_frame_rate(audio_segment.frame_rate)
# List of TTS languages for variety (you can add more accents if supported)
voices = ['en', 'en-au', 'en-uk', 'en-us']
for keyword in keywords:
    for i in range(num_samples_per_keyword):
        # Randomly select a voice/accent for diversity
        selected_voice = random.choice(voices)
        # Generate TTS audio for the keyword with the selected voice
        tts = gTTS(text=keyword, lang=selected_voice)
        tts.save("temp.mp3")
        # Load the audio file using pydub
        audio = AudioSegment.from_mp3("temp.mp3")
        # Randomly adjust speed (between 0.95x to 1.05x), add noise, and change pitch
        speed = random.uniform(0.95, 1.05)  # More subtle speed adjustment
        audio = change_speed(audio, speed)
        # Add noise
        audio = add_noise(audio)
        # Change pitch randomly between -0.5 to +0.5 semitones for subtle variations
        pitch_shift = random.uniform(-0.5, 0.5)
        audio = change_pitch(audio, pitch_shift)
        # Save the modified audio file
        filename = os.path.join(output_dir, f"{keyword}_{i + 1}.mp3")
        audio.export(filename, format="mp3")
        print(f"Saved: {filename}")
print("Audio files with variations generated successfully.")