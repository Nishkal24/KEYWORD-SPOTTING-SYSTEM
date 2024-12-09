#Dataset (Sentence Audio Generator):

from gtts import gTTS
import os

# Define the list of sentences
sentences = ["hello everyone, how are you doing", "start doing your work properly"]

# Directory to save the audio files
output_dir = "sample_sen1"
os.makedirs(output_dir, exist_ok=True)

# Generate and save one audio file for each sentence
for i, sentence in enumerate(sentences):
    # Generate TTS audio for the sentence
    tts = gTTS(text=sentence, lang='en')
    
    # Define the filename
    filename = os.path.join(output_dir, f"sentence_{i + 1}.mp3")
    
    # Save the audio file
    tts.save(filename)
    
    print(f"Saved: {filename}")

print("Audio files generated successfully.")
