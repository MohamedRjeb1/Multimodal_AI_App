

import os
import openai
from test_audio import audio_filename
import whisper

audio = audio_filename
output_file = "files/transcripts/transcript.txt"

model = "whisper-1"


# transcribe with API

print("converting audio to text....")


model = whisper.load_model("small")           # tiny, base, small, medium, large
result = model.transcribe(audio, language="fr")
transcript = result["text"]
print(transcript)



if output_file is None:
    print("No output file specified. Transcript will not be saved.")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
else:
    with open(output_file, "w") as f:
        f.write(transcript)
    print(f"Transcript saved to {output_file}")
    
