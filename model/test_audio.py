import glob
import os 

output_dir = "files/audio/"

audio_file = glob.glob(os.path.join(output_dir, "*.webm"))

audio_filename = audio_file[0]

print(audio_filename)
