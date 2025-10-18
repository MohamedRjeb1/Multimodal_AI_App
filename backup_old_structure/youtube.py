import os 
from dotenv import load_dotenv
import yt_dlp as youtube_dl
import glob


from model import Settings, get_settings

from yt_dlp import DownloadError

load_dotenv()
output_dir = "files/audio/"

ydl_config = {
    "format": "bestaudio/best",
    "postprocessors": [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ],
    "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
    "verbose": True
}


# check if output_dir exists, if not create it

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


print(f'Downloading video from {os.getenv("youtube_url")}')


try :
    with youtube_dl.YoutubeDL(ydl_config) as ydl:
        ydl.download([os.getenv("youtube_url")])
except DownloadError:
    with youtube_dl.YoutubeDL(ydl_config) as ydl:
        ydl.download([os.getenv("youtube_url")])


audio_file = glob.glob(os.path.join(output_dir, "*.mp3"))

audio_filename = audio_file[0]

print(audio_filename)