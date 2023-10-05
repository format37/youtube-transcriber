import logging
import openai
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pytube
import uuid
import os
import ffmpeg
import uvicorn

# Set up logging 
logging.basicConfig(level=logging.INFO)

# Initialize logger
logger = logging.getLogger(__name__)

app = FastAPI()

class VideoUrl(BaseModel):
    url: str
    
@app.post("/transcribe")
async def transcribe(video: VideoUrl):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    if OPENAI_API_KEY == '':
        raise Exception("OPENAI_API_KEY environment variable not found")
        return {"error": "OPENAI_API_KEY environment variable not found"}

    url = video.url

    # Log start of download
    logger.info("Starting video download from url: " + url)
    
    # Download video
    filename = download_video(url)

    # Extract audio
    audio_path = extract_audio(filename)

    # Remove video
    os.remove(filename)

    # Transcribe audio
    text = recognize_whisper(audio_path, OPENAI_API_KEY)

    # Remove audio
    os.remove(audio_path)

    # Log transcription length
    logger.info("Transcription length: " + str(len(text)))
    
    return {"transcription": text}


def download_video(url):
    youtube = pytube.YouTube(url)
    video = youtube.streams.get_highest_resolution()

    unique_id = str(uuid.uuid4())
    outname = unique_id + ".mp4"

    video.download(filename=outname)
    
    return outname


def extract_audio(video_path):

    logger.info(f"Extracting audio from {video_path}")

    # Generate an unique mp3 audio file name
    audio_path = str(uuid.uuid4()) + ".mp3"
    
    try:
        # Extract audio using ffmpeg
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, audio_path)
        ffmpeg.run(stream)
        # Log what patch
        logger.info(f"Audio extracted to {audio_path}")
        return audio_path
    
    except Exception as e:
        logger.error("Error extracting audio: " + str(e))
        return None


def recognize_whisper(audio_path, api_key):
    logger.info(f"Transcribing audio at {audio_path}")
    
    if audio_path is None:
        print("Error extracting audio")
        return

    # OpenAI's Python package uses environment variables for API keys,
    # but since you're reading from a file, we'll set it directly.
    openai.api_key = api_key

    # Load the audio file
    logging.info(f'Loading audio file from {audio_path}...')
    with open(audio_path, "rb") as audio_file:
        # Transcribe the audio
        logging.info('Transcribing the audio...')
        response = openai.Audio.transcribe(
            file=audio_file,
            model="whisper-1",
            response_format="text",
            language="ru"
        )

    # Directly return the response as it's already a string
    return response


def main():
    uvicorn.run(app, host="0.0.0.0", port=8702)


if __name__ == "__main__":
    main()
