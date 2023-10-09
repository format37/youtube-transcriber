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
import math
from pydub import AudioSegment
import subprocess
from telebot import TeleBot

# Set up logging 
logging.basicConfig(level=logging.INFO)

# Initialize logger
logger = logging.getLogger(__name__)

app = FastAPI()

"""class VideoUrl(BaseModel):
    url: str"""

class TranscriptionRequest(BaseModel):
    url: str
    chat_id: str
    message_id: str
    bot_token: str
    
@app.post("/transcribe")
async def transcribe(request_data: TranscriptionRequest):
    # try:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    if OPENAI_API_KEY == '':
        raise Exception("OPENAI_API_KEY environment variable not found")
        return {"error": "OPENAI_API_KEY environment variable not found"}

    url = request_data.url
    chat_id = int(request_data.chat_id)
    message_id = int(request_data.message_id)
    bot_token = request_data.bot_token

    # Log start of download
    """logger.info("Starting video download from url: " + url)
    logger.info("Chat id: " + str(chat_id))
    logger.info("Message id: " + str(message_id))
    logger.info("Bot token: " + bot_token)"""
    

    # Initialize the bot
    bot = TeleBot(bot_token)

    bot.edit_message_text(
            "Downloading video..",
            chat_id=chat_id,
            message_id=message_id
        )
    
    # Download video
    filename = download_video(url)

    bot.edit_message_text(
            "Extracting audio..",
            chat_id=chat_id,
            message_id=message_id
        )

    # Extract audio
    audio_path = extract_audio(filename)

    if 'Error' in audio_path:
        bot.edit_message_text(
            audio_path,
            chat_id=chat_id,
            message_id=message_id
        )
        return {"transcription": audio_path}

    # Remove video
    os.remove(filename)

    # Transcribe audio
    text = recognize_whisper(
        audio_path, 
        OPENAI_API_KEY,
        chat_id,
        message_id,
        bot
        )

    # Remove audio
    os.remove(audio_path)

    # Log transcription length
    logger.info("Transcription length: " + str(len(text)))

    # Edit message that Job has finished with text len
    bot.edit_message_text(
            f"Transcription finished. Text length: {len(text)}",
            chat_id=chat_id,
            message_id=message_id
        )
    
    return {"transcription": text}
    """except Exception as e:
        # Log error
        logger.error("Error: " + str(e))
        # Edit message that Job has finished with error
        bot.edit_message_text(
                f"Error: {str(e)}",
                chat_id=chat_id,
                message_id=message_id
            )
        return {"transcription": str(e)}"""


def download_video(url):
    youtube = pytube.YouTube(url)
    video = youtube.streams.get_highest_resolution()

    unique_id = str(uuid.uuid4())
    outname = unique_id + ".mp4"

    video.download(filename=outname)
    
    return outname


def extract_audio_fast(video_path):

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
        return 'Error extracting audio: '+str(e)


def extract_audio(video_path):
    logger.info(f"Extracting audio from {video_path}")

    # Generate an unique mp3 audio file name
    audio_path = str(uuid.uuid4()) + ".mp3"

    # Command to extract audio using ffmpeg
    cmd = [
        'ffmpeg',
        '-i', video_path,    # input video path
        '-q:a', '0',         # best audio quality
        '-map', 'a',         # map only audio stream
        '-y',                # overwrite output file if it exists
        audio_path           # output audio path
    ]

    try:
        # Run the command and wait for it to complete
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Audio extracted to {audio_path}")
        return audio_path

    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e.stderr.decode('utf-8')}")
        return None

def split_audio_ffmpeg(audio_path, chunk_length=10*60):
    """
    Splits the audio file into chunks using ffmpeg.
    Returns a list of paths to the chunks.
    """
    # Get the duration of the audio in seconds
    cmd_duration = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {audio_path}"
    duration = float(os.popen(cmd_duration).read())

    # Calculate number of chunks
    chunks_count = int(duration // chunk_length) + (1 if duration % chunk_length > 0 else 0)

    chunk_paths = []

    for i in range(chunks_count):
        start_time = i * chunk_length
        # Unique filename for the chunk
        chunk_filename = f"/tmp/{uuid.uuid4()}.mp3"
        # Use ffmpeg to extract a chunk of the audio
        cmd_extract = f"ffmpeg -ss {start_time} -t {chunk_length} -i {audio_path} -acodec copy {chunk_filename}"
        os.system(cmd_extract)

        chunk_paths.append(chunk_filename)

    return chunk_paths


def recognize_whisper(
    audio_path, 
    api_key,
    chat_id,
    message_id,
    bot
    ):
    
    # Split the audio into chunks
    chunk_paths = split_audio_ffmpeg(audio_path)

    full_text = ""

    for idx, chunk_path in enumerate(chunk_paths):

        logger.info(f"Processing chunk {idx+1} of {len(chunk_paths)}")

        bot.edit_message_text(
            f"Transcribing audio.. ({idx+1}/{len(chunk_paths)})",
            chat_id=chat_id,
            message_id=message_id
        )

        # Load chunk into memory using pydub
        chunk_audio = AudioSegment.from_file(chunk_path)

        # Transcribe chunk
        text = transcribe_chunk(chunk_path, api_key)
        full_text += text

        # Remove the temporary chunk file
        os.remove(chunk_path)

    return full_text


def recognize_whisper_memory_expensive(
    audio_path, 
    api_key,
    chat_id,
    message_id,
    bot
    ):

    audio = AudioSegment.from_file(audio_path)

    # If audio len is bigger than 3 hours, decline the transcribation
    if len(audio) > 3 * 60 * 60 * 1000:
        logger.info("Declined: Audio is bigger than 3 hours")
        return "Declined: Audio is bigger than 3 hours"


    chunk_size_ms = 10 * 60 * 1000 # 10 minutes

    start = 0
    end = chunk_size_ms

    full_text = ""

    chunks_count = math.ceil(len(audio) / chunk_size_ms)
    current_chunk = 1

    while start < len(audio):

        logger.info(f"Processing chunk from {start/1000} to {end/1000} second")

        bot.edit_message_text(
            f"Transcribing audio.. ({current_chunk}/{chunks_count})",
            chat_id=chat_id,
            message_id=message_id
        )

        chunk = audio[start:end]

        # Export and transcribe chunk 
        unique_id = str(uuid.uuid4())
        chunk_path = f"/tmp/{unique_id}.mp3"
        chunk.export(chunk_path, format="mp3")

        text = transcribe_chunk(chunk_path, api_key)

        full_text += text

        # Update start and end for next chunk
        start += chunk_size_ms
        end += chunk_size_ms

        os.remove(chunk_path)

    return full_text



def transcribe_chunk(audio_path, api_key):

    openai.api_key = api_key

    with open(audio_path, "rb") as audio_file:
        response = openai.Audio.transcribe(
        file=audio_file,  
        model="whisper-1",
        response_format="text"
        )

    return response


def main():
    uvicorn.run(app, host="0.0.0.0", port=8702)


if __name__ == "__main__":
    main()
