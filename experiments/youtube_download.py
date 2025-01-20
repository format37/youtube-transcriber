import logging
import pytubefix as pytube
import uuid
import os
import ffmpeg
from pydub import AudioSegment

# Set up logging 
logging.basicConfig(level=logging.INFO)

# Initialize logger
logger = logging.getLogger(__name__)


def extract_audio_wav(video_path: str) -> str:
    """
    Extracts audio from the given video file and saves it as a WAV file.
    Returns the path to the newly created WAV file, or an error message if extraction fails.
    """
    logger.info(f"Extracting audio from {video_path} as WAV...")
    audio_path = f"{uuid.uuid4()}.wav"

    try:
        (
            ffmpeg
            .input(video_path)
            .output(audio_path, format='wav', acodec='pcm_s16le')
            .run()
        )
        logger.info(f"Audio extracted to {audio_path}")
        return audio_path
    except Exception as e:
        msg = f"Error extracting audio to WAV: {str(e)}"
        logger.error(msg)
        return msg

def download_youtube_as_wav(youtube_url: str) -> str:
    """
    Downloads a YouTube video from the specified URL and converts it to a WAV file.
    Finally, removes the original video file and returns the path to the WAV file.

    Example:
        download_youtube_as_wav('https://www.youtube.com/watch?v=sELDIy0e6WE')
    """
    logger.info(f"Downloading YouTube video from {youtube_url}")
    video = pytube.YouTube(youtube_url).streams.get_highest_resolution()

    # Create a unique video filename
    video_filename = f"{uuid.uuid4()}.mp4"
    video.download(filename=video_filename)
    logger.info(f"Download complete: {video_filename}")

    # Extract WAV
    wav_path = extract_audio_wav(video_filename)

    # Remove original video file
    if os.path.exists(video_filename):
        os.remove(video_filename)
        logger.info(f"Removed original video file: {video_filename}")

    return wav_path


def main():
    filename = download_youtube_as_wav('https://www.youtube.com/watch?v=sELDIy0e6WE')
    print(filename)

if __name__ == "__main__":
    main()
