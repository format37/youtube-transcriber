import logging
from openai import OpenAI
from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pytubefix as pytube
import uuid
import os
import ffmpeg
import uvicorn
import math
from pydub import AudioSegment
import subprocess
import telebot
import requests
import subprocess
import re
import json

# Set up logging 
logging.basicConfig(level=logging.INFO)

# Initialize logger
logger = logging.getLogger(__name__)

app = FastAPI(session_timeout=60*60) # 1 hour timeout

server_api_uri = 'http://localhost:8081/bot{0}/{1}'
# if server_api_uri != '':
telebot.apihelper.API_URL = server_api_uri
logger.info(f'Setting API_URL: {server_api_uri}')

server_file_url = 'http://0.0.0.0:8081'
# if server_file_url != '':
telebot.apihelper.FILE_URL = server_file_url
logger.info(f'Setting FILE_URL: {server_file_url}')

token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
# Initialize the bot
bot = telebot.TeleBot(token)

# Add this at the beginning of your script, after the imports
USER_LANGUAGES = {}

# Function to load user languages from a file
def load_user_languages():
    global USER_LANGUAGES
    try:
        with open('user_languages.json', 'r') as f:
            USER_LANGUAGES = json.load(f)
    except FileNotFoundError:
        USER_LANGUAGES = {}

# Function to save user languages to a file
def save_user_languages():
    with open('user_languages.json', 'w') as f:
        json.dump(USER_LANGUAGES, f)

# Call this function at the start of your script
load_user_languages()

class TranscriptionRequest(BaseModel):
    url: str
    chat_id: int
    message_id: int
    bot_token: str

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
if OPENAI_API_KEY == '':
    raise Exception("OPENAI_API_KEY environment variable not found")
client = OpenAI(api_key=OPENAI_API_KEY)

def get_language_name(lang_code):
    language_names = {
        'af': 'Afrikaans', 'ar': 'Arabic', 'hy': 'Armenian', 'az': 'Azerbaijani',
        'be': 'Belarusian', 'bs': 'Bosnian', 'bg': 'Bulgarian', 'ca': 'Catalan',
        'zh': 'Chinese', 'hr': 'Croatian', 'cs': 'Czech', 'da': 'Danish',
        'nl': 'Dutch', 'en': 'English', 'et': 'Estonian', 'fi': 'Finnish',
        'fr': 'French', 'gl': 'Galician', 'de': 'German', 'el': 'Greek',
        'he': 'Hebrew', 'hi': 'Hindi', 'hu': 'Hungarian', 'is': 'Icelandic',
        'id': 'Indonesian', 'it': 'Italian', 'ja': 'Japanese', 'kn': 'Kannada',
        'kk': 'Kazakh', 'ko': 'Korean', 'lv': 'Latvian', 'lt': 'Lithuanian',
        'mk': 'Macedonian', 'ms': 'Malay', 'mi': 'Maori', 'mr': 'Marathi',
        'ne': 'Nepali', 'no': 'Norwegian', 'fa': 'Persian', 'pl': 'Polish',
        'pt': 'Portuguese', 'ro': 'Romanian', 'ru': 'Russian', 'sr': 'Serbian',
        'sk': 'Slovak', 'sl': 'Slovenian', 'es': 'Spanish', 'sw': 'Swahili',
        'sv': 'Swedish', 'tl': 'Tagalog', 'ta': 'Tamil', 'th': 'Thai',
        'tr': 'Turkish', 'uk': 'Ukrainian', 'ur': 'Urdu', 'vi': 'Vietnamese',
        'cy': 'Welsh'
    }
    return language_names.get(lang_code, 'Unknown')

@app.post("/message")
async def call_message(request: Request, authorization: str = Header(None)):
    logger.info('post: message')

    lang_list = [
                'af', 'ar', 'hy', 'az', 'be', 'bs', 'bg', 'ca', 'zh', 
                'hr', 'cs', 'da', 'nl', 'en', 'et', 'fi', 'fr', 
                'gl', 'de', 'el', 'he', 'hi', 'hu', 'is', 'id', 
                'it', 'ja', 'kn', 'kk', 'ko', 'lv', 'lt', 
                'mk', 'ms', 'mi', 'mr', 'ne', 'no', 'fa',
                'pl', 'pt', 'ro', 'ru', 'sr', 'sk',
                'sl', 'es', 'sw', 'sv', 'tl', 
                'ta', 'th', 'tr', 'uk', 
                'ur', 'vi', 'cy'
            ]
    
    token = None
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]

    if token:
        logger.info(f'Bot token: {token}')
        pass
    else:
        answer = 'Bot token not found. Please contact the administrator.'
        # if not private chat, return empty
        if message['chat']['type'] != 'private':
            return JSONResponse(content={
                "type": "empty",
                "body": ""
                })
        else:
            return JSONResponse(content={
                "type": "text",
                "body": str(answer)
            })

    message = await request.json()
    logger.info(f'message: {message}')

    # Return if it is a group
    if message['chat']['type'] != 'private':
        return JSONResponse(content={
            "type": "empty",
            "body": ""
            })

    answer = "Please, send video, audio, or youtube link to transcribe."
    data_path = './data/'

    # Check if user is in user_list
    # Read user_list from ./data/users.txt
    with open(data_path + 'users.txt', 'r') as f:
        user_list = f.read().splitlines()
    if str(message['from']['id']) not in user_list:
        answer = "You are not authorized to use this bot.\n"
        answer += "Please forward this message to the administrator.\n"
        answer += f'User id: {message["from"]["id"]}'
        return JSONResponse(content={
            "type": "text",
            "body": str(answer)
            })
    if 'audio' in message or \
        'voice' in message or \
        'video' in message or \
        'video_note' in message or \
        (
            'document' in message and \
            'mime_type' in message['document'] and \
            'audio' in message['document']['mime_type'])\
    :
        try:
            logger.info('audio, voice, video, or video_note found')
            
            if 'audio' in message:
                key = 'audio'
            elif 'voice' in message:
                key = 'voice'
            elif 'video' in message:
                key = 'video'
            elif 'video_note' in message:
                key = 'video_note'
            elif 'document' in message and \
                'mime_type' in message['document'] and \
                'audio' in message['document']['mime_type']:
                key = 'document'
            else:
                return JSONResponse(content={
                    "type": "text",
                    "body": "Unsupported format."
                })

            file_id = message[key]['file_id']
            logger.info(f'file_id: {file_id}')

            original_message_id = message['message_id']
            chat_id = message['chat']['id']
            # Retrieve message object from original_message_id
            message_text = "Job started. Please wait for transcription to be completed.\nDownloading file.."
            update_message = send_reply(token, chat_id, original_message_id, message_text)
            # logger.info("["+str(chat_id)+"] Update message: " + str(update_message))
            message_id = update_message['result']['message_id']

            try:
                file_info = bot.get_file(file_id)
                logger.info(f'file_id: {file_id}')
                logger.info(f'file_info: {file_info}')
                logger.info(f'file_path: {file_info.file_path}')
                # Download the file contents 
                with open(file_info.file_path, 'rb') as f:
                    file_bytes = f.read()
            except Exception as e:
                logger.error(f'Error downloading file: {e}')
                bot.edit_message_text(
                    f"Canceled. Unable to download file. {e}",
                    chat_id=chat_id,
                    message_id=message_id
                )
                return JSONResponse(content={
                    "type": "empty",
                    "body": ""
                })

            # After downloading the file but before processing
            logger.info(f'file_bytes: {len(file_bytes)}')
            # Check if file is too large (e.g., over 50MB)
            if len(file_bytes) > 50 * 1024 * 1024:  # 50MB in bytes
                bot.edit_message_text(
                    "File is too large. Please upload a file smaller than 50MB.",
                    chat_id=chat_id,
                    message_id=message_id
                )
                return JSONResponse(content={
                    "type": "empty",
                    "body": ""
                })

            if 'audio' in message:
                file_name = message[key]['file_name']
            elif 'voice' in message:
                file_name = 'temp.ogg'
            elif 'video' or 'video_note' in message:
                file_name = 'temp.mp4'
            else:
                return JSONResponse(content={
                    "type": "text",
                    "body": "Unsupported format"
                })
            # Add uuid before file_name
            file_name = f'{uuid.uuid4().hex}_{file_name}'
            file_path = os.path.join(data_path, file_name)
            """with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file_bytes, buffer)"""
            with open(file_path, "wb") as f:
                f.write(file_bytes)

            # Re-encode to mp3 before transcribing
            converted_path = reencode_to_mp3(file_path)

            # Now call transcribe_audio_file on the MP3
            transcribe_audio_file(converted_path, bot, chat_id, message_id)

            # Optionally remove the original file
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return JSONResponse(content={
                "type": "empty",
                "body": ""
                })
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            try:
                bot.edit_message_text(
                    f"Error processing your file: {str(e)}",
                    chat_id=chat_id,
                    message_id=message_id
                )
            except:
                pass
            return JSONResponse(content={
                "type": "empty",
                "body": ""
                })

    if 'text' in message:
        if message['text'].lower() == '/languages':
            # Create a formatted list of languages
            lang_info = "Available languages:\n\n"
            for lang in lang_list:
                lang_info += f"/{lang} - {get_language_name(lang)}\n"
            
            return JSONResponse(content={
                "type": "text",
                "body": lang_info
            })
        # Add language setting command handling
        if message['text'].startswith('/'):
            lang = message['text'][1:].lower()
            if lang in lang_list:
                USER_LANGUAGES[str(message['from']['id'])] = lang
                save_user_languages()
                answer = f"Language set to {lang}"
                return JSONResponse(content={
                    "type": "text",
                    "body": str(answer)
                })
        # Add user CMD
        if message['text'].startswith('/add'):
            # Check is current user in atdmins.txt
            admins = []
            with open(data_path + 'admins.txt', 'r') as f:
                admins = f.read().splitlines()
            if str(message['from']['id']) not in admins:
                answer = "You are not authorized to use this command."
                return JSONResponse(content={
                    "type": "text",
                    "body": str(answer)
                    })
            # split cmd from format /add <user_id>
            cmd = message['text'].split(' ')
            if len(cmd) != 2:
                answer = "Invalid command format. Please use /add <user_id>."
                return JSONResponse(content={
                    "type": "text",
                    "body": str(answer)
                    })
            # add user_id to user_list
            user_id = cmd[1]
            user_list.append(user_id)
            # write user_list to ./data/users.txt
            with open(data_path + 'users.txt', 'w') as f:
                f.write('\n'.join(user_list))
            answer = f'User {user_id} added successfully.'        

        # Remove user CMD
        elif message['text'].startswith('/remove'):
            # Check is current user in atdmins.txt
            admins = []
            with open(data_path + 'admins.txt', 'r') as f:
                admins = f.read().splitlines()
            if str(message['from']['id']) not in admins:
                answer = "You are not authorized to use this command."
                return JSONResponse(content={
                    "type": "text",
                    "body": str(answer)
                    })
            # split cmd from format /remove <user_id>
            cmd = message['text'].split(' ')
            if len(cmd) != 2:
                answer = "Invalid command format. Please use /remove <user_id>."
                return JSONResponse(content={
                    "type": "text",
                    "body": str(answer)
                    })
            # remove user_id from user_list
            user_id = cmd[1]
            user_list.remove(user_id)
            # write user_list to ./data/users.txt
            with open(data_path + 'users.txt', 'w') as f:
                f.write('\n'.join(user_list))
            answer = f'User {user_id} removed successfully.'

        elif youtubelink_in_text(message['text']):
            # Regular expression to find URLs
            urls = re.findall(r'(https?://\S+)', message['text'])

            logger.info(f'determined urls: {urls}')

            transcription_request = TranscriptionRequest(
                url=urls[0],
                chat_id=message['chat']['id'],
                message_id=message['message_id'],
                bot_token=token
            )
            transcribe(transcription_request)
            return JSONResponse(content={
                "type": "empty",
                "body": ""
                })
        else:
            answer = 'Please send a youtube link to transcribe the video to text.'

    return JSONResponse(content={
            "type": "text",
            "body": str(answer)
            })

def youtubelink_in_text(text):
    if "https://www.youtube.com/" in text or \
        "https://youtube.com/" in text or \
        "https://www.youtu.be/" in text or \
        "https://youtu.be/" in text:
        return True
    return False


def send_reply(bot_token, chat_id, message_id, text):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': text,
        'reply_to_message_id': message_id
    }
    response = requests.post(url, data=payload)
    return response.json()


def transcribe(request_data: TranscriptionRequest):
    try:

        bot_token = request_data.bot_token
        url = request_data.url
        chat_id = request_data.chat_id

        original_message_id = request_data.message_id
        # Retrieve message object from original_message_id
        message_text = "Job started. Please wait for transcription to be completed."
        update_message = send_reply(bot_token, chat_id, original_message_id, message_text)
        # logger.info("["+str(chat_id)+"] Update message: " + str(update_message))
        message_id = update_message['result']['message_id']

        # Log start of download
        logger.info("["+str(chat_id)+"] Starting video download from url: " + url)    

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
        # Remove video
        os.remove(filename)
        transcribe_audio_file(audio_path, bot, chat_id, message_id)

        return JSONResponse(content={
                "type": "empty",
                "body": ""
                })

    except Exception as e:
        logger.error("Error: " + str(e))
        return JSONResponse(content={
                "type": "text",
                "body": str(e)
                })


def transcribe_audio_file(audio_path, bot, chat_id, message_id):
    if 'Error' in audio_path:
        bot.edit_message_text(
            audio_path,
            chat_id=chat_id,
            message_id=message_id
        )
        return {"transcription": audio_path}
        
    # Check audio duration before proceeding
    try:
        cmd_duration = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {audio_path}"
        duration = float(os.popen(cmd_duration).read())
        
        # If audio is longer than 3 hours (10800 seconds), decline processing
        if duration > 10800:  # 3 hours in seconds
            message = "Audio file is too long (over 3 hours). Please upload a shorter file."
            logger.info(f"[{chat_id}] {message}")
            
            bot.edit_message_text(
                message,
                chat_id=chat_id,
                message_id=message_id
            )
            return {"transcription": message}
    except Exception as e:
        logger.error(f"[{chat_id}] Error checking audio duration: {e}")
    
    # Transcribe audio
    logger.info("["+str(chat_id)+"] Transcribing audio file..")
    text = recognize_whisper(
        audio_path, 
        OPENAI_API_KEY,
        chat_id,
        message_id,
        bot
        )

    # Remove audio
    logger.info("["+str(chat_id)+"] Removing audio file..")
    os.remove(audio_path)

    # Log transcription length
    logger.info("["+str(chat_id)+"] Transcription length: " + str(len(text)))

    # Edit message that Job has finished with text len
    bot.edit_message_text(
            f"Transcription finished. Text length: {len(text)}",
            chat_id=chat_id,
            message_id=message_id
        )

    # Send the transcription
    filename = f'./data/{uuid.uuid4().hex}.txt'

    with open(filename, 'w') as f:
        f.write(text)

    with open(filename, 'rb') as f:
        bot.send_document(
            chat_id, 
            f
        )
    os.remove(filename)


def download_video(url):
    youtube = pytube.YouTube(url)
    logger.info(f"calling: youtube.streams.get_highest_resolution() with url: {url}")
    video = youtube.streams.get_highest_resolution()

    unique_id = str(uuid.uuid4())
    outname = unique_id + ".mp4"
    logger.info(f"calling: video.download() with outname: {outname}")
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
        return 'Error extracting audio: '+str(e)


def extract_audio_subprocess(video_path):
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
    try:
        # Get the duration of the audio in seconds
        cmd_duration = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {audio_path}"
        duration_output = os.popen(cmd_duration).read().strip()
        
        if not duration_output:
            logger.error(f"Failed to get duration for {audio_path}")
            return []
            
        duration = float(duration_output)
        logger.info(f"Audio duration: {duration} seconds")
        
        # Calculate number of chunks
        chunks_count = int(duration // chunk_length) + (1 if duration % chunk_length > 0 else 0)
        logger.info(f"Splitting into {chunks_count} chunks")
        
        chunk_paths = []
        
        for i in range(chunks_count):
            start_time = i * chunk_length
            # Unique filename for the chunk
            chunk_filename = f"/tmp/{uuid.uuid4()}.mp3"
            # Use ffmpeg to extract a chunk of the audio
            cmd_extract = f"ffmpeg -ss {start_time} -t {chunk_length} -i {audio_path} -acodec copy {chunk_filename}"
            
            result = os.system(cmd_extract)
            if result != 0:
                logger.error(f"Error extracting chunk {i+1} with command: {cmd_extract}")
                continue
                
            # Verify the chunk file exists and has a non-zero size
            if os.path.exists(chunk_filename) and os.path.getsize(chunk_filename) > 0:
                chunk_paths.append(chunk_filename)
            else:
                logger.error(f"Chunk file {chunk_filename} does not exist or is empty")
        
        logger.info(f"Successfully created {len(chunk_paths)} chunks")
        return chunk_paths
        
    except Exception as e:
        logger.error(f"Error in split_audio_ffmpeg: {str(e)}")
        return []


def recognize_whisper(
    audio_path, 
    api_key,
    chat_id,
    message_id,
    bot
    ):
    try:
        # Split the audio into chunks
        chunk_paths = split_audio_ffmpeg(audio_path)
        
        if not chunk_paths:
            error_msg = "Failed to split audio file into chunks"
            logger.error(f"[{chat_id}] {error_msg}")
            bot.edit_message_text(
                error_msg,
                chat_id=chat_id,
                message_id=message_id
            )
            return error_msg

        full_text = ""
        
        for idx, chunk_path in enumerate(chunk_paths):
            try:
                logger.info(f"[{chat_id}] Processing chunk {idx+1} of {len(chunk_paths)}")
                
                bot.edit_message_text(
                    f"Transcribing audio.. ({idx+1}/{len(chunk_paths)})",
                    chat_id=chat_id,
                    message_id=message_id
                )
                
                # Transcribe chunk
                text = transcribe_chunk(chunk_path, chat_id)
                full_text += text
                
            except Exception as chunk_error:
                logger.error(f"[{chat_id}] Error processing chunk {idx+1}: {str(chunk_error)}")
                full_text += f"\n[Error in chunk {idx+1}]\n"
            finally:
                # Ensure chunk file is removed even if there's an error
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
        
        return full_text
        
    except Exception as e:
        error_msg = f"Error in transcription process: {str(e)}"
        logger.error(f"[{chat_id}] {error_msg}")
        return error_msg


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

        text = transcribe_chunk(chunk_path, chat_id)

        full_text += text

        # Update start and end for next chunk
        start += chunk_size_ms
        end += chunk_size_ms

        os.remove(chunk_path)

    return full_text



def transcribe_chunk(audio_path, user_id):
    user_lang = USER_LANGUAGES.get(str(user_id), 'en')  # Default to English if not set
    
    try:
        with open(audio_path, "rb") as audio_file:
            # Add timeout parameters to avoid hanging
            response = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio_file, 
                temperature=0,
                response_format="text",
                language=user_lang,
                timeout=300  # 5 minute timeout
            )
        return response
    except Exception as e:
        logger.error(f"Error transcribing chunk for user {user_id}: {str(e)}")
        # Return empty string to avoid breaking the whole process
        return f"\n[Transcription error: {str(e)}]\n"


def reencode_to_mp3(input_path: str) -> str:
    """
    Re-encodes a given input audio file (OGG/Opus, etc.) to MP3, 16kHz mono.
    Returns the path to the newly created MP3 file.
    """
    output_path = os.path.join(
        "./data",
        f"{uuid.uuid4().hex}.mp3"
    )

    # Load the input file (let pydub attempt to detect format)
    audio = AudioSegment.from_file(input_path)
    
    # Set sample rate and channels
    audio = audio.set_frame_rate(16000).set_channels(1)
    
    # Export to MP3
    audio.export(output_path, format="mp3", bitrate="128k")
    
    return output_path


def main():
    uvicorn.run(app, host="0.0.0.0", port=8702)


if __name__ == "__main__":
    main()
