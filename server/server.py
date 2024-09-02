import logging
import openai
# import pickle
from fastapi import FastAPI, Request, Header
# , File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
# import pytube
import pytubefix as pytube
import uuid
import os
import ffmpeg
import uvicorn
import math
from pydub import AudioSegment
import subprocess
# from telebot import TeleBot
import telebot
import requests
# import shutil
import subprocess
# import json

# Set up logging 
logging.basicConfig(level=logging.INFO)

# Initialize logger
logger = logging.getLogger(__name__)

app = FastAPI(session_timeout=60*60) # 1 hour timeout

server_api_uri = 'http://localhost:8081/bot{0}/{1}'
# if server_api_uri != '':
telebot.apihelper.API_URL = server_api_uri
logger.info(f'Setting API_URL: {server_api_uri}')

# server_file_url = 'http://localhost:8081'
server_file_url = 'http://0.0.0.0:8081'
# if server_file_url != '':
telebot.apihelper.FILE_URL = server_file_url
logger.info(f'Setting FILE_URL: {server_file_url}')

token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
# Initialize the bot
bot = telebot.TeleBot(token)

class TranscriptionRequest(BaseModel):
    url: str
    chat_id: int
    message_id: int
    bot_token: str

"""def get_media_info(file_path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", file_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return json.loads(result.stdout)

def extract_audio(video_file_path):
    # Generate random filename for audio file
    audio_file_name = f"{uuid.uuid4().hex}.mp3"  
    audio_file_path = os.path.join("/tmp", audio_file_name)
    
    # FFmpeg command to extract audio
    command = [
        "ffmpeg", 
        "-i", video_file_path, 
        "-vn", "-acodec", "copy", 
        audio_file_path
    ]
    
    # Run FFmpeg process
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    return audio_file_path"""

@app.post("/message")
async def call_message(request: Request, authorization: str = Header(None)):
    logger.info('post: message')
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
        
        # server_api_uri = config['SERVER_API_URI']
        # server_file_url = config['SERVER_FILE_URL']
        # server_api_uri = 'http://localhost:8081/bot{0}/{1}'
        # # if server_api_uri != '':
        # telebot.apihelper.API_URL = server_api_uri
        # logger.info(f'Setting API_URL: {server_api_uri}')

        # server_file_url = 'http://localhost:8081'
        # # if server_file_url != '':
        # telebot.apihelper.FILE_URL = server_file_url
        # logger.info(f'Setting FILE_URL: {server_file_url}')

        # # Initialize the bot
        # bot = telebot.TeleBot(token)
        # Get the audio file ID
        """if 'video_note' in message:
            file_id = message[key]['thumb']['file_id']
        else:"""
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
            # file_bytes = bot.download_file(file_info.file_path)
            # file_info.file_path is a local folder where is file stored. Now we need to read it as bytes
            # file_path = file_info.file_path.replace(token, 'server/files')
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

        logger.info(f'file_bytes: {len(file_bytes)}')
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

        # media_info = get_media_info(file_path)
        # print(json.dumps(media_info, indent=4))
        # logger.info(f'media_info: {media_info}')
        
        # Load the audio file
        try:
            # original_audio = AudioSegment.from_file(file_path, format=media_info['streams'][0]['codec_name'])
            # original_audio = AudioSegment.from_file(file_path, format="3gp")
            """if 'video_note' in message:
                file_path = extract_audio(file_path)
                logger.info(f'audio extracted to file_path: {file_path}')"""

            original_audio = AudioSegment.from_file(file_path)
        except Exception as e:
            logger.error(f'Error loading audio file: {e}')
            bot.edit_message_text(
                f"Canceled. Unsupported format.",
                chat_id=chat_id,
                message_id=message_id
            )
            # remove audio file
            os.remove(file_path)
            return JSONResponse(content={
                "type": "empty",
                "body": ""
            })
        
        # Replace ' ' if ' ' in file_path, using rename or move
        if ' ' in file_path:
            logger.info(f'replacing space in file_path: {file_path}')
            new_file_path = file_path.replace(' ', '_')
            os.rename(file_path, new_file_path)
            file_path = new_file_path
            logger.info(f'new file_path: {file_path}')

        # Convert it to 16khz mono MP3
        logger.info(f'converting audio to {file_path}')
        # Edit message that Job has finished with text len
        bot.edit_message_text(
                f"Converting file..",
                chat_id=chat_id,
                message_id=message_id
            )
        converted_audio = original_audio.set_frame_rate(16000).set_channels(1).export(file_path, format="mp3")
        
        logger.info('Transcribing audio..')
        transcribe_audio_file(file_path, bot, chat_id, message_id)
        logger.info('Transcription finished.')
        # os.remove(file_path)
        return JSONResponse(content={
            "type": "empty",
            "body": ""
            })
        
        
    
    if 'text' in message:
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

        # Youtube transcription CMD
        elif message['text'].startswith("https://www.youtube.com/") or \
            message['text'].startswith("https://youtube.com/") or \
            message['text'].startswith("https://www.youtu.be/") or \
            message['text'].startswith("https://youtu.be/"):
            transcription_request = TranscriptionRequest(
                url=message['text'],
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
        # # Initialize the bot
        # bot = TeleBot(bot_token)

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
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
    if OPENAI_API_KEY == '':
        raise Exception("OPENAI_API_KEY environment variable not found")
    if 'Error' in audio_path:
        bot.edit_message_text(
            audio_path,
            chat_id=chat_id,
            message_id=message_id
        )
        return {"transcription": audio_path}

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
        logger.info(f"[{chat_id}] Processing chunk {idx+1} of {len(chunk_paths)}")

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
