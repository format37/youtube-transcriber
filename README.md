# YouTube Transcriber

This Docker server performs the following tasks:
1. Downloads a YouTube video using the link provided by the client.
2. Extracts audio from the downloaded video.
3. Uses the Whisper API to transcribe the audio to text.
4. Returns the transcribed text to the client.

## Installation

To set up the YouTube Transcriber, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/format37/youtube-transcriber.git
    ```
2. Navigate to the project directory:
    ```bash
    cd youtube-transcriber
    ```
3. Update the `Dockerfile` with your Whisper API key.
  
4. Build and run the Docker container:
    ```bash
    docker build -t youtube-transcriber .
    docker run -p 8702:8702 youtube-transcriber
    ```

## Usage

1. Update `client.py` with the YouTube URL and desired language for transcription.

2. Run `client.py`:
    ```bash
    python3 client.py
    ```

3. Wait for the transcribed text to be saved in `transcription.txt`.
