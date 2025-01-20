import yt_dlp
import sys
import os

def download_audio(url, output_dir=None):
    # Create output directory if specified and doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        # Add output template for audio files
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s') if output_dir else '%(title)s.%(ext)s',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def download_video(url, output_dir=None):
    """
    Download video in best quality with audio from YouTube URL.
    
    Args:
        url (str): YouTube video URL
        output_dir (str, optional): Directory to save the downloaded files
    """
    
    # Create output directory if specified and doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    ydl_opts = {
        # Download best quality video that includes audio
        'format': 'bv*+ba/b',  # This will download best video + best audio
        
        # If you want to specify a different format, you can use:
        # 'format': 'bestvideo+bestaudio/best',  # Specifically best video + best audio
        
        # Output template
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s') if output_dir else '%(title)s.%(ext)s',
        
        # Show progress bar
        'progress_hooks': [lambda d: print(f'Downloading: {d["_percent_str"]} of {d["_total_bytes_str"]}') 
                          if d['status'] == 'downloading' else None],
        
        # Merge video and audio automatically
        'merge_output_format': 'mp4',  # You can change to mkv if preferred
        
        # Additional options
        'writethumbnail': True,  # Save thumbnail
        'writedescription': True,  # Save description
        'writeinfojson': True,    # Save video metadata
        
        # Error handling
        'ignoreerrors': True,     # Skip unavailable videos
        'no_warnings': False,     # Show warnings
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print("Download completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <youtube_url> [output_directory]")
        sys.exit(1)
        
    url = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"

    download_audio(url, output_dir)
    
    download_video(url, output_dir)