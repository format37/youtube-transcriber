import requests
import json

URL = "http://localhost:8702/transcribe"

# Video URL to transcribe
video_url = "https://www.youtube.com/watch?v=VBba1Lsi910"

data = {
  "url": video_url
}

headers = {
  "Content-Type": "application/json"
}

response = requests.post(URL, json=data, headers=headers)

# print(response.json()["transcription"])
# Save to txt file
with open("transcription.txt", "w") as f:
    f.write(response.json()["transcription"])

print("Transcription saved to transcription.txt")