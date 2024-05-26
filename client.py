import requests
from base64 import b64decode
import dotenv
import os
import json
from scipy.io.wavfile import write


dotenv.load_dotenv()
url = os.getenv("url")


payload = json.dumps({
  "text": "Every morning, Sarah walked her dog, Max, through the old, enchanting forest  her home. One day, thanyou have a noce day",
  "prompt":"A female speaker with a slightly low-pitched voice delivers her words quite expressively, in a very confined sounding environment with clear audio quality. She speaks very fast."
})
headers = {
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)
if response.ok:
    response:dict = json.loads(response.text)
    audio = b64decode(response["audio"])
    print(response["time"])
    write("text.wav",response["sr"],audio)