# Real time transcirption tool

## Project goal

The goal of this project is transcribing the speech from the live streams in any platforms, for example, Youtube, Twitch and NicoNico etc.

## Mechanism

1. The audio recorder in the proejct is recording and sending the signal to server every second.
2. the server retrieve the signal and perform transcription
3. meanwhile, the signal also store in a buffer
4. the transcription result is sent back to the UI for display
5. if the voice activity detection detects the end of conversation or the buffer reachs the limits of data that the PC can handle, then all the singal in the buffer, inculding the last chunk of audio, will be used to perform transcription once more to retrieve a more accurate result.
6. the full sentence transcription result is sent to UI and replace the buffered transcription used for display previously


## Requirement
Windows OS as the loopback audio library is Windows base

Python3

Cuda developmenet tool kit

## How to use
1. Install library
```
pip install requirements.txt
```

2. Start up the UI
```
python app_ui.py
```

3. Start up the server
```
python server.py
```

4. Press start after server started and library is downloaded for the first time

## Demo video

Demo for Japanese transcription

https://youtu.be/8CgkU5jdeOg

Demo for English transcription

https://youtu.be/obY3jOO6UlI

### Demo environment

Spec     | Model
---------|-------------
PC model | OMEN by HP Laptop
Memory   | 16GB
CPU      | Intel core i7 (Intel64 Family 6 Model 141 Stepping 1 GenuineIntel ~2304 Mhz)
GPU      | NVIDIA GeForce RTX 3070 Laptop GPU


## Reference projects

KoljaB/RealtimeSTT

ufal/whisper_streaming
