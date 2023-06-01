import os

IN_IMAGE_PATH = os.path.join('data', 'incoming', 'images')
IN_AUDIO_PATH = os.path.join('data', 'incoming', 'audio')
OUT_IMAGE_PATH = os.path.join('data', 'outgoing', 'images')
OUT_AUDIO_PATH = os.path.join('data', 'outgoing', 'audio')

if not os.path.exists(IN_IMAGE_PATH):
    os.makedirs(IN_IMAGE_PATH)

if not os.path.exists(IN_AUDIO_PATH):
    os.makedirs(IN_AUDIO_PATH)

if not os.path.exists(OUT_IMAGE_PATH):
    os.makedirs(OUT_IMAGE_PATH)

if not os.path.exists(OUT_AUDIO_PATH):
    os.makedirs(OUT_AUDIO_PATH)
