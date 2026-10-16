#!/usr/bin/env python3
import argparse, json
from music_genre.predict import predict_audio
parser=argparse.ArgumentParser(); parser.add_argument('--audio', required=True); parser.add_argument('--checkpoint', required=True); parser.add_argument('--device', default='auto')
args=parser.parse_args(); print(json.dumps(predict_audio(args.audio, args.checkpoint, args.device), indent=2))
