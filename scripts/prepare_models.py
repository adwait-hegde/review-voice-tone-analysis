"""Explicit opt-in weight download; resolve immutable revisions before loading."""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
from words_voice.data import read_json, write_json

p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/default.json')
p.add_argument('--out',default='artifacts/pretrained-config.json')
p.add_argument('--allow-download',action='store_true')
a=p.parse_args()
if not a.allow_download:
    p.error('Estimated download ~0.6 GB; pass --allow-download explicitly to proceed')
cfg=read_json(a.config)
for kind in ('text','audio'):
    repo=cfg[kind+'_encoder']
    revision=cfg.get(kind+'_revision') or HfApi().model_info(repo).sha
    cfg[kind+'_revision']=revision
    path=snapshot_download(repo,revision=revision,allow_patterns=['*.json','*.txt','*.safetensors'],ignore_patterns=['onnx/*','openvino/*','onnx/**','openvino/**'])
    print(kind,revision,path,flush=True)
repo='Systran/faster-whisper-tiny.en'
revision=cfg.get('asr_revision') or HfApi().model_info(repo).sha
path=snapshot_download(repo,revision=revision,allow_patterns=['model.bin','config.json','tokenizer.json','vocabulary.*'])
cfg['asr_model']=path
cfg['asr_revision']=revision
write_json(a.out,cfg)
print('Saved immutable encoder revisions and local ASR snapshot:',a.out)
