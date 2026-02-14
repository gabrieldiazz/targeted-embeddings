import os
from datasets import load_dataset, Audio
from tqdm import tqdm

#set cache to project space
os.environ["HF_HOME"] = "/projects/bgbh/gdiazjr/hf_cache"
OUTPUT_DIR = "/projects/bgbh/datasets/l2-artic/data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ds = load_dataset("KoelLabs/L2Arctic", split="scripted")
ds = ds.cast_column("audio", Audio(decode=False))

for i, row in enumerate(tqdm(ds)):
    accent = str(row['speaker_native_language']).lower()
    accent_path = os.path.join(OUTPUT_DIR, accent)
    os.makedirs(accent_path, exist_ok=True)
    
    #filename: speakerID_index.wav
    filename = f"{row['speaker_code']}_idx{i:05d}.wav"
    filepath = os.path.join(accent_path, filename)
    
    if not os.path.exists(filepath):
        # We just write the raw bytes straight to a .wav file
        with open(filepath, "wb") as f:
            f.write(row['audio']['bytes'])
