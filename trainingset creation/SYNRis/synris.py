from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi
import webdataset as wds
import numpy as np
import torch
import os
import albumentations as A
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

IMAGE_FOLDER = "/run/media/mulen/2tb nvme/FakeInversion/SYNris"

#----------------Datset implementation ---------------------

class ImageWithPaths(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.transform = A.Compose([
            A.SmallestMaxSize(max_size=512), 
            A.CenterCrop(height=512, width=512),
        ])

        self.image_files = []
        for root, _, files in os.walk(root_dir):
            for f in files:
                # Good: Case-insensitive check
                if f.lower().endswith('.png'):
                    self.image_files.append(os.path.join(root, f))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        full_path = self.image_files[idx]
        
        img = Image.open(full_path).convert("RGB")
        img_np = np.array(img)
        
        transformed = self.transform(image=img_np)
        img_final = Image.fromarray(transformed['image'])
        rel_path = os.path.relpath(full_path, self.root_dir)
        clean_path = os.path.splitext(rel_path)[0] #drop the etention #all pngs
        
        return img_final, clean_path

#--------------------uploadstuff------------------------------------
api = HfApi()
repo_id = "hmu013/SynRIS_captioned" #to upload to

UPLOAD_WORKERS = 2       
MAX_SHARDS = 300
shards_written = 0

executor = ThreadPoolExecutor(max_workers=UPLOAD_WORKERS)

def upload_and_cleanup(shard_path):
    filename = os.path.basename(shard_path)
    try:
        api.upload_file(
            path_or_fileobj=shard_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"uploaded: {filename}")
        os.remove(shard_path) 
    except Exception as e:
        print(f"Failed {filename}. Keeping on disk. Error: {e}")

def on_shard_close(shard_name):
    global shards_written
    shards_written += 1
    print(f"Shard {shards_written}/{MAX_SHARDS} closed: {shard_name}")
    executor.submit(upload_and_cleanup, shard_name)

sink = wds.ShardWriter(
    "%05d.tar",
    maxcount=1000,
    post=on_shard_close,
)

#------------------------Loading images--------------------------------------------

dataset = ImageWithPaths(IMAGE_FOLDER)

def list_collate(batch): #to keep images, captions and keys as lists - default cused some problems at one point...
    return list(zip(*batch))

loader= DataLoader(
    dataset= dataset,
    batch_size=200,
    shuffle=None,
    num_workers= 8,
    prefetch_factor=2,
    pin_memory=True,
    collate_fn=list_collate
    )


#---------------generate captions--------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: ", device)

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast = True)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    dtype=torch.float16
    ).to(device)

with torch.inference_mode():
    for images , keys in tqdm(loader, desc= "Generating captions"):
        
        if shards_written >= MAX_SHARDS:
            break

        captions = []
        
        inputs = processor(images=images, return_tensors="pt", do_rescale = True )
        inputs = {k: v.to(device, non_blocking=True, dtype = torch.float16) for k, v in inputs.items()}

        generated_ids = model.generate(**inputs, max_new_tokens=20)

        batch_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )

        for caption in batch_text:
            captions.append({ "caption": caption.strip()})
        
        for i in range(len(images)):

            sample = {
                "__key__": keys[i],
                "json": captions[i],
                "png": images[i], 
            }
            sink.write(sample)


print("Done captioning")

sink.close()   

executor.shutdown(wait=True)
print("finished final uploads")