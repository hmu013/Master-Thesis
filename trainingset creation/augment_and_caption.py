
#Because i had generated captions and uploaded image along with captions before trainingset was augmentetd (flips, rotations, crops, grayscale etc)
#captions had to be re computed so that it for included grayscale in caption if image had been augmentet to grayscale 
#For diffusion db i stremed from my already uploaded dataset on hf, and for LAION i used the local data on my computer 

from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import HfApi
import webdataset as wds
import numpy as np
import torch
import os
from queue import Queue
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from tqdm import tqdm
from PIL import Image

from utils.trainset_augmentations import strong_transform, image_transform

#-------------------- Trainset transforms -----------------------
def trainset_transform(img):
    
    if img is None: #diffusion db has some none values we need to handle
        return None
  
    img = np.array(img)

    img = image_transform(image=img)['image']
    img = strong_transform(image =img)['image']

    img = Image.fromarray(img) 
    
    return img

#--------------------uploadstuff------------------------------------
api = HfApi()
repo_id = "hmu013/LAION-300k-processed" #to upload to

upload_queue = Queue(maxsize=8)
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

def filter_none(sample): #add filter to handle a few non values in diffusiondb
    return sample[0] is not None

#------------------------Loading images--------------------------------------------
dataset = (
    wds.WebDataset(
        "my_webdataset/{00000..00899}.tar", #tar files to stream locla or url
        handler=wds.warn_and_continue, 
        resampled=False, 
        shardshuffle=False,
    )
    .decode("pil")
    .to_tuple("png", "__key__", missing_is_error=False) # missing_is_error returns none insted of breaking when missing values are found
    .map_tuple(
        trainset_transform,   #lambda j: j["caption"],
        lambda k: k,           
    )
    .select(filter_none)
)

def list_collate(batch): #to keep images, captions and keys as lists - default cused some problems at one point...
    return list(zip(*batch))

loader = wds.WebLoader(
    dataset,
    batch_size=200,
    shuffle = False,
    num_workers=4, 
    pin_memory=True,
    prefetch_factor = 2,
    persistent_workers = True,
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
    for images , keys in tqdm(loader, desc= "Generating captions"): # captions_old ,
        
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

        for i in range(len(keys)):
            sample = {
                "__key__": keys[i],
                "json": captions[i],
                "png": images[i], 
            }
            sink.write(sample)


print("Done captioning")

sink.close()           
upload_queue.join()#wait for uploads to finish

executor.shutdown(wait=True)
print("finished final uploads")