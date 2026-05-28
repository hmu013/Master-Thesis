#thsi script lacks the rainset augmentaiton that should have beed done before captioning.
#data was reuploaded using augment_and_vaption found in utils folder

from huggingface_hub import hf_hub_download
import zipfile
import os
import io
import tarfile
from pathlib import Path
from huggingface_hub import HfApi
import random
from tqdm import tqdm
import json
from utils.captions import TarImageDataset, generate_captions
from torch.utils.data import DataLoader
import time

repo = "poloclub/diffusiondb"
api = HfApi()
repo_id="hmu013/DiffusionDB-300k-processed"
random.seed(777)
shard_index = random.sample(range(1,2000),300) #select 300 random shards form diffuson db

def get_download_filepath(index: int):
    index = f"{index:06d}"
    return f"images/part-{index}.zip"

def get_upload_filepath(index: int):
    index = f"{index:06d}"
    return f"images/part-{index}.tar"

def sleep_one_hour ():
    total_seconds = 60*60
    timebar = tqdm(total=total_seconds, desc= "1 hour sleep progress")
    for i in range(0,total_seconds):
        time.sleep(1)
        timebar.update(1)

it = 1
pbar = tqdm(total=300, desc="Total Progres: ")

if __name__ == "__main__":

    files_in_repo = api.list_repo_files(repo_id=repo_id, repo_type="dataset")

    for i in shard_index:
        filepath = get_download_filepath(i)
        upload_path = get_upload_filepath(it) # randomly selected shardes are givennames 1-301

        # print(upload_path)
        # print(files_in_repo)
        
        if upload_path in files_in_repo:
            print("Skipped: ", upload_path)
            it += 1 
            pbar.update(1)    
            continue    

        print("downloading: ", filepath)
        local_zip = hf_hub_download(
            repo_id=repo,
            filename=filepath,
            repo_type="dataset"
        )

        img_list = []
        img_names = []
        
        with zipfile.ZipFile(local_zip, 'r') as z:

            with tarfile.open(upload_path, "w") as tar:
                for name in z.namelist():
                    if name.endswith(".json"):
                        continue

                    try:
                        with z.open(name) as f:
                            img_bytes = f.read()

                        info = tarfile.TarInfo(name=Path(name).name)
                        info.size = len(img_bytes)
                        tar.addfile(info, io.BytesIO(img_bytes))

                    except Exception as e:
                        print("Error loading", name, e)
                        continue
                
        dataset = TarImageDataset(upload_path)

        print("Dataset length:", len(dataset))
        loader = DataLoader(
            dataset,
            batch_size=96,   #Adjust settings according to vram capasity on graphicscard!
            shuffle=False,   #for my system (rtx 3090) 2 workes and 96 batch size kept the gpu at almost 100% usage
            num_workers=2,   
            pin_memory=True,
            prefetch_factor=2
        )

        captions = generate_captions(loader)

        with tarfile.open(upload_path, "a") as tar:
            for member, caption in zip(tar.getmembers(),captions):

                name, ext = os.path.splitext(member.name)

                base,_ = os.path.splitext(name)
                
                json_bytes = json.dumps(caption).encode("utf-8")

                info = tarfile.TarInfo(name=f"{name}.json")
                info.size = len(json_bytes)

                # Add it to the tar archive
                tar.addfile(info, io.BytesIO(json_bytes))


        print("Wrote:", upload_path) 

        try:
            api.upload_file(
                path_or_fileobj=upload_path,
                path_in_repo=upload_path,
                repo_id=repo_id,
                repo_type="dataset"
            )
        except:
            sleep_one_hour() #to bypas rate limit
            api.upload_file(
                path_or_fileobj=upload_path,
                path_in_repo=upload_path,
                repo_id=repo_id,
                repo_type="dataset"
            )


        print("Uploaded:", upload_path)

        os.remove(upload_path)
        os.remove(local_zip)

        it+= 1
        pbar.update(1)  