##my first script for uploading laion, icluded no augments for trainset and is therfore now repladec by the caption and augment script pending rename

from huggingface_hub import HfApi
from tqdm import tqdm
import time
import json
import tarfile
from utils.captions import TarImageDataset, generate_captions
import webdataset as wds
from torch.utils.data import DataLoader
import tarfile
import os
import io
from tqdm import tqdm

def is_png(filename):
    return os.path.splitext(filename.lower())[1] == ".png"

def sleep_one_hour ():
    total_seconds = 60*60
    timebar = tqdm(total=total_seconds, desc= "1 hour sleep progress")
    for i in range(0,total_seconds):
        time.sleep(1)
        timebar.update(1)

def create_tar_filepath(index: int):
    index = f"{index:05d}"
    return f"{index}.tar"


api = HfApi()
repo_id="hmu013/LAION-300k"

folder_path = "/run/media/mulen/2tb nvme/FakeInversion/my_webdataset"

image_count = 0

for i in tqdm(range(0,500)):
    if image_count>=300000:
        pass
    tar_name = create_tar_filepath(i)
    tar_path = os.path.join(folder_path,tar_name)
    output_tar_path = tar_path+"temp"

    files_in_repo = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    if tar_name in files_in_repo:
            print("Skipped: ", tar_name)
            image_count+=1000
            continue    

    dataset= TarImageDataset(tar_path)

    print("Dataset length:", len(dataset))
    loader = DataLoader(
            dataset,
            batch_size=96,   #Adjust settings according to vram capasity on graphicscard!
            shuffle=False,   #for my system (rtx 3090 24gb vram) 2 workes and 96 batch size kept the gpu at almost 100% usage
            num_workers=2,   
            pin_memory=True,
            prefetch_factor=2
        )

    captions = generate_captions(loader)

    
    with tarfile.open(tar_path, "r") as in_tar, \
    tarfile.open(output_tar_path, "w") as out_tar:

        caption_idx = 0 

        for member in in_tar.getmembers():
            if member.isfile() and is_png(member.name) and image_count < 300000:
                fileobj = in_tar.extractfile(member)
                
                out_tar.addfile(member, fileobj)

                name, ext = os.path.splitext(member.name)

                base,_ = os.path.splitext(name)
                
                caption = captions[caption_idx]
                caption_idx += 1

                json_bytes = json.dumps(caption).encode("utf-8")

                info = tarfile.TarInfo(name=f"{name}.json")
                info.size = len(json_bytes)

                # Add it to the tar archive
                out_tar.addfile(info, io.BytesIO(json_bytes))

                image_count+=1

    print("Wrote:", output_tar_path) 

    try:
        api.upload_file(
            path_or_fileobj=output_tar_path,
            path_in_repo=tar_name,#kept path same as origina tar
            repo_id=repo_id,
            repo_type="dataset"
        )
    except Exception as e:
        print(e)
        sleep_one_hour() #to bypas rate limit
        api.upload_file(
            path_or_fileobj=output_tar_path,
            path_in_repo=tar_name,#kept path same as origina tar
            repo_id=repo_id,
            repo_type="dataset"
        )

    os.remove(output_tar_path)
    print("Deleted temporary tar")

    print("Uploaded:", tar_path)

print("Finished!")

