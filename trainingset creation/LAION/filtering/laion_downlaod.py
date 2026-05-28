import os
from img2dataset import download
import sys
import types
import filetype

# Create a dummy imghdr module to fix weird dependency problem that should not really affect anything...
imghdr_mock = types.ModuleType("imghdr")

def download_webdataset(csv_path, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    download(
        processes_count=12,         
        thread_count=64,
        resize_mode = "no",           
        url_list=csv_path,                  
        output_folder=output_dir,   
        output_format="webdataset",  
        input_format="csv",          
        url_col="URL",                 
        enable_wandb=False,         
        number_sample_per_shard=1000, 
        distributor="multiprocessing",
        encode_format= "png",
        encode_quality= 3 #lossles png encoding!
    )

if __name__ == "__main__":
    input_csv = "/run/media/mulen/2tb nvme/FakeInversion/Master-Theisis-Working-Repo/trainingset creation/filtered_laion_900k.csv" 
    output_folder = "my_webdataset"

    download_webdataset(input_csv, output_folder)  