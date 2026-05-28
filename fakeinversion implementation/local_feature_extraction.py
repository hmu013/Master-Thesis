import os
import csv
from tqdm import tqdm
from utils.UnprocessedDataset import UnprocessedDataset
from utils.CaptionedDataset import CaptionedDataset
from utils.captions import generate_captions
from utils.inversion import invert_and_reconstruct
from torch.utils.data import DataLoader
    
if __name__ == "__main__":
    data_path =r"/run/media/mulen/2tb nvme/FakeInversion/SYNris"
    output_path = r"/run/media/mulen/2tb nvme/FakeInversion/SYNris_extracted_features"

    dataset = UnprocessedDataset(data_path, output_path = output_path)

    #can b used to check for corupted images, and delete them if set to true
    # stats = dataset.verify_all_images(delete_bad=True)
    # print(stats)

    if dataset.has_captions != True:
        #loader for captioning: can have much higer batch size than feature extraction!
        print("Dataset length:", len(dataset))
        loader = DataLoader(
            dataset,
            batch_size=96,  
            shuffle=False,   
            num_workers=2,   
            pin_memory=True,
            prefetch_factor=2
        )
        
        generate_captions(loader, output_path = output_path)
    
    dataset = CaptionedDataset(data_path, output_path)

    loader = DataLoader(
            dataset,
            batch_size=16,  
            shuffle=False,   
            num_workers=2,   
            pin_memory=True,
            prefetch_factor=2,
        )

    invert_and_reconstruct(loader, output_path = output_path)  

