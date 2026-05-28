import os
import csv
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageOps
from tqdm import tqdm
import albumentations as A
import numpy as np

class UnprocessedDataset(Dataset):
    def __init__(self, folder, output_path):
        self.folder = folder
        self.image_rel_paths = [
                os.path.relpath(os.path.join(root, f), folder).replace('\\', '/')
                for root, _, files in os.walk(folder)
                for f in files
                if f.lower().endswith(('.jpg','.jpeg','.png'))
            ]

         #check for captions
        self.captions_path = os.path.join(output_path,'captions.csv')
        self.captions = {}
        self.has_captions = None
        try: #try to load captions if already exists
            with open(self.captions_path,'r') as data:
                print("Dataset has captions!: \n")
                for line in tqdm(csv.reader(data),desc = "Loading captions"):
                        self.captions[line[0].replace('\\', '/')] = line[1]

            self.has_captions= True
        except:
            self.has_captions=False
        
        self.transform = image_transform = A.Compose([
            A.SmallestMaxSize(512), 
            A.CenterCrop(512, 512),
        ])

    def __len__(self):
        return len(self.image_rel_paths)
    
    def __getitem__(self, idx):
        path = self.image_rel_paths[idx]
        try:
            img = Image.open(os.path.join(self.folder, path)).convert("RGB")
            img= np.array(img)
            augmented = self.transform(image=img)
            img_tensor = transforms.ToTensor()(augmented["image"])
            return img_tensor, path
        except Exception as e:
            print(f"Failed to load image {path}: ", e) # Print the actual error
            return None                                                                         

    def verify_all_images(self, delete_bad=False):
        good = []
        bad = []

        iterator = tqdm(self.image_rel_paths, desc="Verifying images")
        for path in iterator:
            full_path = os.path.join(self.folder, path)

            try:
                with Image.open(full_path) as img:
                    img.verify()
                good.append(path)
            except Exception:
                bad.append(path)
                if delete_bad:
                    os.remove(full_path)

        if delete_bad:
            print("Files deleted:",bad)

        return {
            "total": len(good) + len(bad),
            "valid": len(good),
            "corrupt": len(bad),
            "removed": len(bad) if delete_bad else 0
        }