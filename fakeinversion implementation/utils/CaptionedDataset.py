import os
import csv
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import albumentations as A
import numpy as np

class CaptionedDataset(Dataset):
    def __init__(self, folder, output_path):
        self.folder = folder
        self.all_image_rel_paths = [
                os.path.relpath(os.path.join(root, f), folder).replace('\\', '/')
                for root, _, files in os.walk(folder)
                for f in files
                if f.lower().endswith(('.jpg','.jpeg','.png'))
            ]
        self.image_rel_paths = []

        #check for captions.csv in outputfolder
        self.captions_path = os.path.join(output_path,'captions.csv')
        self.captions = {}
        self.transform = A.Compose([
            A.SmallestMaxSize(512), 
            A.CenterCrop(512, 512),
        ])
        try: #try to load captions if already exists
            with open(self.captions_path,'r') as data:
                print("Dataset has captions!: \n")
                for line in tqdm(csv.reader(data),desc = "Loading captions"):
                        self.captions[line[0].replace('\\', '/')] = line[1]
            print("Sucessfully loaded captions")
        except:
                print("Could not load captions")

        #chck for already processed images, when no image is found in output we add it to list of paths to be proccessed
        skipped_count = 0
        print(f"Checking for processed images in output path")
        for rel_path in self.all_image_rel_paths:
            base_name = os.path.splitext(rel_path)[0]  # remove file extention
            processed_path = os.path.join(output_path, "original", base_name + ".png") #all processed images are stored lossles as png
            if os.path.exists(processed_path):
                skipped_count += 1
            else:
                self.image_rel_paths.append(rel_path)
        print(f"Skipped {skipped_count} images already processed ")
    
    def __len__(self):
        return len(self.image_rel_paths)

    def __getitem__(self, idx):
        path = self.image_rel_paths[idx]
        try:
            img = Image.open(os.path.join(self.folder, path)).convert("RGB")
            img= np.array(img)
            augmented = self.transform(image=img)
            img_tensor = transforms.ToTensor()(augmented["image"])
        except Exception as e:
            print("Failed to load image: ",path, e)
            return None
        
        image_caption = self.captions[path]
        return img_tensor, path, image_caption
    
#main function to test loading dataset
#usefull for debugging
if __name__ == "__main__":

    path = r"Master-Theisis-Working-Repo/fakeinversion implementation/datasets/PreSocial" #put path to your data here for testing
    out_path = r"Master-Theisis-Working-Repo/fakeinversion implementation/processed_datasets/PreSocial_processed"
    dataset = CaptionedDataset(path, output_path=out_path)
    img, name, caption = dataset.__getitem__(0)
    inv_image_path = os.path.join(dataset.folder,'inverted',name)
    print(inv_image_path)
    print (name)
    print(caption)