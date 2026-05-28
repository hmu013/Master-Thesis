import os
import csv
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

class FakeInversionDataset(Dataset):
    def __init__(self, folder):

        self.folder = folder

        #create list of paths to all roiginal images in the processed folder
        self.original_dir = os.path.join(folder,"original")
        self.original_paths = []
        for root, _, files in os.walk(self.original_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rel_path = os.path.relpath(os.path.join(root, f), self.original_dir)
                    self.original_paths.append(rel_path)
        
        self.inverted_dir = os.path.join(folder,"inverted")
        self.inverted_paths = []
        for root, _, files in os.walk(self.inverted_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rel_path = os.path.relpath(os.path.join(root, f), self.inverted_dir)
                    self.inverted_paths.append(rel_path)

        self.reconstructed_dir = os.path.join(folder,"reconstructed")
        self.reconstructed_paths = []
        for root, _, files in os.walk(self.reconstructed_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rel_path = os.path.relpath(os.path.join(root, f), self.reconstructed_dir)
                    self.reconstructed_paths.append(rel_path)

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)), #resize to 512 like in fake inversion paper
            transforms.ToTensor() #data loader does not support PIL images so we want tp return tensors 
        ])

        #check for captions
        self.captions_path = os.path.join(folder,'captions.csv')
        self.captions = {}
        self.has_captions = None

        try: #load captions, should not fail if dataset was produced by process_dataset.py
            with open(self.captions_path,'r') as data:
                print("Dataset has captions!: \n")
                for line in tqdm(csv.reader(data),desc = "Loading captions"):
                        self.captions[line[0]] = line[1]

            self.has_captions= True
        except:
             self.has_captions=False

    def __len__(self):
        return len(self.original_image_paths)

    def __getitem__(self, idx):
        
        original_path = self.image_rel_paths[idx]
        original_img = Image.open(os.path.join(self.folder, "original", path)).convert("RGB")
        original_img = self.transform(original_img)

        inverted_path = self.image_rel_paths[idx]
        inverted_img = Image.open(os.path.join(self.folder, "inverted", path)).convert("RGB")
        inverted_img = self.transform(original_img)

        reconstructed_path = self.image_rel_paths[idx]
        reconstructed_img = Image.open(os.path.join(self.folder, "reconstructed" ,path)).convert("RGB")
        reconstructed_img = self.transform(original_img)

        caption = self.captions[path]
        name = original_path

        return original_img, inverted_img, reconstructed_img, caption, name
        

#main function to test loading dataset
#usefull for debugging
if __name__ == "__main__":

    path = r"Master-Theisis-Working-Repo\fakeinversion implementation\processed_datasets\test_processed"   
    dataset = FakeInversionDataset(path)
    print(dataset.original_paths)