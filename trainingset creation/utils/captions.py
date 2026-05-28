from torch.utils.data import DataLoader
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from tqdm import tqdm
import io
import tarfile
from PIL import Image
from torch.utils.data import Dataset
from .trainset_augmentations import strong_transform, image_transform
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using: ", device)

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast = True)
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    dtype=torch.float16
    ).to(device)

model.eval()

def generate_captions(data_loader : DataLoader):
    
    captions = []

    with torch.inference_mode():
        for batched_images in tqdm(data_loader, desc= "Generating captions"):
            
            inputs = processor(images=batched_images, return_tensors="pt", do_rescale = False )
            inputs = {k: v.to(device, non_blocking=True, dtype = torch.float16) for k, v in inputs.items()}

            generated_ids = model.generate(**inputs, max_new_tokens=20)

            batch_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            for caption in batch_text:
                captions.append({ "caption": caption.strip()})

    print("Done captioning")

    return captions

class TarImageDataset(Dataset):
    def __init__(self, tar_path):
        self.tar_path = tar_path
        self.image_transform = image_transform
        self.strong_transform = strong_transform
        self.members = []

        with tarfile.open(tar_path) as tar:
            for m in tar.getmembers():
                if m.isfile() and m.name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.members.append(m)

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx):
        member = self.members[idx]

        with tarfile.open(self.tar_path) as tar:
            file = tar.extractfile(member)
            img = Image.open(io.BytesIO(file.read())).convert("RGB")

            img = np.array(img)
            img = self.image_transform(image=img)["image"]
            img = self.strong_transform(image=img)["image"]

            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img