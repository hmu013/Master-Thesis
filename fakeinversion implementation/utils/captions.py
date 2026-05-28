import torch
import pandas as pd
import os
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from .UnprocessedDataset import UnprocessedDataset

def generate_captions( loader, output_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using: ", device)

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast = True)
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        dtype=torch.float16
    ).to(device)

    captions = []

    model.eval()
    with torch.inference_mode():
        for batch_images, batch_names in tqdm(loader, desc= "Generating captions"):
            
            inputs = processor(images=batch_images, return_tensors="pt", do_rescale = False )
            inputs = {k: v.to(device, non_blocking=True, dtype = torch.float16) for k, v in inputs.items()} #sends batch of images to device

            generated_ids = model.generate(**inputs, max_new_tokens=20)

            batch_text = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )

            for name, cap in zip(batch_names, batch_text):
                captions.append({"image": name.replace('\\', '/'), "caption": cap.strip()})

    os.makedirs(output_path, exist_ok=True)
    pd.DataFrame(captions).to_csv(os.path.join(output_path,"captions.csv"), index=False)

    print("Done captioning")