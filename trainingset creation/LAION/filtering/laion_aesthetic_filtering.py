import pandas as pd
import os
from tqdm import tqdm
from datasets import load_dataset


target_count = 900_000
aesthetic_threshold = 6.25
save_chunks = 5000  
output_file = "filtered_laion_900k.csv"

hf_token = os.getenv("HF_TOKEN")

ds = load_dataset(
    "laion/aesthetics_v2_4.75",
    split="train",
    streaming=True,
    storage_options={"token": hf_token},
)

ds_filtered = ds.filter(
    lambda x: [
        score is not None and h is not None and w is not None and 
        score > aesthetic_threshold and h >= 512 and w >= 512
        for score, h, w in zip(x["AESTHETIC_SCORE"], x["HEIGHT"], x["WIDTH"])
    ],
    batched=True, 
    batch_size=1000 
) 
pbar= tqdm(total=target_count, desc="Saving filtered data")
buffer = []
total_saved = 0

for datapoint in ds_filtered:
    
    buffer.append({
        "URL": datapoint.get("URL"),
        "TEXT": datapoint.get("TEXT"),
        "AESTHETIC_SCORE": datapoint.get("AESTHETIC_SCORE"),
        "WIDTH": datapoint.get("WIDTH"),
        "HEIGHT": datapoint.get("HEIGHT"),
        "similarity": datapoint.get("similarity"),
    })
    
    total_saved += 1
    pbar.update(1)

    if len(buffer) >= save_chunks:
        df = pd.DataFrame(buffer)
        df.to_csv(output_file, mode="a", header=not os.path.exists(output_file), index=False)
        buffer.clear()

    if total_saved >= target_count:
        break

pbar.close()

if buffer:
    df = pd.DataFrame(buffer)
    df.to_csv(output_file, mode="a", header=not os.path.exists(output_file), index=False)

print("Finished! Total saved:", total_saved)