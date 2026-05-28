import webdataset as wds
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import csv
from tqdm import tqdm
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from torchmetrics.aggregation import MeanMetric

URLS = "https://huggingface.co/datasets/hmu013/SynRIS-extracted-features/resolve/main/{00000..00088}.tar"
CHECKPOINT_PATH = "Master-Theisis-Working-Repo/weights/resnet_forensic_epoch_19.ptrom"
OUTPUT_CSV = "Master-Theisis-Working-Repo/results/predictions.csv"

#---------------------------Preparing resnet model-----------------------------------
model = models.resnet50(weights=None, norm_layer=nn.InstanceNorm2d)

weights = models.ResNet50_Weights.IMAGENET1K_V1
state_dict = weights.get_state_dict(progress=True)
filtered_dict = {k: v for k, v in state_dict.items() if "running" not in k and "num_batches_tracked" not in k}
model.load_state_dict(filtered_dict, strict=False)

with torch.no_grad():
    old_weight = model.conv1.weight.data
    new_weight = torch.cat([old_weight * 0.25] * 4, dim=1)
    model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight = nn.Parameter(new_weight)

model.fc = nn.Linear(2048, 1)
torch.nn.init.normal_(model.fc.weight.data, 0.0, 0.02)

model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

#---------------- Loading WebDataset -----------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #normalization for resnet model
])

def concat_pluss_residuals(org, inv, rec):
    residual = torch.abs(org - rec)
    return torch.cat([org, inv, rec, residual], dim=0)

def infer_label_from_key(key: str):
    key = key.lower()
    if "real" in key: return 0
    elif "fake" in key: return 1
    else: raise ValueError(f"Cannot infer label from key: {key}")

dataset = (
    wds.WebDataset(
        URLS,
        handler=wds.warn_and_continue,
        resampled=False,
        shardshuffle=False
    )
     .decode(wds.imagehandler("pil"), wds.handle_extension("json", lambda x: x)) #corrupted json due to not being uploaded as dict, if we read bits we get text
    .to_tuple("__key__", "org.png", "inv.png", "rec.png")
    .map(lambda x: (
        x[0],  # key
        concat_pluss_residuals(
            transform(x[1]),
            transform(x[2]),
            transform(x[3])
        ),
        infer_label_from_key(x[0])
    ))
)

loader = wds.WebLoader(
    dataset,
    batch_size=32,
    num_workers=12,
    pin_memory=True
)


#-------------------------------eval loop-----------------------------------------------------------
acc_metric = BinaryAccuracy().to(device)
auc_metric = BinaryAUROC().to(device)
loss_metric = MeanMetric().to(device)

@torch.no_grad()
def evaluate_and_save(model, loader, out_csv):
    acc_metric.reset()
    auc_metric.reset()
    loss_metric.reset()

    rows = []

    for keys, x, y in tqdm(loader, desc="Evaluating"):
        x = x.to(device)
        y = y.float().to(device)

        logits = model(x).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        probs = torch.sigmoid(logits)

        acc_metric.update(probs, y)
        auc_metric.update(probs, y)
        loss_metric.update(loss)

        for k, gt, logit, prob in zip(
            keys,
            y.cpu(),
            logits.cpu(),
            probs.cpu()
        ):
            rows.append({
                "key": k,
                "label": int(gt.item()),
                "logit": float(logit.item()),
                "prob": float(prob.item())
            })

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["key", "label", "logit", "prob"]
        )
        writer.writeheader()
        writer.writerows(rows)

    results = {
        "loss": loss_metric.compute().item(),
        "accuracy": acc_metric.compute().item(),
        "auroc": auc_metric.compute().item(),
        "num_samples": len(rows)
    }

    print("\n====== Evaluation Results ======")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k:>12}: {v:.4f}")
        else:
            print(f"{k:>12}: {v}")
    print("================================")

    print(f"\nSaved predictions to: {out_csv}")

    return results

# ---------------- Run ----------------
evaluate_and_save(model, loader, OUTPUT_CSV)