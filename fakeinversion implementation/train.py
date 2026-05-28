import webdataset as wds
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy
import wandb
from torchmetrics.aggregation import MeanMetric

REAL_TRAIN_URLS = "https://huggingface.co/datasets/hmu013/LAION-300k-extracted-features/resolve/main/{00000..00275}.tar"
FAKE_TRAIN_URLS = "https://huggingface.co/datasets/hmu013/DiffusionDB-300k-extracted-features/resolve/main/{00000..00275}.tar"

REAL_VAL_URLS ="https://huggingface.co/datasets/hmu013/LAION-300k-extracted-features/resolve/main/{00276..00295}.tar"
FAKE_VAL_URLS ="https://huggingface.co/datasets/hmu013/DiffusionDB-300k-extracted-features/resolve/main/{00276..00295}.tar"

USE_RESIDUALS = False
print(f"Stariting training with USE_RESIDUSLS = {USE_RESIDUALS}")

NUM_IMAGES = 4 if USE_RESIDUALS else 3
IN_CHANNELS = 3 * NUM_IMAGES

#---------------------------Preparing resnet model-----------------------------------
model = models.resnet50(weights=None, norm_layer=nn.InstanceNorm2d)

weights = models.ResNet50_Weights.IMAGENET1K_V1
state_dict = weights.get_state_dict(progress=True)
filtered_dict = {k: v for k, v in state_dict.items() if "running" not in k and "num_batches_tracked" not in k}
model.load_state_dict(filtered_dict, strict=False)

with torch.no_grad(): #replace first layer of convolustion to take multiple input images
    old_weight = model.conv1.weight.data
    repeats = IN_CHANNELS // 3
    new_weight = torch.cat([old_weight / repeats] * repeats, dim=1)
    model.conv1 = nn.Conv2d(
        IN_CHANNELS,
        64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False
    )
    model.conv1.weight = nn.Parameter(new_weight)

#final layer for binary classification
model.fc = nn.Linear(2048, 1)
torch.nn.init.normal_(model.fc.weight.data, 0.0, 0.02)

#----------------Loading WebDataset x2------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def build_input(org, inv, rec):
    if USE_RESIDUALS:
        residual = torch.abs(org - rec)
        return torch.cat([org, inv, rec, residual], dim=0)
    else:
        return torch.cat([org, inv, rec], dim=0)
    
def make_dataset(urls, resampled = True, shardshuffle=True, label= 0):
    return (
        wds.WebDataset(urls=urls, handler=wds.warn_and_continue, resampled=resampled, shardshuffle=shardshuffle)
        .shuffle(200) 
        .decode(wds.imagehandler("pil"), wds.handle_extension("json", lambda x: x)) #corrupted json due to not being uploaded as dict, if we reat bits we get text
        .to_tuple("org.png", "inv.png", "rec.png")
        .map(lambda x: (transform(x[0]), transform(x[1]), transform(x[2])))
        .map(lambda x: (build_input(x[0], x[1], x[2]), label))
    )

real_train_dataset = make_dataset(REAL_TRAIN_URLS)
fake_train_dataset = make_dataset(FAKE_TRAIN_URLS, label=1)

real_val_dataset=make_dataset(REAL_VAL_URLS, resampled=False, shardshuffle=False)
fake_val_dataset=make_dataset(FAKE_VAL_URLS, resampled=False, shardshuffle=False, label=1)

real_train_loader = wds.WebLoader(real_train_dataset, batch_size=32, num_workers=36, pin_memory = True)
fake_train_loader = wds.WebLoader(fake_train_dataset, batch_size=32, num_workers=36, pin_memory = True)

real_val_loader = wds.WebLoader(real_val_dataset, batch_size=32, num_workers=36, pin_memory = True)
fake_val_loader = wds.WebLoader(fake_val_dataset, batch_size=32, num_workers=36, pin_memory = True)

#---------------------Training----------------------
wandb.init(
    project="fake-inversion",
    config={
        "learning_rate": 0.0001,
        "architecture": "ResNet50-Forensic",
        "norm_layer": "InstanceNorm",
        "batch_size": 64,
        "USE_RESIDUALS": USE_RESIDUALS,
        "input_channels": IN_CHANNELS
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
train_acc_metric = BinaryAccuracy().to(device)
train_loss_metric = MeanMetric().to(device)
val_acc_metric = BinaryAccuracy().to(device)
val_loss_metric = MeanMetric().to(device)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=5,
    threshold=0.002
)
#loop for validatinon inside training loop
def validate(model, real_val_loader, fake_val_loader):
    model.eval()
    val_acc_metric.reset()
    val_loss_metric.reset()

    real_val_iter = iter(real_val_loader)
    fake_val_iter = iter(fake_val_loader)

    with torch.no_grad():
        while True:
            try:
                x_real, y_real = next(real_val_iter)
                x_fake, y_fake = next(fake_val_iter)

                x = torch.cat([x_real, x_fake], dim=0).to(device)
                y = torch.cat([y_real, y_fake], dim=0).to(device).float()

                logits = model(x).view(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y)

                probs = torch.sigmoid(logits)
                val_acc_metric.update(probs, y)
                val_loss_metric.update(loss)

            except StopIteration:
                break

    return val_loss_metric.compute(), val_acc_metric.compute()

#-------------------------train loop ---------------------------------------------

def train_model_dual_loader(model, real_train_loader, fake_train_loader,real_val_loader,fake_val_loader, optimizer, epochs, steps_per_epoch):
    global_step = 0
    real_iter = iter(real_train_loader)
    fake_iter = iter(fake_train_loader)

    for epoch in range(epochs):
        model.train()
        train_acc_metric.reset()
        train_loss_metric.reset()
        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}")

        for step in pbar:
            try:
                x_real, y_real = next(real_iter)
                x_fake, y_fake = next(fake_iter)
                
                x_comb = torch.cat([x_real, x_fake], dim=0)
                y_comb = torch.cat([y_real, y_fake], dim=0)
                
                #shuffles the bacth for random order of fake/real
                idx = torch.randperm(x_comb.size(0))
                x = x_comb[idx].to(device)
                y = y_comb[idx].to(device).float()

                optimizer.zero_grad()
                logits = model(x).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                loss.backward()
                optimizer.step()

                probs = torch.sigmoid(logits)
                train_acc_metric.update(probs, y)
                train_loss_metric.update(loss)

                if global_step % 10 == 0:
                    wandb.log({
                        "batch/loss": loss.item(),
                        "running/acc": train_acc_metric.compute(), #this was mislabeld
                        "global_step": global_step
                    })

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{train_acc_metric.compute():.3f}"
                })
                global_step += 1

            except Exception as e:
                import traceback
                print(f"\n[Error] at step {global_step}:")
                traceback.print_exc() 
                real_iter = iter(real_train_loader)
                fake_iter = iter(fake_train_loader)
                continue

        wandb.log({
            "epoch/loss": train_loss_metric.compute(),
            "epoch/accuracy": train_acc_metric.compute(),
            "epoch": epoch
        })
        
        #validate on validation set
        val_loss, val_acc = validate(model, real_val_loader, fake_val_loader)

        scheduler.step(val_loss)

        wandb.log({
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"]
        })
        
        #save
        save_path = f"resnet_noResiduals_epoch_{epoch}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Saved checkpoint: {save_path}")

# 2x 275 shards = 550. 1k samples per shar = 550k  divided by batch size of 64 = 8,593.75 steps per epoch (on pass over dataset)
train_model_dual_loader(model, real_train_loader, fake_train_loader, real_val_loader,fake_val_loader, optimizer, epochs=25, steps_per_epoch=8593)