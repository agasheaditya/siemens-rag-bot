import os
import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import get_peft_model, LoraConfig, TaskType, LoraModel

from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ----------------------------
# CONFIGURATIONS
# ----------------------------
model_id = "CompVis/stable-diffusion-v1-4"
output_dir = "lora_finetune/output"
batch_size = 2
image_size = 512
learning_rate = 1e-4
num_epochs = 5
use_fp16 = True

train_dataset_path = "../dataset/bottle/image/" 

# ----------------------------
# DATASET
# ----------------------------
class CustomBottleDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # For now dummy caption (improve later!)
        caption = "a defective bottle"

        return {"image": image, "caption": caption}

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

train_dataset = CustomBottleDataset(root_dir=train_dataset_path, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ----------------------------
# MODEL
# ----------------------------
print("Loading Stable Diffusion...")
pipe = StableDiffusionPipeline.from_pretrained( model_id, revision="fp16", torch_dtype=torch.float16 if use_fp16 else torch.float32).to("cuda")

unet = pipe.unet

# ----------------------------
# Apply LoRA
# ----------------------------
print("Applying LoRA adapters...")
lora_config = LoraConfig(
    r=4,
    lora_alpha=32,
    target_modules=["to_k", "to_q", "to_v", "to_out"],
    lora_dropout=0.1,
    bias="none",
)

unet = LoraModel(unet, lora_config, adapter_name="damage_lora")
# unet = get_peft_model(unet, lora_config)

# ----------------------------
# Optimizer
# ----------------------------
optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

# ----------------------------
# Training Loop
# ----------------------------
print("Starting Training...")

unet.train()

for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in pbar:
        optimizer.zero_grad()

        images = batch["image"].to("cuda")

        noise = torch.randn_like(images)
        timesteps = torch.randint(0, 1000, (images.size(0),), device=images.device).long()

        noisy_images = pipe.scheduler.add_noise(images, noise, timesteps)

        model_pred = unet(noisy_images, timesteps).sample

        loss = torch.nn.functional.mse_loss(model_pred, noise)
        loss.backward()

        optimizer.step()

        pbar.set_postfix(loss=loss.item())

    # Save checkpoint after each epoch
    os.makedirs(output_dir, exist_ok=True)
    torch.save(unet.state_dict(), os.path.join(output_dir, f"unet_lora_epoch{epoch+1}.pt"))

print("Fine-tuning Complete and Checkpoints Saved!")

