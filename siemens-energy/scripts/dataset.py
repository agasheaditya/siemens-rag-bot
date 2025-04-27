import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class BottleDataset(Dataset):
    def __init__(self, root_dir, image_size=512):
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.image_size = image_size
        
        self.transforms = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),  # Center between [-1, 1]
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        
        # Get the corresponding caption
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                caption = f.read().strip()
        else:
            caption = ""
        
        return {
            'pixel_values': img,
            'text': caption
        }
