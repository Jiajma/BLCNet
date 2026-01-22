import torch
from torchvision.utils import save_image
from models.APBSN import APBSN
from dataloader import get_dataloader
import argparse
from tqdm import tqdm
import os
parser = argparse.ArgumentParser(description='.')
parser.add_argument('--test_dir', type=str, default=".",required=True, help='.')
parser.add_argument('--save_dir', type=str, default='.', help='.')
parser.add_argument('--model_path', type=str, required=True, help='.')
args = parser.parse_args()
model = APBSN(bsn='TDBSNl')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
checkpoint = torch.load(args.model_path)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)
model.eval()
test_dataloader = get_dataloader(image_dir=args.test_dir, noise_type=None, noise_level=0, batch_size=1, shuffle=False, num_workers=0, is_train=False)
with torch.no_grad():
    progress_bar = tqdm(test_dataloader, desc='Testing', leave=False)
    for i, (inputs, targets, image_names) in enumerate(progress_bar):
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model.denoise(inputs)
        for j in range(outputs.shape[0]):
            original_name = os.path.splitext(image_names[j])[0]
            save_path = os.path.join(args.save_dir, f'{original_name}.jpg')
            save_image(outputs[j], save_path)