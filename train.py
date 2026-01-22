import torch
import torch.optim as optim
from torch.nn import L1Loss
from models.APBSN import APBSN
from dataloader import get_dataloader
import os
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser(description='.')
parser.add_argument('--train_dir',type=str,default=".",required=True,help='.')
parser.add_argument('--val_dir',type=str,default=".",required=True,help='.')
parser.add_argument('--save_model_path',type=str,default='.',help='.')
args = parser.parse_args()
model = APBSN(bsn='TDBSNl')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
start_epoch = 0
train_dataloader = get_dataloader(image_dir=args.train_dir,noise_type=None,noise_level=0,batch_size=1,shuffle=True,num_workers=0,is_train=True)
val_dataloader = get_dataloader(image_dir=args.val_dir,noise_type=None,noise_level=0,batch_size=1,shuffle=False,num_workers=0,is_train=False)
num_epochs = 100
save_interval = 5
os.makedirs(args.save_model_path, exist_ok=True)
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}', leave=False)
    for inputs, targets, _ in progress_bar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    model.eval()
    val_loss = 0.0
    val_psnr, val_ssim = 0.0, 0.0
    val_progress_bar = tqdm(val_dataloader, desc='Validation', leave=False)
    with torch.no_grad():
        for inputs, targets, _ in val_progress_bar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    if (epoch + 1) % save_interval == 0:
        model_save_path = os.path.join(args.save_model_path, f'model_epoch_{epoch + 1}.pth')
        torch.save({'epoch': epoch + 1,'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict()}, model_save_path)