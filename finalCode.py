from PIL import Image
from IPython.display import display
from matplotlib import pyplot as plt
import os as os
from os import listdir
from os.path import isfile, join
import random
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm

# Parameters you migh wanna change
device = torch.device('cuda:1')
BATCH_SIZE = 1
NUM_WORKERS = 4
NUM_EPOCHS = 181
list_of_paths = ["./train","./val"]
desire_height = 256
desire_width = 512
random_picture = 9

# Generating file name for each file
def gener_file_names(desired_nr_samples, name_folders):
    all_available_img = []
    for folder in name_folders:
        avlbl_images = [folder + "/" + im for im in os.listdir(folder) if os.path.isfile(os.path.join(folder, im))]
        all_available_img += avlbl_images
    random.shuffle(all_available_img)
    return all_available_img[:desired_nr_samples]
all_files_names = gener_file_names(1002, list_of_paths)

# splitising data in 3 sets
def split_t_v_t(big_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    random.seed(44)
    lngth = len(big_list)
    train_list = big_list[:int(lngth*train_ratio)]
    val_list = big_list[int(lngth*train_ratio):int(lngth*(train_ratio+val_ratio))]
    test_list = big_list[int(lngth*(train_ratio+val_ratio)):]
    return train_list, val_list, test_list
train_files, val_files, test_files = split_t_v_t(all_files_names)

# processing images to cv2 and than to than acceptable tensors
def process_file(file_name):
    file_cv = cv2.imread(file_name)
    file_cv = cv2.resize(file_cv, (desire_width, desire_height))
    height, width, _ = file_cv.shape
    width_cutoff = width // 2
    
    s1 = file_cv[:, :width_cutoff]
    s2 = file_cv[:, width_cutoff:]

    s1 = s1.astype('float32')
    s2 = s2.astype('float32')

    s1 /= 255.0
    s2 /= 255.0

    s1 = s1 * 2 - 1
    s2 = s2 * 2 - 1
    
    s1 = s1.transpose(0, 1, 2)
    s2 = s2.transpose(0, 1, 2)

    return(s1,s2)

#applying processing for each file 
def inp_outp_tuple(entry_list, max_workers=8):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_file, entry_list))
    return [r for r in results]
    
train_in_out = inp_outp_tuple(train_files)
val_in_out = inp_outp_tuple(val_files)
test_in_out = inp_outp_tuple(test_files)

# permuting tensors
def cv2_to_tensor(img):
    t = torch.from_numpy(img)              
    t = t.permute(2, 0, 1)
    
    return t

# creaing 3 dataset within Dataset class
class pix2pixDS(Dataset):
    def __init__(self, array_list):
        self.array_list = array_list
        
    def __len__(self):
        return len(self.array_list)

    def __getitem__(self, idx):

        x_cv, y_cv = self.array_list[idx]

        x = cv2_to_tensor(x_cv)   
        y = cv2_to_tensor(y_cv)   

        return x, y

train_ds = pix2pixDS(train_in_out)
val_ds = pix2pixDS(val_in_out)
test_ds = pix2pixDS(test_in_out)

# creating 3 data loaders
train_dataloader = DataLoader(dataset=train_ds,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            # num_workers=NUM_WORKERS,
                            pin_memory=True
                            )

val_dataloader = DataLoader(dataset=val_ds,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            # num_workers=NUM_WORKERS,
                            pin_memory=True
                            )

test_dataloader = DataLoader(dataset=test_ds,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            # num_workers=NUM_WORKERS,
                            pin_memory=True
                            )

# downsampling part of U-net
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# upsampling part of U-net
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        return torch.cat([x, skip], dim=1)

# finale class for U-net architecture
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # encoder
        self.down1 = DownBlock(in_channels, 64, normalize=False)  
        self.down2 = DownBlock(64, 128)                            
        self.down3 = DownBlock(128, 256)                           
        self.down4 = DownBlock(256, 512)                         
        self.down5 = DownBlock(512, 512)                           
        self.down6 = DownBlock(512, 512)                           
        self.down7 = DownBlock(512, 512)                          

        #bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(inplace=True)
        )

        # decoder
        self.up1 = UpBlock(512, 512, dropout=True)
        self.up2 = UpBlock(1024, 512, dropout=True)
        self.up3 = UpBlock(1024, 512, dropout=True)
        self.up4 = UpBlock(1024, 512)
        self.up5 = UpBlock(1024, 256)
        self.up6 = UpBlock(512, 128)
        self.up7 = UpBlock(256, 64)

        # output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        b = self.bottleneck(d7)

        u1 = self.up1(b, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

# discriminator architecture
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        
        def disc_block(in_ch, out_ch, normalize=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *disc_block(in_channels * 2, 64, normalize=False),  
            *disc_block(64, 128),                               
            *disc_block(128, 256),                               
            *disc_block(256, 512),                               

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )
        
    def forward(self, x, y):
        input = torch.cat([x, y], dim=1)
        return self.model(input)

# creating models, choosing loss functions and optimisers
G = UNetGenerator(in_channels=3, out_channels=3).to(device)
D = Discriminator(in_channels=3).to(device)

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

lambda_L1 = 100

optimizer_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# training the model
for epoch in tqdm(range(NUM_EPOCHS)):
    for x, y in train_dataloader:

        x = x.to(device)
        y = y.to(device)

        #training discriminator
        optimizer_D.zero_grad()

        fake_y = G(x)

        #correct pairs
        pred_real = D(x, y)
        loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))

        #fake pairs
        pred_fake = D(x, fake_y.detach())
        loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        optimizer_D.step()

        #training generator
        optimizer_G.zero_grad()

        pred_fake = D(x, fake_y)
        loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_G_L1 = criterion_L1(fake_y, y) * lambda_L1

        loss_G = loss_G_GAN + loss_G_L1
        loss_G.backward()
        optimizer_G.step()
        
    # saving state dictionaries
    if epoch % 10 == 0:
        torch.save(G.state_dict(), f"G_statessnd{epoch}.pt")

    # validating with val dataset
    G.eval()
    val_l1 = 0.0
    
    with torch.no_grad():
        for x, y in val_dataloader:
            x = x.to(device)
            y = y.to(device)
    
            fake_y = G(x)
            val_l1 += criterion_L1(fake_y, y).item()
    
    val_l1 /= len(val_dataloader)
    print(f"Validation L1: {val_l1:.4f}")
    
    G.train()

# saving model
torch.save(G.state_dict(), "G_statessnd.pt")

# printing outcomes from dataset
list_3_imgs = []

for save in range(0, NUM_EPOCHS+1, 10):
    
    G.load_state_dict(torch.load(f"G_statessnd{save}.pt", map_location=device))
    G.eval()
    
    x_cv, y_cv = test_in_out[random_picture]
    x = cv2_to_tensor(x_cv).unsqueeze(0).to(device)
        
    with torch.no_grad():
        fake_y = G(x)
    
    def tensor_to_img(t):
        t = t.squeeze(0).cpu()
        t = (t + 1) / 2
        t = t.clamp(0, 1)
        t = t.permute(1, 2, 0)
        return t.numpy()
    
    x_img = tensor_to_img(x)
    fake_img = tensor_to_img(fake_y)
    gt_img = cv2_to_tensor(y_cv)
    gt_img = tensor_to_img(gt_img.unsqueeze(0))
    list_3_imgs.append((x_img, fake_img, gt_img))
    
# printing images from test set with pyplotlib
for x_img, fake_img, gt_img in list_3_imgs:
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,3,1)
    plt.title("Input")
    plt.imshow(x_img)
    plt.axis("off")
    
    plt.subplot(1,3,2)
    plt.title("Generated")
    plt.imshow(fake_img)
    plt.axis("off")
    
    plt.subplot(1,3,3)
    plt.title("Ground Truth")
    plt.imshow(gt_img)
    plt.axis("off")
    
    plt.show()
print("Done")