from __future__ import print_function, division
import os 
import torch 
import pandas as pd 
from skimage import io, transform 
import numpy as np 
import matplotlib.pyplot as plt 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

class BengaliTestLoader(Dataset):
    def __init__(self, image_folder, label_file = None, transform=False):
        self.image_folder = image_folder 
        self.label_file = label_file 
        #self.label_file = self.label_file.image_id
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                 std = [0.229 ,0.224, 0.225]),
        ])
        
    def __len__(self):
        return len(os.listdir(self.image_folder)) 
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        trans = transforms.ToTensor()
        img_name = os.path.join(self.image_folder, '{}.png'.format(self.label_file.iloc[idx])) 
        image = Image.open(img_name).convert('RGB')
        #label = self.label_file.iloc[idx]  
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample
            
class BengaliDataLoader(Dataset):
    def __init__(self, image_folder, label_file, transform=False):
        self.image_folder = image_folder  
        self.label_file = pd.read_csv(label_file)
        self.grapheme  =  self.label_file.grapheme_root
        self.vowel = self.label_file.vowel_diacritic
        self.consonant = self.label_file.consonant_diacritic
        self.label_file = self.label_file.image_id
        self.transform =  transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.label_file)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        trans = transforms.ToTensor()
        img_name = os.path.join(self.image_folder, '{}.png'.format(self.label_file.iloc[idx]) )
        image=Image.open(img_name).convert("RGB")
        #image = trans(image) 
        
        #image = transforms.ToPILImage()(image).convert("RGB")
        label = self.label_file.iloc[idx]
        grapheme = self.grapheme.iloc[idx]
        vowel = self.vowel.iloc[idx]
        consonant = self.consonant[idx]
        sample = {'image' : image, 'label' : label, 
                 'grapheme': grapheme, 
                 'vowel' : vowel, 
                 'consonant' : consonant}
        grapheme  =  label[0]
        vowel  = label[1]
        consonant = label[2] 
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

class GraphemeDataset(Dataset):
    def __init__(self, fname):
        print(fname)
        self.df = pd.read_parquet(fname)
        self.data = 255 - self.df.iloc[:, 1:].values.reshape(-1, 128, 128).astype(np.uint8)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        name = self.df.iloc[idx, 0]
        img = (self.data[idx]*(255.0/self.data[idx].max())).astype(np.uint8)
        #img = crop_resize(img)
        img = cv2.resize(img, (128,128))
        img = img.astype(np.float32)/255.0 
        return img, name 
    
def __inference(test_data = list()):
    row_id, target = [], [] 
    for fname in test_data:
        test_image = GraphemeDataset(fname)
        dl = torch.utils.data.DataLoader(test_image, 
                                         batch_size=128, shuffle=False)
        with torch.no_grad():
            for x, y in tqdm(dl):
                x = x.unsqueeze(1).float().cuda()
                p1, p2, p3 = model(x)
                p1 = p1.argmax(-1).view(-1).cpu()
                p2 = p2.argmax(-1).view(-1).cpu()
                p3 = p3.argmax(-1).view(-1).cpu()
                for idx, name in enumerate(y):
                    row_id += [f'{name}_vowel_diacritic', f'{name}_grapheme_root', 
                               f'{name}_consonant_diacritic']
                    target += [p1[idx].item(), p2[idx].item(), p3[idx].item()]
                    
    sub_df = pd.DataFrame({'row_id' : row_id, 'target' : target})
    sub_df.to_csv('submission_csv', index=False)
    sub_df.head(20)
    
    
    
def main():

    pass
if __name__ == '__main__':
    main()