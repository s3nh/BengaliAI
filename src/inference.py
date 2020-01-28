# SUbmission file loader 

import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from src.dataloader import BengaliDataLoader

import pandas as pd 
# Eval model, inference on every data in 
# test_image_data-file . 

device = 'cuda'
def get_params():
    """ 
    Simple helper 
    """
    model = None 
    test_data = ['data/test_image_data_{}.parquet'.format(el) for el in range(4)]
    predictions = []
    batch_size = 1
    return model, test_data, predictions, batch_size 

def main():
    #model, test_data, predictions, batch_size  = get_params()
    #data = Resize(data)
    test_image = BengaliDataLoader('data_test', 'data/test.csv')
    test_loader = torch.utils.data.DataLoader(test_image, 
                                                batch_size=1, 
                                                shuffle=False) 
    
    with torch.no_grad():
        for idx, (inputs) in tqdm(enumerate(test_loader), total = len(test_loader)):
            inputs.to(device)
            output1, output2, output3 = model(inputs).float().cuda()
            predictions.append(output3.argmax(1).cpu().detach().numpy())
            predictions.append(output2.argmax(1).cpu().detach().numpy())
            predictions.apepnd(output1.argmax(1).cpu().detach().numpy()) 
        
    
if __name__ == "__main__":
    main()