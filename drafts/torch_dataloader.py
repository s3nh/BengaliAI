import torch 
import torch.nn as nn 
import os 
import pandas as pd 
import numpy as np 
import cv2 


def main():

    test = pd.read_csv('data/test.csv')
    os.makedirs('data_test', exist_ok=True)

    for el in range(4):
        test = pd.read_parquet('data/test_image_data_{}.parquet'.format(el))
        _ixes = test.image_id
        print(_ixes)
        for _num,  _name in enumerate(_ixes):
            print(_num)
            _img = test.iloc[_num, :].drop('image_id').values.reshape(137, 236)
            _img = _img.astype(np.uint8)
            cv2.resize(_img, (128, 128))
            cv2.imwrite(os.path.join('data_test', '{}.png'.format(_name)), _img)

if __name__ == "__main__":
    main()
