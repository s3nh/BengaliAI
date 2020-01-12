import gc
from models.fishnet.loader import MoFishnet150, load_checkpoint
import pandas as pd 
# FIshnet path from config? rethink
from src.data_loader import _convert_data, get_and_split

FISHNET_PATH = 'pretrain/fishnet/fishnet150_ckpt_welltrained.tar'
gc.enable()
def main():
    mod = MoFishnet150(path=FISHNET_PATH)
    mod.cuda()
    train_data = pd.read_csv('data/train.csv') 
    print(train_data.shape)
    train_image = _convert_data(train_data = train_data)

    print(train_image.shape)
    x_train, x_test, y_train, y_test = get_and_split(train_image, label = 'grapheme_root')

    x = x_train[0] 
    x = x.cuda() 
    x1, x2, x3 = mod(x)

if __name__ == "__main__":
    main()

