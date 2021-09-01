import argparse
import os
import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split

def parse_args(args):
    """
    Set args parameters
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_paths", nargs='+', help="define the dir of image and annotation files")
    parser.add_argument("--code_list", nargs='*', default=[], help="Define the select code to train")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Define the ratio of validation set.")
    parser.add_argument("--debug_dir", type=str, default='/temporary/debug/results', help="Define the dir to store output csv")
    parser.add_argument("--predict_method", type=str, default='predict', choices=['predict', 'train'])

    args = parser.parse_args(args)
    return args

def get_subdirs(data_dir):
    for i,(root, dirs, files) in enumerate(os.walk(data_dir)):
        if i == 0:
            sub_dirs = dirs
        return sub_dirs

def get_extensions(path):
    ListFiles = os.walk(path)
    SplitTypes = []
    for root, dirs, files in ListFiles:
        for file in files:
            SplitTypes.append(file.split(".")[-1])
    extensions, counts = np.unique(SplitTypes, return_counts=True)
    return extensions[np.argmax(counts)]

def main(args):
    args = parse_args(args)
    code_list = args.code_list
    lenCode = len(code_list)

    df_total = pd.DataFrame()
    for data_path in args.data_paths:
        img_dir = "{}/Images".format(data_path)
        code_name_list = get_subdirs(img_dir)
        for code_name in code_name_list:
            if lenCode == 0 or (lenCode>0 and code_name in code_list):
                data_dir = os.path.join(img_dir,code_name)
                img_ext = get_extensions(data_dir)
                img_list = sorted(glob.glob(os.path.join(data_dir,'*.'+str(img_ext))))
                for img_cache in tqdm(img_list, f'{code_name}'):
                    try:
                        df_total_cache = pd.DataFrame()
                        base_name = Path(img_cache).stem
                        df_total_cache['image'] = [img_cache]
                        df_total_cache['code'] = [code_name]
                        df_total = pd.concat([df_total, df_total_cache])
                    except:
                        print('Image file %s has something wrong, will not used for train.' % img_cache)
    if args.predict_method == "train":
        train_csv = os.path.join(args.debug_dir, "train.csv")
        val_csv = os.path.join(args.debug_dir, "val.csv")
        df_train, df_valid = train_test_split(df_total, test_size=args.val_ratio, random_state=752)
        df_train.to_csv(train_csv, index=False)
        df_valid.to_csv(val_csv, index=False)
    elif args.predict_method == "predict":
        test_csv = os.path.join(args.debug_dir, "test.csv")
        df_total.to_csv(test_csv, index=False)
