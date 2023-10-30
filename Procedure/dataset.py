from datasets import *
import pandas as pd 
import numpy as np
import os
from PIL import Image 

dataset_root = "D:\Dataset\\breast_cancer\\"

df_train_calc = pd.read_csv(dataset_root+"csv\calc_case_description_train_set.csv")
df_test_calc = pd.read_csv(dataset_root+"csv\calc_case_description_test_set.csv")

df_train_mass = pd.read_csv(dataset_root+"csv\mass_case_description_train_set.csv")
df_test_mass = pd.read_csv(dataset_root+"csv\mass_case_description_test_set.csv")

def get_image_loc(path) -> list:
    img_list = []
    for img in os.listdir(dataset_root +'jpeg\\'+ path.split('/')[2]):
        img_list.append(dataset_root +'jpeg\\'+ path.split('/')[2]+'\\'+img)
    return img_list


train_dataset_dict = {
    'image_file_path' : [],
    'image' : [],
    'labels' : []
}

validate_dataset_dict = {
    'image_file_path' : [],
    'image' : [],
    'labels' : []
}

test_dataset_dict = {
    'image_file_path' : [],
    'image' : [],
    'labels' : []
}

for idx in range(df_train_calc.shape[0]):
    img_loc_list = get_image_loc(df_train_calc['image file path'][idx])
    for img_loc in img_loc_list:
        image_var = Image.open(img_loc).convert('RGB')
        train_dataset_dict['image_file_path'].append(img_loc)
        train_dataset_dict['image'].append(image_var)
        train_dataset_dict['labels'].append(df_train_calc['pathology'][idx])
        del(image_var)


for idx in range(df_train_mass.shape[0]):
    img_loc_list = get_image_loc(df_train_mass['image file path'][idx])
    for img_loc in img_loc_list:
        image_var = Image.open(img_loc).convert('RGB')
        train_dataset_dict['image_file_path'].append(img_loc)
        train_dataset_dict['image'].append(image_var)
        train_dataset_dict['labels'].append(df_train_mass['pathology'][idx])
        del(image_var)


for idx in range(df_test_calc.shape[0]):
    img_loc_list = get_image_loc(df_train_calc['image file path'][idx])
    for img_loc in img_loc_list:
        image_var = Image.open(img_loc).convert('RGB')
        if idx%2 == 0:
            test_dataset_dict['image_file_path'].append(img_loc)
            test_dataset_dict['image'].append(image_var)
            test_dataset_dict['labels'].append(df_test_calc['pathology'][idx])
        else:
            validate_dataset_dict['image_file_path'].append(img_loc)
            validate_dataset_dict['image'].append(Image.open(img_loc).convert('RGB'))
            validate_dataset_dict['labels'].append(df_test_calc['pathology'][idx])
        del(image_var)
        
for idx in range(df_test_mass.shape[0]):
    img_loc_list = get_image_loc(df_train_calc['image file path'][idx])
    for img_loc in img_loc_list:
        image_var = Image.open(img_loc).convert('RGB')
        if idx%2 == 0:
            test_dataset_dict['image_file_path'].append(img_loc)
            test_dataset_dict['image'].append(image_var)
            test_dataset_dict['labels'].append(df_test_mass['pathology'][idx])
        else:
            validate_dataset_dict['image_file_path'].append(img_loc)
            validate_dataset_dict['image'].append(Image.open(img_loc).convert('RGB'))
            validate_dataset_dict['labels'].append(df_test_mass['pathology'][idx])
        del(image_var)


dataset = DatasetDict({
    'train':Dataset.from_dict(train_dataset_dict),
    'test':Dataset.from_dict(test_dataset_dict),
    'validate':Dataset.from_dict(validate_dataset_dict)},
)

import pickle

ds_file = open("dataset-1", 'ab')

pickle.dump(dataset, ds_file)
ds_file.close()