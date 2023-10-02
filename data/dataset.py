import numpy as np
import pandas as pd
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import os


class ChestXray14Dataset(Dataset):
    def __init__(self, 
                data_root, 
                classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
                            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
                            'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'],
                mode='train',
                split='official',
                has_val_set=True,
                transform=None):
        super().__init__()
        self.data_root = data_root
        self.classes = classes
        self.num_classes = len(self.classes)
        self.mode = mode
        if split == 'official':
            self.dataframe, self.num_patients = self.load_split_file(self.data_root, self.mode, has_val_set)
        else:
            self.dataframe, self.num_patients = self.load_split_file_non_official(self.data_root, self.mode)
        self.transform = transform
        self.num_samples = len(self.dataframe)

    def load_split_file(self, folder, mode, has_val=True):
        df = pd.read_csv(os.path.join(folder, 'Data_Entry_2017_v2020.csv'))
        if mode in ['train',  'valid']:
            file_name = os.path.join(folder, 'train_val_list.txt')
            with open(file_name, 'r') as f:
                lines = f.read().splitlines()
            df_train_val = df[df['Image Index'].isin(lines)]
            if has_val:
                patient_ids = df_train_val['Patient ID'].unique()
                train_ids, val_ids = train_test_split(patient_ids, test_size=1-0.7/0.8, random_state=0, shuffle=True)
                target_ids = train_ids if mode == 'train' else val_ids
                df = df_train_val[df_train_val['Patient ID'].isin(target_ids)]
            else:
                df = df_train_val
        elif mode == 'test':
            file_name = os.path.join(folder, 'test_list.txt')
            with open(file_name, 'r') as f:
                target_files = f.read().splitlines()
            df = df[df['Image Index'].isin(target_files)]
        else:
            raise NotImplementedError(f'Unidentified split: {mode}')
        num_patients = len(df['Patient ID'].unique())
        return df, num_patients

    def load_split_file_non_official(self, folder, mode):
        train_rt, val_rt, test_rt = 0.7, 0.1, 0.2
        df = pd.read_csv(os.path.join(folder, 'Data_Entry_2017_v2020.csv'))
        patient_ids = df['Patient ID'].unique()
        train_val_ids, test_ids = train_test_split(patient_ids, test_size=test_rt, random_state=0, shuffle=True)
        train_ids, val_ids = train_test_split(train_val_ids, test_size=val_rt / (train_rt + val_rt), random_state=0, shuffle=True)
    
        target_ids = {'train': train_ids, 'valid': val_ids, 'test': test_ids}[mode]
        df = df[df['Patient ID'].isin(target_ids)]
        num_patients = len(target_ids)
        return df, num_patients
    
    def encode_label(self, label):
        encoded_label = np.zeros(self.num_classes, dtype=np.float32)
        if label != 'No Finding':
            for l in label.split('|'):
                encoded_label[self.classes.index(l)] = 1
        return encoded_label

    def pre_process(self, img):
        h, w = img.shape
        img = cv2.resize(img, dsize=(max(h, w), max(h, w)))
        return img

    def count_class_dist(self):
        class_counts = np.zeros(self.num_classes)
        for index, row in self.dataframe.iterrows():
            class_counts += self.encode_label(row['Finding Labels'])
        return self.num_samples, class_counts

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_file, label = row['Image Index'], row['Finding Labels']
        img = cv2.imread(os.path.join(self.data_root, 'images', img_file))
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        label = self.encode_label(label)
        return img, label
