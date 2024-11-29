# adding_problem.py (Version PyTorch corrigée)

import numpy as np
import torch
from torch.utils.data import Dataset

class AddingProblemDataset(Dataset):
    def __init__(self, num_samples, sample_len, split='train', split_ratio=0.8, seed=None):
        """
        Args:
            num_samples (int): Nombre total d'échantillons à générer.
            sample_len (int): Longueur de chaque séquence d'entrée.
            split (str): 'train', 'validation', ou 'test'.
            split_ratio (float): Proportion des données utilisées pour l'entraînement.
            seed (int, optional): Graine aléatoire pour la reproductibilité.
        """
        super(AddingProblemDataset, self).__init__()
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.sample_len = sample_len
        self.split = split
        self.split_ratio = split_ratio
        
        # Générer les données
        X, Y = self.generate(num_samples)
        
        # Convertir en tensors PyTorch
        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(Y, dtype=torch.float32)
        
        # Split des données en entraînement, validation et test
        train_size = int(split_ratio * num_samples)
        val_size = int((num_samples - train_size) / 2)
        test_size = num_samples - train_size - val_size
        
        self.train_data = (X[:train_size], Y[:train_size])
        self.validation_data = (X[train_size:train_size + val_size], Y[train_size:train_size + val_size])
        self.test_data = (X[train_size + val_size:], Y[train_size + val_size:])
        
    def generate(self, num_samples):
        X_value = np.random.uniform(low=0, high=1, size=(num_samples, self.sample_len, 1))
        X_mask = np.zeros((num_samples, self.sample_len, 1))
        Y = np.ones((num_samples, 1))
        for i in range(num_samples):
            half = int(self.sample_len / 2)
            first_i = np.random.randint(half)
            second_i = np.random.randint(half) + half
            X_mask[i, (first_i, second_i), 0] = 1
            Y[i, 0] = np.sum(X_value[i, (first_i, second_i), 0])
        X = np.concatenate((X_value, X_mask), axis=2)
        return X, Y
    
    def __len__(self):
        if self.split == 'train':
            return self.train_data[0].shape[0]
        elif self.split == 'validation':
            return self.validation_data[0].shape[0]
        elif self.split == 'test':
            return self.test_data[0].shape[0]
        else:
            raise ValueError("split doit être 'train', 'validation' ou 'test'")
    
    def __getitem__(self, idx):
        if self.split == 'train':
            return self.train_data[0][idx], self.train_data[1][idx]
        elif self.split == 'validation':
            return self.validation_data[0][idx], self.validation_data[1][idx]
        elif self.split == 'test':
            return self.test_data[0][idx], self.test_data[1][idx]
        else:
            raise ValueError("split doit être 'train', 'validation' ou 'test'")
    
    def get_validation_data(self):
        return self.validation_data
    
    def get_test_data(self):
        return self.test_data
    
    def get_sample_len(self):
        return self.sample_len
