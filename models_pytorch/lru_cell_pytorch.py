#lru_cell.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def lru_initialization(num_units, max_timesteps, n_filters):
    alpha = 1
    # Calcul du nombre d'unités par filtre pour les n_filters - 1 premiers filtres
    S = num_units // n_filters  # Nombre d'unités par filtre
    # Calcul du nombre d'unités restantes pour le dernier filtre
    remainder = num_units - S * (n_filters - 1)
    # Liste des unités par filtre
    S_list = [S] * (n_filters - 1) + [remainder]
    
    # Initialisation de alpha et K si nécessaire
    alpha = torch.tensor(alpha, dtype=torch.float32) if not isinstance(alpha, torch.Tensor) else alpha
    K = max_timesteps

    # Génération des valeurs de K_i allant de K / n_filters à K
    K_values = torch.linspace(K / n_filters, K, steps=n_filters)

    # Listes pour stocker les as_i et bs_i individuels
    as_list = []
    bs_list = []

    for K_i, S_i in zip(K_values, S_list):
        # Calcul du rayon pour le filtre actuel
        radius = torch.exp(-alpha / K_i)
        # Calcul des phases pour le filtre actuel
        start = -torch.pi * S_i / (2 * K_i)
        end = torch.pi * S_i / (2 * K_i)
        phases = torch.linspace(start=start, end=end, steps=S_i)
        # Calcul de as_i (tenseur complexe de taille (S_i,))
        as_i = radius * torch.exp(1j * phases)
        # Calcul de bs_i (tenseur réel de taille (S_i,))
        indices = torch.arange(S_i)  # Exposants entiers de 0 à S_i - 1
        z_i = (-1.0) ** indices
        bs_i = z_i * (torch.exp(2 * alpha) - torch.exp(-2 * alpha)) * torch.exp(-alpha) / K_i
        # Ajout des résultats aux listes
        as_list.append(as_i)
        bs_list.append(bs_i)

    # Concatenation de tous les as_i et bs_i en vecteurs finaux
    as_ = torch.cat(as_list, dim=0)  # Taille (num_units,)
    bs = torch.cat(bs_list, dim=0)   # Taille (num_units,)
    as_ = torch.stack((as_.real, as_.imag), dim=-1).view(-1)
    bs = torch.stack((bs, torch.zeros_like(bs)), dim=-1).view(-1)

    return as_, bs


class LRUCell(nn.Module):

    def __init__(self,num_units, num_in, timesteps, n_filters):
        super(LRUCell, self).__init__()
        self.num_units = num_units #S
        self.num_in = num_in
        self.max_timesteps = timesteps #K
        self.n_filters = n_filters 

        self.B = nn.Parameter(torch.Tensor(2 * num_units, num_in)) #complex matrix, contains both real and imaginary parts
        print('num_units:', num_units, 'num_in:', num_in, 'timesteps:', timesteps, 'n_filters:', n_filters)
        as_, bs = lru_initialization(num_units, timesteps, n_filters)
        self.as_ = nn.Parameter(as_) #size (2*num_units)
        # concatenate num_in copies of bs
        B = bs.repeat(num_in,1)
        self.B = nn.Parameter(B) #complex matrix, contains both real and imaginary parts
        #plot as_ on unit disk
        as_c = torch.view_as_complex(self.as_.view(self.num_units,2))
        print("as init:", as_c)
        as_ = as_c.real.cpu().detach().numpy() + 1j*as_c.imag.cpu().detach().numpy()
        theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        plt.plot(np.real(as_), np.imag(as_), 'o')
        plt.plot(np.cos(theta), np.sin(theta))
        plt.show()



    @property
    def state_size(self):
        return self.num_units * 2
    
    @property
    def output_size(self):
        return self.num_units * 2


    def forward(self, inputs, states):
        """
        Args:
            inputs (Tensor): Input tensor of shape [batch_size, num_in].
            states (Tensor): Previous state tensor of shape [batch_size, 2 * num_units].
        Returns:
            new_state (Tensor): New state tensor of shape [batch_size, 2 * num_units]. (real and imaginary parts)
        """

        # Previous state
        state = states

        # Prepare input linear combination

        inputs_mul = torch.matmul(inputs, self.B)  # [batch_size, 2 * num_units] 
        inputs_mul_c = torch.view_as_complex(inputs_mul.view(inputs_mul.size(0), self.num_units, 2))  # [batch_size, num_units]

        as_c = torch.view_as_complex(self.as_.view(self.num_units,2))
        A = torch.diag(as_c) # [num_units,num_units]

        # Prepare state linear combination (always complex)
        state_c = torch.view_as_complex(state.view(state.size(0), self.num_units, 2)) # [batch_size, num_units]

        #Recurrence multiplication
        state_mul = torch.matmul(A,state_c.t()).t() # [batch_size, num_units]

        # Calculate new state
        #real part new state
        
        new_state_c = state_mul + inputs_mul_c # [batch_size, num_units] (complex)
        # new_state_c = new_state_c.real
        # new_state_c = nn.ReLU(new_state_c)
        new_state = torch.cat((new_state_c.real,new_state_c.imag),dim=1) # [batch_size, 2 * num_units] (real)

        return new_state