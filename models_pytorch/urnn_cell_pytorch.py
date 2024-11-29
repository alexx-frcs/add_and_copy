# urnn_cell.py (Version PyTorch corrigée)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiagonalMatrix(nn.Module):
    def __init__(self, name, num_units):
        super(DiagonalMatrix, self).__init__()
        self.name = name
        self.num_units = num_units
        # Initialiser les phases uniformément entre -pi et pi
        self.w = nn.Parameter(torch.rand(num_units) * 2 * np.pi - np.pi)

    def mul(self, z):
        # z: [batch_size, num_units]
        vec = torch.cos(self.w) + 1j * torch.sin(self.w)  # [num_units]
        return vec * z  # Diffusion sur la dimension batch

# Reflection unitary matrix
class ReflectionMatrix(nn.Module):
    def __init__(self, name, num_units):
        super(ReflectionMatrix, self).__init__()
        self.name = name
        self.num_units = num_units
        # Initialiser les parties réelle et imaginaire uniformément entre -1 et 1
        self.re = nn.Parameter(torch.rand(num_units) * 2 - 1)  # [num_units]
        self.im = nn.Parameter(torch.rand(num_units) * 2 - 1)  # [num_units]

    def mul(self, z):
        # z: [batch_size, num_units]
        v = self.re + 1j * self.im  # [num_units]
        vstar = torch.conj(v)  # [num_units]
        # Calcul du produit scalaire entre v* et z
        vstar_z = torch.sum(vstar * z, dim=1, keepdim=True)  # [batch_size, 1]
        # Calcul de la norme au carré
        sq_norm = torch.sum(torch.abs(v)**2)  # scalaire
        factor = 2.0 / sq_norm
        return z - factor * vstar_z * v  # [batch_size, num_units]

# Permutation unitary matrix
class PermutationMatrix(nn.Module):
    def __init__(self, name, num_units):
        super(PermutationMatrix, self).__init__()
        self.name = name
        self.num_units = num_units
        perm = np.random.permutation(num_units)
        self.register_buffer('P', torch.tensor(perm, dtype=torch.long))

    def mul(self, z):
        # z: [batch_size, num_units]
        return z[:, self.P]

# FFTs
# z: complex[batch_size, num_units]

def FFT(z):
    return torch.fft.fft(z, dim=1)

def IFFT(z):
    return torch.fft.ifft(z, dim=1)

def normalize(z):
    norm = torch.sqrt(torch.sum(torch.abs(z)**2, dim=1, keepdim=True))
    factor = norm + 1e-6
    return z / factor

# z: complex[batch_size, num_units]
# bias: real[num_units]
def modReLU(z, bias):
    # relu(|z| + b) * (z / |z|)
    norm = torch.abs(z)
    scale = F.relu(norm + bias) / (norm + 1e-6)
    return scale * z

# URNNCell implementation
class URNNCell(nn.Module):
    """La cellule URNN la plus basique.
    Args:
        num_units (int): Nombre d'unités dans la cellule URNN, taille de la couche cachée.
        num_in (int): Taille du vecteur d'entrée, taille de la couche d'entrée.
    """

    def __init__(self, num_units, num_in):
        super(URNNCell, self).__init__()
        self.num_units = num_units
        self.num_in = num_in

        # Connexion entrée -> cachée
        self.w_ih = nn.Parameter(torch.Tensor(2 * num_units, num_in))
        nn.init.xavier_uniform_(self.w_ih)

        self.b_h = nn.Parameter(torch.zeros(num_units))

        # Matrices unitaires élémentaires pour obtenir la grande
        self.D1 = DiagonalMatrix("D1", num_units)
        self.R1 = ReflectionMatrix("R1", num_units)
        self.P = PermutationMatrix("P", num_units)
        self.D2 = DiagonalMatrix("D2", num_units)
        self.R2 = ReflectionMatrix("R2", num_units)
        self.D3 = DiagonalMatrix("D3", num_units)

    @property
    def state_size(self):
        return self.num_units * 2  # Taille de l'état réel

    @property
    def output_size(self):
        return self.num_units * 2  # Taille de la sortie réelle

    def forward(self, inputs, states):
        """
        La cellule URNN la plus basique.
        Args:
            inputs (Tensor): Tenseur d'entrée de forme [batch_size, num_in].
            states (Tensor): État précédent de forme [batch_size, 2 * num_units].
        Returns:
            new_state (Tensor): Nouveau état de forme [batch_size, 2 * num_units].
        """
        # État précédent
        state = states  # [batch_size, 2 * num_units]

        # Combinaison linéaire d'entrée
        inputs_mul = torch.matmul(inputs, self.w_ih.t())  # [batch_size, 2 * num_units]
        inputs_mul_c = torch.view_as_complex(inputs_mul.view(inputs_mul.size(0), self.num_units, 2))  # [batch_size, num_units]

        # Combinaison linéaire d'état (toujours complexe)
        state_c = torch.view_as_complex(state.view(state.size(0), self.num_units, 2))  # [batch_size, num_units]

        # Appliquer les transformations unitaires
        state_mul = self.D1.mul(state_c)
        state_mul = FFT(state_mul)
        state_mul = self.R1.mul(state_mul)
        state_mul = self.P.mul(state_mul)
        state_mul = self.D2.mul(state_mul)
        state_mul = IFFT(state_mul)
        state_mul = self.R2.mul(state_mul)
        state_mul = self.D3.mul(state_mul)  # [batch_size, num_units]

        # Calculer la pré-activation
        preact = inputs_mul_c + state_mul  # [batch_size, num_units]

        # Appliquer l'activation modReLU
        new_state_c = modReLU(preact, self.b_h)  # [batch_size, num_units] (complexe)
        new_state = torch.cat([new_state_c.real, new_state_c.imag], dim=1)  # [batch_size, 2 * num_units] (réel)

        # La sortie est le nouvel état
        return new_state