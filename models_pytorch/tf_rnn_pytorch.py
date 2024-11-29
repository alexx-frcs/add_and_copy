# torch_rnn.py (Version PyTorch corrigée)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .urnn_cell_pytorch import URNNCell
from .lru_cell_pytorch import LRUCell
from .my_RNN_pytorch import SimpleRNNCell
from .new_LRU_pytorch import my_LRUCell
import matplotlib.pyplot as plt


def serialize_to_file(loss, filename='losses.txt'):
    with open(filename, 'w') as file:
        for l in loss:
            file.write("{0}\n".format(l))

class SequenceModel_Complex(nn.Module):
    def __init__(self, hidden_size, num_out):
        super(SequenceModel_Complex, self).__init__()
        # Define the MLP to be applied to each hidden state
        hidden_size *= 2
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            # Add more layers if needed
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_size, num_out)

    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_len, hidden_size]
        
        # Apply MLP to each hidden state at each time step
        mlp_output = self.mlp(hidden_states)  # [batch_size, seq_len, hidden_size]
        
        # Mean Pooling over the sequence length dimension
        # pooled_output = mlp_output.mean(dim=1)  # [batch_size, hidden_size]
        
        outputs_o = self.fc(mlp_output)  # [batch_size, num_out]
        
        return outputs_o
    
class SequenceModel(nn.Module):
    def __init__(self, hidden_size, num_out):
        super(SequenceModel, self).__init__()
        # Define the MLP to be applied to each hidden state
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(), 
            # Add more layers if needed
            # nn.Linear(hidden_size, hidden_size),
            # nn.ReLU(),
        )
        self.fc = nn.Linear(hidden_size, num_out)

    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_len, hidden_size]
        
        # Apply MLP to each hidden state at each time step
        mlp_output = self.mlp(hidden_states)  # [batch_size, seq_len, hidden_size]
        
        # Mean Pooling over the sequence length dimension
        # pooled_output = mlp_output.mean(dim=1)  # [batch_size, hidden_size]
        
        outputs_o = self.fc(mlp_output)  # [batch_size, num_out]
        
        return outputs_o

class TFRNN(nn.Module):
    def __init__(
        self,
        name,
        rnn_cell,
        num_in,
        num_hidden, 
        num_out,
        num_target,
        single_output,
        activation_hidden,
        activation_out,
        optimizer_class,
        loss_function,
        timesteps, 
        n_filters,
        learning_rate=0.001,
        decay=0.9
    ):
        super(TFRNN, self).__init__()

        # Attributs
        self.name = name
        self.loss_list = []
        self.init_state_C = np.sqrt(3 / (2 * num_hidden))
        self.log_dir = './logs/'

        # Fonctions d'activation
        self.activation_out = activation_out
        self.single_output = single_output
        self.num_target = num_target

        # Initialiser la cellule RNN
        if rnn_cell == URNNCell:
            self.cell = rnn_cell(num_units=num_hidden, num_in=num_in)
            # self.w_ho = nn.Parameter(torch.Tensor(2 * num_hidden, num_out))  # Pour URNNCell
            self.mlp_output = SequenceModel_Complex(num_hidden, num_out)
        elif rnn_cell == LRUCell:
            self.cell = rnn_cell(num_units=num_hidden, num_in=num_in,timesteps=timesteps, n_filters=n_filters)
            # self.w_ho = nn.Parameter(torch.Tensor(2 * num_hidden, num_out))  # Pour LRUCell
            self.mlp_output = SequenceModel(num_hidden, num_out)
        elif rnn_cell == nn.LSTMCell:
            self.cell = rnn_cell(input_size=num_in, hidden_size=num_hidden)
            # self.w_ho = nn.Parameter(torch.Tensor(num_hidden, num_out))
            self.mlp_output = SequenceModel(num_hidden, num_out)
        elif rnn_cell == nn.RNNCell:
            self.cell = rnn_cell(input_size=num_in, hidden_size=num_hidden, nonlinearity='tanh')
            # self.w_ho = nn.Parameter(torch.Tensor(num_hidden, num_out))
            self.mlp_output = SequenceModel(num_hidden, num_out)
        elif rnn_cell == SimpleRNNCell:
            self.cell = rnn_cell(num_in=num_in, num_units=num_hidden)
            self.mlp_output = SequenceModel_Complex(num_hidden, num_out)
        elif rnn_cell == my_LRUCell:
            self.cell = rnn_cell(num_in=num_in, num_units=num_hidden, timesteps=timesteps, n_filters=n_filters)
            self.mlp_output = SequenceModel_Complex(num_hidden, num_out)
        else:
            raise NotImplementedError('Unsupported RNN cell type')

        # if rnn_cell != SimpleRNNCell and rnn_cell != my_LRUCell:
        #     nn.init.xavier_uniform_(self.w_ho)

        self.b_o = nn.Parameter(torch.zeros(num_out))

        # Définir l'optimiseur
        if optimizer_class == optim.RMSprop:
            self.optimizer = optimizer_class(self.parameters(), lr=learning_rate, alpha=decay)
        else:
            raise NotImplementedError('Unsupported optimizer')

        # Définir la fonction de perte
        if loss_function == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_function == 'sparse_softmax':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise Exception('Unsupported loss function')

        # Nombre de paramètres entraînables
        t_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print('Network __init__ over. Number of trainable params=', t_params)

    def forward(self, inputs, hidden_state=None):
        """
        Args:
            inputs (Tensor): [batch_size, seq_len, num_in]
            hidden_state (Tensor or tuple, optional): État caché initial
        Returns:
            outputs_o (Tensor): [batch_size, num_out] ou [batch_size, seq_len, num_out]
            final_state (Tensor or tuple): État caché final
        """
        batch_size, seq_len, _ = inputs.size()
        if hidden_state is None:
            hidden_state = self.get_initial_state(batch_size)
        
        outputs = []
        for t in range(seq_len):
            input_t = inputs[:, t, :]  # [batch_size, num_in]
            if isinstance(self.cell, nn.LSTMCell):
                hidden_state = self.cell(input_t, hidden_state)  # (h, c)
                h_t = hidden_state[0]
            else:
                hidden_state = self.cell(input_t, hidden_state)  # [batch_size, hidden_size] 
                h_t = hidden_state  # [batch_size, hidden_size]

            outputs.append(h_t.unsqueeze(1))  # [batch_size, 1, hidden_size]
        
        outputs = torch.cat(outputs, dim=1)  # [batch_size, seq_len, hidden_size]
  
        # if self.single_output:
        #     # Prendre la dernière sortie
        #     # outputs_h_last = outputs[:, -1, :]  # [batch_size, hidden_size]
        #     outputs_h_last = outputs.mean(dim=1)  # [batch_size, hidden_size], mean pooling
        #     preact = torch.matmul(outputs_h_last, self.w_ho) + self.b_o  # [batch_size, num_out]
        #     outputs_o = self.activation_out(preact)  # [batch_size, num_out]
        if self.single_output and (not isinstance(self.cell, SimpleRNNCell) and not isinstance(self.cell, my_LRUCell)): #tentative avec MLP avant pooling
            # Prendre la dernière sortie
            outputs_h_last = outputs[:, -1, :]  # [batch_size, hidden_size]
            outputs_o = self.mlp_output(outputs_h_last)
        elif self.single_output and (isinstance(self.cell, SimpleRNNCell) or isinstance(self.cell, my_LRUCell)):
            outputs_h_last = outputs[:, -1, :]  # [batch_size, hidden_size]
            outputs_o = self.mlp_output(outputs_h_last)
        else:
            # Appliquer à tous les pas de temps
            preact = torch.matmul(outputs, self.w_ho) + self.b_o  # [batch_size, seq_len, num_out]
            outputs_o = self.activation_out(preact)  # [batch_size, seq_len, num_out]

        return outputs_o, hidden_state

    def compute_loss(self, outputs_o, targets):
        """
        Args:
            outputs_o (Tensor): Sorties du modèle
            targets (Tensor): Cibles
        Returns:
            loss (Tensor): Valeur de la perte
        """
        # Assurez-vous que outputs_o et targets sont du même type
        if isinstance(outputs_o, torch.Tensor) and isinstance(targets, torch.Tensor):
            if isinstance(self.loss_fn, nn.MSELoss):
                loss = self.loss_fn(outputs_o, targets)
            elif isinstance(self.loss_fn, nn.CrossEntropyLoss):
                # targets doit être de type LongTensor
                targets = targets.squeeze().long()
                loss = self.loss_fn(outputs_o, targets)
            else:
                raise Exception('Unsupported loss function')
        else:
            raise Exception('Outputs and targets must be tensors')
        return loss

    def train_network(self, dataset, batch_size, epochs):
        """
        Entraîne le réseau RNN avec les données fournies.

        Args:
            dataset (Dataset): Dataset PyTorch
            batch_size (int): Taille des batches
            epochs (int): Nombre d'époques
        """
        # Utiliser DataLoader pour gérer les batches
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        X_val, Y_val = dataset.get_validation_data()
        X_val = X_val.to(next(self.parameters()).device)
        Y_val = Y_val.to(next(self.parameters()).device)

        # Initialiser la liste des pertes
        self.loss_list = []
        print("Starting training for", self.name)
        num_batches = len(dataloader)
        print("NumEpochs:", '{0:3d}'.format(epochs), 
              "|BatchSize:", '{0:3d}'.format(batch_size), 
              "|NumBatches:", '{0:5d}'.format(num_batches),'\n')

        for epoch_idx in range(epochs):
            print("Epoch Starting:", epoch_idx, '\n')
            for batch_idx, (X_batch, Y_batch) in enumerate(dataloader):
                X_batch = X_batch.to(next(self.parameters()).device)
                Y_batch = Y_batch.to(next(self.parameters()).device)

                self.optimizer.zero_grad()
                outputs_o, _ = self.forward(X_batch)
                loss = self.compute_loss(outputs_o, Y_batch)
                loss.backward()
                self.optimizer.step()

                self.loss_list.append(loss.item())

                if batch_idx % 10 == 0:
                    total_examples = (
                        batch_size * num_batches * epoch_idx
                        + batch_size * batch_idx
                        + batch_size
                    )
                    # Sérialiser la perte
                    serialize_to_file(self.loss_list, filename=f'{self.name}_loss.txt')
                    print("Epoch:", '{0:3d}'.format(epoch_idx), 
                          "|Batch:", '{0:3d}'.format(batch_idx), 
                          "|TotalExamples:", '{0:5d}'.format(total_examples), 
                          "|BatchLoss:", '{0:8.4f}'.format(loss.item()))
                if batch_idx % 300 == 0:
                    if type(self.cell)==my_LRUCell:
                        angles = torch.exp(1j * self.cell.phases) # [num_units,]
                        radiuses = torch.exp(-torch.exp(self.cell.log_nus)) # [num_units,]
                        as_ = radiuses * angles # [num_units,]
                        print("as:", as_)
                        #plot as_ on unit disk
                        as_ = as_.real.cpu().detach().numpy() + 1j*as_.imag.cpu().detach().numpy()
                        theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)
                        plt.plot(np.real(as_), np.imag(as_), 'o')
                        plt.plot(np.cos(theta), np.sin(theta))
                        plt.show()

                        

            # Valider après chaque epoch
            with torch.no_grad():
                outputs_val, _ = self.forward(X_val)
                val_loss = self.compute_loss(outputs_val, Y_val).item()
                mean_epoch_loss = np.mean(self.loss_list[-num_batches:])
                print("Epoch Over:", '{0:3d}'.format(epoch_idx), 
                    "|MeanEpochLoss:", '{0:8.4f}'.format(mean_epoch_loss),
                    "|ValidationSetLoss:", '{0:8.4f}'.format(val_loss),'\n')

    def evaluate(self, X, Y):
        """
        Évalue le modèle sur un ensemble de données.

        Args:
            X (numpy.ndarray): Entrées
            Y (numpy.ndarray): Cibles
        Returns:
            average_loss (float): Perte moyenne sur l'ensemble
        """
        # Convertir en tensors et déplacer sur le bon device
        X = torch.tensor(X, dtype=torch.float32).to(next(self.parameters()).device)
        Y = torch.tensor(Y, dtype=torch.float32).to(next(self.parameters()).device)
        with torch.no_grad():
            outputs_o, _ = self.forward(X)
            loss = self.compute_loss(outputs_o, Y).item()
        return loss

    def get_initial_state(self, batch_size):
        """
        Génère l'état caché initial.

        Args:
            batch_size (int): Taille du batch
        Returns:
            hidden_state (Tensor or tuple): État caché initial
        """
        if isinstance(self.cell, URNNCell):
            # Pour URNNCell, l'état caché est [batch_size, 2 * num_units]
            hidden_size = 2 * self.cell.num_units
            return torch.randn(batch_size, hidden_size) * self.init_state_C
        elif isinstance(self.cell, LRUCell):
            # Pour LRUCell, l'état caché est [batch_size, 2 * num_units]
            #the original hidden state is made of 0
            hidden_size = 2 * self.cell.num_units
            return torch.zeros(batch_size, hidden_size)
        elif isinstance(self.cell, my_LRUCell):
            # Pour LRUCell, l'état caché est [batch_size, 2 * num_units]
            #the original hidden state is made of 0
            hidden_size = 2 * self.cell.num_units
            return torch.zeros(batch_size, hidden_size)
        elif isinstance(self.cell, SimpleRNNCell):
            # Pour SimpleRNNCell, l'état caché est [batch_size, num_units]
            hidden_size = self.cell.num_units
            return torch.zeros(batch_size, hidden_size)
        elif isinstance(self.cell, nn.LSTMCell):
            # LSTM nécessite deux états cachés : (h_0, c_0)
            hidden_size = self.cell.hidden_size
            h0 = torch.randn(batch_size, hidden_size) * self.init_state_C
            c0 = torch.randn(batch_size, hidden_size) * self.init_state_C
            return (h0, c0)
        else:
            # Pour RNN, GRU ou autres, un seul état caché suffit
            hidden_size = self.cell.hidden_size
            return torch.zeros(batch_size, hidden_size)

    def test(self, dataset, batch_size=64):
        """
        Teste le modèle sur un ensemble de données.

        Args:
            dataset (Dataset): Dataset PyTorch
            batch_size (int, optional): Taille des batches. Par défaut à 64.
        """
        # Utiliser DataLoader pour le test
        test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        num_batches = len(test_loader)
        for X_batch, Y_batch in test_loader:
            X_batch = X_batch.to(next(self.parameters()).device)
            Y_batch = Y_batch.to(next(self.parameters()).device)
            outputs_o, _ = self.forward(X_batch)
            loss = self.compute_loss(outputs_o, Y_batch)
            total_loss += loss.item()
        average_loss = total_loss / num_batches
        print("Test set loss:", average_loss)

    # Getter pour la liste des pertes
    def get_loss_list(self):
        return self.loss_list