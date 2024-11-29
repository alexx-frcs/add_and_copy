# Simple RNN implementation
import torch
import torch.nn as nn

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
    phases_list = []
    log_nu_list = []
    bs_list = []

    for K_i, S_i in zip(K_values, S_list):
        # Calcul du rayon pour le filtre actuel
        log_nu = torch.full((S_i,), -torch.log(K_i))
        # Calcul des phases pour le filtre actuel
        start = -torch.pi * S_i / (2 * K_i)
        end = torch.pi * S_i / (2 * K_i)
        phases = torch.linspace(start=start, end=end, steps=S_i)
        # phases = torch.randn(S_i) *torch.pi
        # Calcul de as_i (tenseur complexe de taille (S_i,))
        # Calcul de bs_i (tenseur réel de taille (S_i,))
        indices = torch.arange(S_i)  # Exposants entiers de 0 à S_i - 1
        z_i = (-1.0) ** indices
        bs_i = z_i * (torch.exp(2 * alpha) - torch.exp(-2 * alpha)) * torch.exp(-alpha) / K_i
        # Ajout des résultats aux listes
        phases_list.append(phases)
        log_nu_list.append(log_nu)
        bs_list.append(bs_i)

    # Concatenation de tous les as_i et bs_i en vecteurs finaux
    phases = torch.cat(phases_list, dim=0)  # Taille (num_units,)
    log_nus = torch.cat(log_nu_list, dim=0)  # Taille (num_units,)
    bs = torch.cat(bs_list, dim=0)   # Taille (num_units,)
    bs = torch.stack((bs, torch.zeros_like(bs)), dim=-1).view(-1) # Taille (num_units, 2)

    return phases, log_nus, bs


class my_LRUCell(nn.Module):

    def __init__(self, num_units, num_in, timesteps, n_filters):
        super(my_LRUCell, self).__init__()
        self.num_units = num_units
        self.num_in = num_in
        self.max_timesteps = timesteps
        self.n_filters = n_filters
        self.hidden_size = num_units

        phases, log_nus, bs = lru_initialization(num_units, timesteps, n_filters)
        self.phases = nn.Parameter(phases)
        self.log_nus = nn.Parameter(log_nus)

        B = bs.repeat(num_in, 1)
        self.B = nn.Parameter(B)
        # Input -> Hidden
        # self.w_ih = nn.Parameter(torch.Tensor(num_units, num_in))
        # nn.init.xavier_uniform_(self.w_ih)

        # Hidden -> Hidden
        # self.w_hh = nn.Parameter(torch.Tensor(num_units, num_units))
        # nn.init.xavier_uniform_(self.w_hh)

        # Hidden bias
        self.b_h = nn.Parameter(torch.zeros(num_units))

    @property
    def state_size(self):
        return self.num_units
    
    @property
    def output_size(self):
        return self.num_units
    
    def forward(self, inputs, states):
        """
        Args:
            inputs (Tensor): Input tensor of shape [batch_size, num_in].
            states (Tensor): Previous state tensor of shape [batch_size, num_units].
        Returns:
            new_state (Tensor): New state tensor of shape [batch_size, num_units].
        """
        # Previous state
        state = states

        # Linear combination of input
        inputs_mul = torch.matmul(inputs, self.B)
        inputs_mul_c = torch.view_as_complex(inputs_mul.view(inputs_mul.size(0), self.num_units, 2))  # [batch_size, num_units]

        angles = torch.exp(1j * self.phases) # [num_units,]
        radiuses = torch.exp(-torch.exp(self.log_nus)) # [num_units,]
        as_ = radiuses * angles # [num_units,]
        A = torch.diag(as_) # [num_units, num_units]

        state_c = torch.view_as_complex(state.view(state.size(0), self.num_units, 2))  # [batch_size, num_units]

        # Recurrence multiplication
        state_mul = torch.matmul(A,state_c.t()).t() # [batch_size, num_units]

        # New state
        # new_state = torch.tanh(inputs_mul + state_mul + self.b_h)
        new_state_c = inputs_mul_c + state_mul + self.b_h  # [batch_size, num_units]
        new_state = torch.cat((new_state_c.real,new_state_c.imag),dim=1) # [batch_size, 2 * num_units] (real)


        return new_state