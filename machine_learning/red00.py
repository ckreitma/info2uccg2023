import torch

# Definir la red neuronal
class RedNeuronal(torch.nn.Module):
    def __init__(self, entradas, capas_intermedias, salidas):
        super(RedNeuronal, self).__init__()
        self.capa_entrada = torch.nn.Linear(entradas, capas_intermedias[0])
        self.capas_intermedias = torch.nn.ModuleList()
        for i in range(len(capas_intermedias)-1):
            self.capas_intermedias.append(torch.nn.Linear(capas_intermedias[i], capas_intermedias[i+1]))
        self.capa_salida = torch.nn.Linear(capas_intermedias[-1], salidas)
    
    def forward(self, x):
        x = self.capa_entrada(x)
        for capa in self.capas_intermedias:
            x = capa(x)
        x = self.capa_salida(x)
        return x

# Crear la red
red = RedNeuronal(100, [128, 64], 5)

