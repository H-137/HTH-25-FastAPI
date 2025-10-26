import torch
import torch.nn as nn

class ClimateGRU(nn.Module):
    def __init__(self, s_in, s_h = 64, n_layers = 1, horizon = 20):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(s_in, s_h, n_layers, batch_first = True)
        self.fc = nn.Linear(s_h, horizon)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :] 
        out = self.fc(out)
        return out
    
def load_model(s_in, path = './model_params.pth'):
    model = ClimateGRU(s_in)
    model.load_state_dict(torch.load(path)) 
    model.eval()
    return model