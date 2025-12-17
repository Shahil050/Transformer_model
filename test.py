import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.layer_norm = nn.LayerNorm(128)  
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.layer_norm(x) 
        x = torch.relu(x)
        x = self.fc2(x)
        return x

input_data = torch.randn(2, 3)
print(f"input data:{input_data}")
model = SimpleNN(3,2)
output = model(input_data)
print(output)