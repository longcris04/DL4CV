import torch
import torch.nn as nn


class SimpleNeuralNetwork(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=3 * 32 * 32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_classes)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == '__main__':
    model = SimpleNeuralNetwork()
    data = torch.rand(8, 3, 32, 32)   # B, C, H, W

    print(data.shape)
    output = model(data)
    print(output)
