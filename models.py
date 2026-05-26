import torch.nn as nn

class CNN1Conv(nn.Module):
    def __init__(self, num_features=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(0.35)
        )
        conv_out_length = 180 + 2*2 - 4 + 1
        pool_out_length = conv_out_length // 3
        flattened_size = 64 * pool_out_length
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.classifier(self.features(x))