import torch.nn as nn

WINDOW_SAMPLES = 180
LEARNING_RATE = 3e-4
DROPOUT = 0.5
DROPOUT_HYBRID = 0.35

class CNN1Conv(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=4, padding=2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3), nn.Dropout(DROPOUT)
        )
        pool_out_length = (WINDOW_SAMPLES + 2 * 2 - 4 + 1) // 3
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * pool_out_length, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

class DeepConvLSTM(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=DROPOUT_HYBRID)
        self.dropout = nn.Dropout(DROPOUT_HYBRID)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.conv_block(x).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.classifier(self.dropout(lstm_out[:, -1, :]))

class LSTMModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=200, num_layers=2, batch_first=True, dropout=DROPOUT)
        self.classifier = nn.Sequential(nn.Linear(200, 200), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(200, num_classes))
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, 100), nn.ReLU(), nn.Linear(100, num_classes))
    def forward(self, x):
        return self.network(x)