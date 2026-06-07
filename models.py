import torch.nn as nn

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

WINDOW_SAMPLES = 180
DROPOUT = 0.35

def get_classical_model():
    if config.CLASSICAL_MODEL == "SVM":
        return SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    elif config.CLASSICAL_MODEL == "RF":
        return RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1, random_state=42)
    elif config.CLASSICAL_MODEL == "KNN":
        return KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    raise ValueError(f"Modelo clássico {config.CLASSICAL_MODEL} não reconhecido.")

class CNN1Conv(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=4, padding=2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3), nn.Dropout(config.DROPOUT)
        )
        pool_out_length = (config.WINDOW_SAMPLES + 2 * 2 - 4 + 1) // 3
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
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=config.DROPOUT)
        self.dropout = nn.Dropout(config.DROPOUT)
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        x = self.conv_block(x).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.classifier(self.dropout(lstm_out[:, -1, :]))

class LSTMModel(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=200, num_layers=2, batch_first=True, dropout=config.DROPOUT)
        self.classifier = nn.Sequential(nn.Linear(200, 200), nn.ReLU(), nn.Dropout(config.DROPOUT), nn.Linear(200, num_classes))
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.classifier(hidden[-1])

# class MLP(nn.Module):
#    def __init__(self, input_size, num_classes):
#        super().__init__()
#        self.network = nn.Sequential(nn.Linear(input_size, 100), nn.ReLU(), nn.Linear(100, num_classes))
#    def forward(self, x):
#        return self.network(x)

class CNN1Conv_MultiTask(nn.Module):
    def __init__(self, num_features, num_classes_dict):
        super().__init__()
        
        self.shared_features = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=4, padding=2), nn.ReLU(),
            nn.MaxPool1d(kernel_size=3), nn.Dropout(config.DROPOUT)
        )
        pool_out_length = (config.WINDOW_SAMPLES + 2 * 2 - 4 + 1) // 3
        shared_size = 64 * pool_out_length
        
        self.heads = nn.ModuleDict()
        for task_name, out_features in num_classes_dict.items():
            self.heads[task_name] = nn.Sequential(
                nn.Flatten(),
                nn.Linear(shared_size, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, out_features)
            )

    def forward(self, x):
        shared = self.shared_features(x)
        return {task_name: head(shared) for task_name, head in self.heads.items()}


class DeepConvLSTM_MultiTask(nn.Module):
    def __init__(self, num_features, num_classes_dict):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True, dropout=config.DROPOUT)
        self.dropout = nn.Dropout(config.DROPOUT)
        
        self.heads = nn.ModuleDict()
        for task_name, out_features in num_classes_dict.items():
            self.heads[task_name] = nn.Linear(128, out_features)

    def forward(self, x):
        x = self.conv_block(x).permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        
        shared = self.dropout(lstm_out[:, -1, :])
        
        return {task_name: head(shared) for task_name, head in self.heads.items()}


class LSTMModel_MultiTask(nn.Module):
    def __init__(self, num_features, num_classes_dict):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=num_features, hidden_size=200, num_layers=2, batch_first=True, dropout=config.DROPOUT)
        
        self.heads = nn.ModuleDict()
        for task_name, out_features in num_classes_dict.items():
            self.heads[task_name] = nn.Sequential(
                nn.Linear(200, 200), nn.ReLU(), 
                nn.Dropout(config.DROPOUT), 
                nn.Linear(200, out_features)
            )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        
        shared = hidden[-1]
        
        return {task_name: head(shared) for task_name, head in self.heads.items()}

# class MLP_MultiTask(nn.Module):
#    def __init__(self, input_size, num_classes_dict):
#        super().__init__()
#        self.shared_features = nn.Sequential(nn.Linear(input_size, 100), nn.ReLU())
#        self.heads = nn.ModuleDict()
#        for task_name, out_features in num_classes_dict.items():
#            self.heads[task_name] = nn.Linear(100, out_features)
#    def forward(self, x):
#        shared = self.shared_features(x)
#        return {task_name: head(shared) for task_name, head in self.heads.items()}