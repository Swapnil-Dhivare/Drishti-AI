import numpy as np
import torch
from torch.utils.data import Dataset
from fastai.vision.all import *
import os
from pathlib import Path



class SignLanguageDataset(Dataset):
    def __init__(self, data_root, window_size=90, transform=None):
        self.data_root   = Path(data_root)
        self.window_size = window_size
        self.transform   = transform

        self.samples      = []  # List of (npy_path, class_idx)
        self.class_to_idx = {}  # Map sign name → index
        self.classes      = []  # List of sign names

        # Iterate over each category folder (e.g., Animals, Clothes, House, etc.)
        for category_folder in sorted(self.data_root.iterdir()):
            if not category_folder.is_dir(): 
                continue

            # Inside each category, find the “Landmarks” folder
            landmarks_root = category_folder / "Landmarks"
            if not landmarks_root.exists(): 
                continue

            # Each sign has its own subfolder under Landmarks
            for sign_folder in sorted(landmarks_root.iterdir()):
                if not sign_folder.is_dir(): 
                    continue

                sign_name = sign_folder.name
                # Assign a new index if this sign is first encountered
                if sign_name not in self.class_to_idx:
                    self.class_to_idx[sign_name] = len(self.classes)
                    self.classes.append(sign_name)
                class_idx = self.class_to_idx[sign_name]

                # Collect all .npy files in this sign’s folder
                for npy_path in sign_folder.glob("*.npy"):
                    self.samples.append((npy_path, class_idx))

        print(f"Found {len(self.samples)} samples across {len(self.classes)} sign classes")
        print(f"Sign classes: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        # Load landmark sequence from .npy
        sequence = np.load(npy_path).astype(np.float32)
        # Preprocess to fixed length
        sequence = self.preprocess_sequence(sequence)
        # Convert to torch tensors
        return torch.tensor(sequence), torch.tensor(label, dtype=torch.long)

    def preprocess_sequence(self, sequence):
        """Pad or truncate a sequence to self.window_size frames."""
        frames, features = sequence.shape
        if frames >= self.window_size:
            return sequence[:self.window_size]
        # Pad with zeros if shorter
        padded = np.zeros((self.window_size, features), dtype=np.float32)
        padded[:frames] = sequence
        return padded



def create_dataloaders(data_root, batch_size=32, window_size=90, valid_pct=0.2):
    # Create full dataset
    full_dataset = SignLanguageDataset(data_root, window_size=window_size)
    
    # Split into train/validation
    total_size = len(full_dataset)
    valid_size = int(valid_pct * total_size)
    train_size = total_size - valid_size
    
    train_dataset, valid_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, valid_size]
    )
    
    # Create DataLoaders
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Create FastAI DataLoaders object
    dls = DataLoaders(train_dl, valid_dl)
    dls.c = len(full_dataset.classes)  # Number of classes
    dls.vocab = full_dataset.classes   # Class names
    
    return dls

class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=1040, hidden_size=256, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output for classification
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Classification
        logits = self.classifier(last_output)
        return logits

# Create data
data_root = "Data/INCLUDE"  # Your root folder
dls = create_dataloaders(data_root, batch_size=16, window_size=60)

# Create model
model = SignLanguageLSTM(
    input_size=1040,      # Your landmark features
    hidden_size=256,
    num_layers=2,
    num_classes=len(dls.vocab),  # Number of sign classes
    dropout=0.3
)

# Create FastAI Learner
learn = Learner(dls, model, loss_func=nn.CrossEntropyLoss(), metrics=[accuracy])

# Find optimal learning rate
learn.lr_find()

# Train the model
learn.fit_one_cycle(20, lr_max=3e-4)

# Save the trained model
#learn.save('sign_language_model')
