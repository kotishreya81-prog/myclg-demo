import torch
import torch.nn as nn
import torch.optim as optim


# ==========================================================
# CNN + LSTM Fatigue Detection Model
# ==========================================================

class FatigueModel(nn.Module):
    def __init__(self, feature_size=128, hidden_size=64):
        super(FatigueModel, self).__init__()

        # CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1)),  # makes model size independent
            nn.Flatten(),
            nn.Linear(64, feature_size),
            nn.ReLU()
        )

        # LSTM Temporal Model
        self.lstm = nn.LSTM(feature_size, hidden_size, batch_first=True)

        # Output Layer
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x shape: (batch, sequence, 3, 224, 224)
        """

        batch_size, seq_len, C, H, W = x.shape

        features = []

        for t in range(seq_len):
            frame = x[:, t]
            f = self.cnn(frame)
            features.append(f)

        features = torch.stack(features, dim=1)  # (batch, seq, feature)

        _, (hidden, _) = self.lstm(features)

        output = self.fc(hidden[-1])
        return self.sigmoid(output)


# ==========================================================
# Dummy Training + Testing (Standalone Execution)
# ==========================================================

if __name__ == "__main__":

    print("Initializing Fatigue Detection Model...")

    model = FatigueModel()

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Simulated dataset
    batch_size = 4
    sequence_length = 10
    image_size = 224

    # Random dummy input
    dummy_input = torch.randn(batch_size, sequence_length, 3, image_size, image_size)

    # Random labels (0 or 1)
    dummy_labels = torch.randint(0, 2, (batch_size, 1)).float()

    print("Starting Dummy Training...")

    for epoch in range(5):

        optimizer.zero_grad()

        outputs = model(dummy_input)

        loss = criterion(outputs, dummy_labels)

        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch+1}/5], Loss: {loss.item():.4f}")

    print("\nTesting Inference...")

    with torch.no_grad():
        test_output = model(dummy_input)
        print("Output Shape:", test_output.shape)
        print("Sample Output:", test_output)