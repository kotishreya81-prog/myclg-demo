import torch
import torch.nn as nn
import torch.optim as optim


class ECG1DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ECG1DCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # 🔥 IMPORTANT FIX
            nn.AdaptiveAvgPool1d(1)   # makes output size fixed
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":

    print("Initializing ECG 1D CNN Model...")

    model = ECG1DCNN()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dummy_input = torch.randn(4, 1, 360)
    dummy_labels = torch.randint(0, 2, (4,))

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