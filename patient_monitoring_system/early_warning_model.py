import torch
import torch.nn as nn
import torch.optim as optim



class EarlyWarningLSTM(nn.Module):
    def __init__(self, feature_dim=8):
        super(EarlyWarningLSTM, self).__init__()

        self.lstm = nn.LSTM(feature_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)



if __name__ == "__main__":

    print("Initializing Early Warning LSTM...")

    model = EarlyWarningLSTM(feature_dim=8)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

   
    dummy_input = torch.randn(4, 60, 8)

    dummy_labels = torch.randint(0, 2, (4, 1)).float()

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
        print("Predicted Collapse Probability:\n", test_output)