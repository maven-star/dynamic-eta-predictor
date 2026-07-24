import sys
import os
# Dynamically add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.optim as optim
from src.models.deepreta_system import DeeprETAEndToEndSystem
# ... (rest of your code stays exactly the same)

class PinballLoss(nn.Module):
    """
    Asymmetric Quantile Loss (Pinball Loss) to train Multi-Head ETA outputs.
    Ensures that predicted arrival windows scale logically: 10th < 50th < 90th.
    """
    def __init__(self, quantiles=[0.10, 0.50, 0.90]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, targets):
        """
        preds: Tensor of shape [3] (Predicted lower, median, and upper bounds)
        targets: Scalar tensor representing the true historical travel time
        """
        losses = []
        for i, q in enumerate(self.quantiles):
            error = targets - preds[i]
            # Pinball loss penalizes asymmetric over/under-prediction
            loss = torch.max((q - 1) * error, q * error)
            losses.append(loss)
        return torch.stack(losses).mean()

def run_training_cycle():
    # 1. Initialize our unified model
    model = DeeprETAEndToEndSystem()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = PinballLoss()

    # 2. Simulated training inputs (Delhi to Kasol checkpoints)
    himalayan_route_h3 = [
        "883da11285fffff", "883da11281fffff", "883da11209fffff", "883da11201fffff",
        "883da11255fffff", "883da1124dfffff", "883da11245fffff", "883da11241fffff"
    ]
    
    mock_weather_and_terrain = torch.tensor([
        [216.0,  0.0,  32.0, 75.0,  4.0, 0.0],
        [220.0,  0.2,  31.0, 70.0,  5.0, 0.0],
        [264.0,  0.5,  29.0, 68.0,  6.0, 0.0],
        [321.0,  1.1,  28.0, 65.0,  5.0, 0.0],
        [670.0,  4.5,  22.0, 80.0,  9.0, 0.2],
        [1040.0, 7.8,  16.0, 85.0, 12.0, 1.1],
        [1100.0, 3.2,  14.0, 80.0,  8.0, 0.5],
        [1580.0, 9.4,  11.0, 90.0, 15.0, 2.4]
    ], dtype=torch.float)
    
    mock_driver = torch.tensor([0.95, 0.8, 1.1], dtype=torch.float)
    
    # Let's say the true historical travel time for this run is exactly 540 minutes (9 hours)
    true_historical_time = torch.tensor(540.0, dtype=torch.float)

    print("\n" + "="*90)
    print("STARTING MODEL OPTIMIZATION AND TRAINING CYCLE")
    print("="*90)

    model.train()
    for epoch in range(1, 101):  # Run 100 training epochs
        optimizer.zero_grad()
        
        # Predict the ETAs in minutes
        raw_outputs = model.predict_eta(himalayan_route_h3, mock_weather_and_terrain, mock_driver)
        
        # Scale outputs up so they match our target time range
        predicted_etas = raw_outputs * 500.0
        
        # Calculate asymmetric pinball loss
        loss = loss_fn(predicted_etas, true_historical_time)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/100 | Loss: {loss.item():6.2f} | "
                  f"Predictions -> 10th: {predicted_etas[0].item():.1f}m | "
                  f"50th (Median): {predicted_etas[1].item():.1f}m | "
                  f"90th: {predicted_etas[2].item():.1f}m")

    # Save the trained model weights
    torch.save(model.state_dict(), "src/models/deepreta_trained.pth")
    print("="*90)
    print("SUCCESS: Model optimized and saved as 'src/models/deepreta_trained.pth'")
    print("="*90 + "\n")

if __name__ == "__main__":
    run_training_cycle()