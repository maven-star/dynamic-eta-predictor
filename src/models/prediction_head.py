# src/models/prediction_head.py
import torch
import torch.nn as nn

class DeeprETAPredictionHead(nn.Module):
    """
    Official Block 3: Uber DeeprETA Multi-Head Quantile Regression.
    Converts structural sequence embeddings into three distinct ETA horizons.
    """
    def __init__(self, embed_dim=128, hidden_dim=64):
        super().__init__()
        
        # Aggregate sequence tokens across the route into a single summary vector
        self.temporal_pooler = nn.AdaptiveAvgPool1d(1)
        
        # Dense fully-connected layers matching Uber's final prediction MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Multi-Head Quantile Outputs: [10th Percentile, 50th Percentile, 90th Percentile]
        self.quantile_head = nn.Linear(hidden_dim, 3)

    def forward(self, encoded_route_tokens):
        """
        Shapes:
          encoded_route_tokens: [SeqLen, EmbedDim] (e.g., [8, 128])
        Output:
          quantiles: [3] -> [Lower Bound, Median ETA, Upper Bound] (in minutes)
        """
        # Step A: Pool sequential nodes down to a single global trip representation
        # [SeqLen, EmbedDim] -> [1, EmbedDim, SeqLen]
        x = encoded_route_tokens.unsqueeze(0).transpose(1, 2)
        pooled = self.temporal_pooler(x).squeeze(-1) # [1, EmbedDim]
        
        # Step B: Pass through dense projection layers
        features = self.mlp(pooled) # [1, HiddenDim]
        
        # Step C: Predict the three time horizons
        raw_predictions = self.quantile_head(features).squeeze(0) # [3]
        
        # Ensure outputs are strictly positive travel times using softplus activation
        predicted_eta_windows = torch.軟plus = nn.functional.softplus(raw_predictions)
        
        return predicted_eta_windows

if __name__ == "__main__":
    # Simulate the exact output tensor we just received from our Step 2 Encoder
    mock_encoder_output = torch.randn(8, 128)
    
    predictor = DeeprETAPredictionHead()
    eta_windows = predictor(mock_encoder_output)
    
    print("\n" + "="*90)
    print("UBER DeeprETA PREDICTION HEAD COMPILATION SUCCESSFUL")
    print("="*90)
    print(f"10th Percentile (Optimistic Lower Window) : {eta_windows[0].item():.2f} mins")
    print(f"50th Percentile (Expected Median ETA)       : {eta_windows[1].item():.2f} mins")
    print(f"90th Percentile (Pessimistic Upper Window)  : {eta_windows[2].item():.2f} mins")
    print("="*90 + "\n")