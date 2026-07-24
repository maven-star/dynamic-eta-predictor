# src/models/deepreta_system.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib

# =====================================================================
# 1. INDUSTRY-SCALE DEEPRETA ENCODER
# =====================================================================
class RobustDeeprETAEncoder(nn.Module):
    """
    Production-grade Uber DeeprETA Encoder. Uses Feature Hashing to close 
    the spatial Out-Of-Vocabulary loophole, stabilizes continuous weather metrics,
    and applies Linformer attention to process route tokens.
    """
    def __init__(self, vocab_size=5000, h3_dim=32, num_bins=20, 
                 continuous_dim=16, max_seq_len=8, embed_dim=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_bins = num_bins
        self.num_continuous_features = 6  # elevation, incline, temp, humidity, wind, precip
        
        # Closing the loophole: Fixed-size hashing vocabulary matrix
        self.h3_embedding = nn.Embedding(vocab_size, h3_dim)
        
        self.continuous_embeddings = nn.ModuleList([
            nn.Embedding(num_bins, continuous_dim) for _ in range(self.num_continuous_features)
        ])
        
        self.driver_projection = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        sequence_input_dim = h3_dim + (self.num_continuous_features * continuous_dim)
        self.sequence_projection = nn.Linear(sequence_input_dim, embed_dim)
        
        # Linformer Projection Layers
        self.linformer_k = 4 
        self.E_proj = nn.Parameter(torch.randn(self.linformer_k, max_seq_len))
        self.W_proj = nn.Parameter(torch.randn(self.linformer_k, max_seq_len))
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_projection = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        self.final_fusion = nn.Linear(embed_dim + 32, embed_dim)

    def _hash_h3_cells(self, h3_strings):
        """Converts raw string H3 tokens into stable hashed index tensors."""
        hashed_indices = []
        for h3_str in h3_strings:
            # Hash to limit vocabulary and map unseen coordinates safely
            hash_val = int(hashlib.md5(str(h3_str).encode('utf-8')).hexdigest(), 16)
            hashed_indices.append(hash_val % self.vocab_size)
        return torch.tensor(hashed_indices, dtype=torch.long)

    def _discretize_features(self, continuous_tensor):
        """Quantizes raw continuous weather/incline data into stable bins."""
        min_val = continuous_tensor.min(dim=0, keepdim=True)[0]
        max_val = continuous_tensor.max(dim=0, keepdim=True)[0]
        denom = (max_val - min_val) + 1e-8
        normalized = (continuous_tensor - min_val) / denom
        return torch.clamp((normalized * self.num_bins).long(), 0, self.num_bins - 1)

    def forward(self, h3_strings, continuous_features, driver_profile):
        # Step A: Safe Hashed Spatial Lookup
        h3_tokens = self._hash_h3_cells(h3_strings)
        spatial_emb = self.h3_embedding(h3_tokens)
        
        # Step B: Continuous Feature Discretization
        bucketed_indices = self._discretize_features(continuous_features)
        continuous_emb_list = [
            self.continuous_embeddings[i](bucketed_indices[:, i])
            for i in range(self.num_continuous_features)
        ]
        
        seq_x = torch.cat([spatial_emb] + continuous_emb_list, dim=-1)
        seq_x = self.sequence_projection(seq_x).unsqueeze(0)
        
        # Step C: Driver Embedding Integration
        driver_emb = self.driver_projection(driver_profile.unsqueeze(0))
        
        # Step D: Linformer Self-Attention (Sequential Projection)
        residual = seq_x
        q = self.q_linear(seq_x)                                     
        k = self.k_linear(seq_x)                                     
        v = self.v_linear(seq_x)                                     
        
        k_projected = torch.matmul(self.E_proj, k.squeeze(0)).unsqueeze(0) 
        v_projected = torch.matmul(self.W_proj, v.squeeze(0)).unsqueeze(0) 
        
        scores = torch.matmul(q, k_projected.transpose(1, 2)) / (seq_x.size(-1) ** 0.5) 
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, v_projected)        
        
        seq_output = self.layer_norm(residual + self.out_projection(context)).squeeze(0)
        
        # Step E: Latent Fusion
        driver_emb_expanded = driver_emb.repeat(seq_output.size(0), 1)
        fused_representation = torch.cat([seq_output, driver_emb_expanded], dim=-1)
        
        return self.final_fusion(fused_representation)


# =====================================================================
# 2. MULTI-HEAD QUANTILE PREDICTION HEAD
# =====================================================================
class DeeprETAPredictionHead(nn.Module):
    """
    Official Block 3: Converts route embeddings into lower, expected, 
    and upper ETA bounds (Quantiles) using asymmetric regression.
    """
    def __init__(self, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.temporal_pooler = nn.AdaptiveAvgPool1d(1)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.quantile_head = nn.Linear(hidden_dim, 3)

    def forward(self, encoded_route_tokens):
        x = encoded_route_tokens.unsqueeze(0).transpose(1, 2)
        pooled = self.temporal_pooler(x).squeeze(-1)
        features = self.mlp(pooled)
        raw_predictions = self.quantile_head(features).squeeze(0)
        
        # Restrict outputs to positive numbers using softplus
        return nn.functional.softplus(raw_predictions)


# =====================================================================
# 3. END-TO-END SYSTEM INTEGRATION
# =====================================================================
class DeeprETAEndToEndSystem(nn.Module):
    """
    The unified DeeprETA system that coordinates spatial encoding, 
    attention processing, and quantile travel-time predictions.
    """
    def __init__(self):
        super().__init__()
        self.encoder = RobustDeeprETAEncoder()
        self.predictor = DeeprETAPredictionHead()

    def predict_eta(self, h3_strings, continuous_features, driver_profile):
        # 1. Process inputs through the attention layers
        route_embeddings = self.encoder(h3_strings, continuous_features, driver_profile)
        # 2. Extract final quantile ETA bounds
        predicted_quantiles = self.predictor(route_embeddings)
        return predicted_quantiles

if __name__ == "__main__":
    # Test simulation: Delhi to Kasol Route
    print("\n" + "="*90)
    print("SIMULATING TRANSIT SYSTEM: Delhi to Kasol (Himalayan Route Profile)")
    print("="*90)
    
    # Raw string H3 coordinates representing major checkpoints along the highway
    himalayan_route_h3 = [
        "883da11285fffff",  # Delhi Origin
        "883da11281fffff",  # Panipat
        "883da11209fffff",  # Ambala
        "883da11201fffff",  # Chandigarh Transition
        "883da11255fffff",  # Bilaspur (Incline starts)
        "883da1124dfffff",  # Mandi (High-altitude climb)
        "883da11245fffff",  # Bhuntar Junction
        "883da11241fffff"   # Kasol Destination
    ]
    
    # 6 Dynamic variables: [elevation, incline, temp, humidity, wind, precip]
    mock_weather_and_terrain = torch.tensor([
        [216.0,  0.0,  32.0, 75.0,  4.0, 0.0],  # Delhi (Flat, warm)
        [220.0,  0.2,  31.0, 70.0,  5.0, 0.0],  # Panipat
        [264.0,  0.5,  29.0, 68.0,  6.0, 0.0],  # Ambala
        [321.0,  1.1,  28.0, 65.0,  5.0, 0.0],  # Chandigarh
        [670.0,  4.5,  22.0, 80.0,  9.0, 0.2],  # Bilaspur (Climbing)
        [1040.0, 7.8,  16.0, 85.0, 12.0, 1.1],  # Mandi (Rainy, steep)
        [1100.0, 3.2,  14.0, 80.0,  8.0, 0.5],  # Bhuntar
        [1580.0, 9.4,  11.0, 90.0, 15.0, 2.4]   # Kasol (High, freezing climb)
    ], dtype=torch.float)
    
    # Telematics: [speed_multiplier, harsh_braking_rate, aggressive_accel_rate]
    mock_driver = torch.tensor([0.95, 0.8, 1.1], dtype=torch.float) # Cautious driver
    
    # Run predictions
    system = DeeprETAEndToEndSystem()
    eta_bounds = system.predict_eta(himalayan_route_h3, mock_weather_and_terrain, mock_driver)
    
    # Format mock values for visualization
    print(f"Route segments evaluated           : {len(himalayan_route_h3)} geographic steps")
    print(f"Driver Telematics Vector           : Cautious (multiplier {mock_driver[0].item():.2f}x)")
    print("-" * 90)
    print(f"10th Percentile (Best-Case Route Time) : {eta_bounds[0].item() * 100:.1f} mins")
    print(f"50th Percentile (Most Likely Median)   : {eta_bounds[1].item() * 100:.1f} mins")
    print(f"90th Percentile (Worst-Case Time Risk) : {eta_bounds[2].item() * 100:.1f} mins")
    print("="*90 + "\n")