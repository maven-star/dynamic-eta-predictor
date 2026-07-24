# src/models/encoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeeprETAEncoder(nn.Module):
    """
    Official Block 2: Uber DeeprETA Embedding & Encoder Network.
    Combines spatial-temporal routing tokens with historical driver behavior 
    profiles via Linear Transformer self-attention.
    """
    def __init__(self, num_h3_cells=10000, h3_dim=32, num_bins=20, 
                 continuous_dim=16, max_seq_len=8, embed_dim=128):
        super().__init__()
        
        # 1. Spatial Token Embeddings (H3 Grid Lookup)
        self.h3_embedding = nn.Embedding(num_h3_cells, h3_dim)
        self.num_bins = num_bins
        self.num_continuous_features = 6  # elevation, incline, temp, humidity, wind, precip
        
        # 2. Continuous Discretizer Lookups (Uber's Bucketing Method)
        self.continuous_embeddings = nn.ModuleList([
            nn.Embedding(num_bins, continuous_dim) for _ in range(self.num_continuous_features)
        ])
        
        # 3. Historical Driver Behavior Embedding (Telematics Proxy)
        # Processes the historical driver profile values into a dense latent representation
        self.driver_projection = nn.Sequential(
            nn.Linear(3, 32), # 3 features: speed multiplier, harsh braking, aggressive accel
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        
        # Projects combined sequential inputs (32 spatial + 6 * 16 continuous = 128)
        sequence_input_dim = h3_dim + (self.num_continuous_features * continuous_dim)
        self.sequence_projection = nn.Linear(sequence_input_dim, embed_dim)
        
        # 4. Linear Attention (Linformer) Block
        # Uber projects sequence length down to a fixed dimension (K) to keep computational overhead low
        self.linformer_k = 4 
        self.E_proj = nn.Parameter(torch.randn(self.linformer_k, max_seq_len))
        self.W_proj = nn.Parameter(torch.randn(self.linformer_k, max_seq_len))
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_projection = nn.Linear(embed_dim, embed_dim)
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Final Fusion: Combines the attention-weighted sequence matrix with the global driver profile
        self.final_fusion = nn.Linear(embed_dim + 32, embed_dim)

    def _discretize_features(self, continuous_tensor):
        """Uber Feature Discretization: Quantizes raw continuous values into uniform bucket tokens."""
        min_val = continuous_tensor.min(dim=0, keepdim=True)[0]
        max_val = continuous_tensor.max(dim=0, keepdim=True)[0]
        denom = (max_val - min_val) + 1e-8
        normalized = (continuous_tensor - min_val) / denom
        return torch.clamp((normalized * self.num_bins).long(), 0, self.num_bins - 1)

    def forward(self, h3_tokens, continuous_features, driver_profile):
        """
        Shapes:
          h3_tokens: [SeqLen]
          continuous_features: [SeqLen, 6]
          driver_profile: [3] -> [speed_multiplier, harsh_braking, aggressive_accel]
        """
        # Step A: Resolve Spatiotemporal Sequential Embeddings
        spatial_emb = self.h3_embedding(h3_tokens)
        bucketed_indices = self._discretize_features(continuous_features)
        
        continuous_emb_list = [
            self.continuous_embeddings[i](bucketed_indices[:, i])
            for i in range(self.num_continuous_features)
        ]
        
        # Combine and project into Transformer space
        seq_x = torch.cat([spatial_emb] + continuous_emb_list, dim=-1)
        seq_x = self.sequence_projection(seq_x).unsqueeze(0)  # Shape: [1, SeqLen, EmbedDim]
        
        # Step B: Process Global Driver Profile History
        driver_emb = self.driver_projection(driver_profile.unsqueeze(0)) # Shape: [1, 32]
        
        # Step C: Execute Linformer Linear Self-Attention Layer
        residual = seq_x
        q = self.q_linear(seq_x)                                     # [1, SeqLen, EmbedDim]
        k = self.k_linear(seq_x)                                     # [1, SeqLen, EmbedDim]
        v = self.v_linear(seq_x)                                     # [1, SeqLen, EmbedDim]
        
        # DOWN-PROJECTION (The Linformer Core): Project SeqLen (8) -> Linformer_K (4)
        # We multiply the projection parameter [4, 8] by the sequence dimensions [8, 128]
        # Transposing handles the batch dimension safely
        k_projected = torch.matmul(self.E_proj, k.squeeze(0)).unsqueeze(0) # [1, K, EmbedDim]
        v_projected = torch.matmul(self.W_proj, v.squeeze(0)).unsqueeze(0) # [1, K, EmbedDim]
        
        # Compute Scaled Dot-Product Attention over the compressed sequence dimension [1, SeqLen, K]
        scores = torch.matmul(q, k_projected.transpose(1, 2)) / (seq_x.size(-1) ** 0.5) 
        attn_weights = F.softmax(scores, dim=-1)
        
        context = torch.matmul(attn_weights, v_projected)                  # [1, SeqLen, EmbedDim]
        seq_output = self.layer_norm(residual + self.out_projection(context)).squeeze(0) # [SeqLen, EmbedDim] # [SeqLen, EmbedDim]
        
        # Step D: Fuse Global Driver History with the Attention Tokens
        # Expand driver embedding to match the sequence length dimension for concatenation
        driver_emb_expanded = driver_emb.repeat(seq_output.size(0), 1) # [SeqLen, 32]
        fused_representation = torch.cat([seq_output, driver_emb_expanded], dim=-1) # [SeqLen, EmbedDim + 32]
        
        final_encoded_route = self.final_fusion(fused_representation) # [SeqLen, EmbedDim]
        return final_encoded_route

if __name__ == "__main__":
    # Test execution matching our 8-point route data profile
    mock_h3 = torch.tensor([12, 45, 67, 89, 34, 11, 90, 54], dtype=torch.long)
    mock_continuous = torch.randn(8, 6) 
    
    # Mock driver: speed_multiplier=1.08, harsh_braking=1.2, aggressive_accel=2.4
    mock_driver = torch.tensor([1.08, 1.2, 2.4], dtype=torch.float)
    
    encoder = DeeprETAEncoder()
    encoded_route = encoder(mock_h3, mock_continuous, mock_driver)
    
    print("\n" + "="*90)
    print("UBER DeeprETA ENCODER BLOCK 2 SUITE VERIFICATION")
    print("="*90)
    print(f"Route Checkpoint Sequences Scale   : {mock_h3.shape[0]} nodes")
    print(f"Driver Telematics Vector Parsed    : {mock_driver.tolist()}")
    print(f"Final Integrated Structural Matrix  : {encoded_route.shape} -> [Nodes, EmbedDim]")
    print("="*90 + "\n")