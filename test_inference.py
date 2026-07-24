import torch
from src.models.deepreta_system import DeeprETAEndToEndSystem

MODEL_PATH = 'src/models/deepreta_trained.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('🧠 Loading core DeeprETA system...')
try:
    model = DeeprETAEndToEndSystem()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print('✅ Model loaded perfectly!')
    
    # 1. Sequence of 8 H3 tokens
    h3_strings = ['8826801431fffff'] * 8
    
    # 2. 2D Tensor for continuous feature bins: shape (8, 6)
    continuous_features = torch.randint(0, 20, (8, 6), dtype=torch.long).to(device)
    
    # 3. 1D Tensor for driver profile: shape (3,)
    driver_profile = torch.randn(3, dtype=torch.float32).to(device)
    
    print('🚀 Running direct forward inference...')
    with torch.no_grad():
        predicted_quantiles = model.predict_eta(
            h3_strings=h3_strings, 
            continuous_features=continuous_features, 
            driver_profile=driver_profile
        )
        
    print('\n🎉 INFERENCE SUCCESSFUL!')
    out_values = predicted_quantiles.cpu().numpy().flatten()
    print(f'10th Percentile (Optimistic) : {out_values[0]:.2f}')
    print(f'50th Percentile (Most Likely): {out_values[1]:.2f}')
    print(f'90th Percentile (Pessimistic) : {out_values[2]:.2f}')

except Exception as e:
    print(f'❌ Core execution failed: {str(e)}')