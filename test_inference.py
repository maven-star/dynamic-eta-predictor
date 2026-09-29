"""Standalone checkpoint smoke test with semantic assertions."""

import torch

from src.models.deepreta_system import DeeprETAEndToEndSystem

MODEL_PATH = "src/models/deepreta_trained.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading core DeeprETA system...")
model = DeeprETAEndToEndSystem()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.to(device).eval()

h3_strings = ["8826801431fffff"] * 8
continuous_features = torch.tensor(
    [[200.0, 1.0, 25.0, 50.0, 12.0, 0.0]] * 8, dtype=torch.float32, device=device
)
driver_profile = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)

with torch.inference_mode():
    predicted_quantiles = model.predict_eta(h3_strings, continuous_features, driver_profile).cpu()

assert predicted_quantiles.shape == (3,)
assert torch.isfinite(predicted_quantiles).all()
assert predicted_quantiles[0] < predicted_quantiles[1] < predicted_quantiles[2]
print("Inference passed; ordered ETA multipliers:", [round(value, 3) for value in predicted_quantiles.tolist()])
