import torch
import sys
import os
from pathlib import Path

# Add src to path if needed (though running from root usually works if package is installed or pythonpath set)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path: sys.path.append(str(ROOT))

from src.models.afno import AFNOMultiHorizon

def test_afno():
    print("Initializing AFNOMultiHorizon...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Instantiate model
    model = AFNOMultiHorizon(
        input_timesteps=12,
        input_channels=1,
        output_timesteps=6,
        output_channels=1,
        model_name='resnet50d',
        pretrained=False # Avoid downloading weights if not needed for shape test
    ).to(device)
    
    # Dummy input: [Batch, Tin, Cin, H, W]
    B, Tin, Cin, H, W = 2, 12, 1, 880, 970
    dummy_input = torch.randn(B, Tin, Cin, H, W).to(device)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    
    expected_shape = (B, 6, 1, H, W)
    if output.shape == expected_shape:
        print("✅ Output shape matches expectation.")
    else:
        print(f"❌ Output shape mismatch! Expected {expected_shape}, got {output.shape}")
        sys.exit(1)

    # Check non-negativity
    if (output < 0).any():
        print("❌ Output contains negative values (Relu failed?)")
    else:
        print("✅ Output is non-negative.")

    print("Test passed successfully.")

if __name__ == "__main__":
    test_afno()
