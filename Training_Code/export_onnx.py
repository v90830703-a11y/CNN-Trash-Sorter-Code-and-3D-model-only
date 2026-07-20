import torch
from Model import TrashCNN
import os

# 1. Create model
model = TrashCNN(num_classes=4)

# 2. Load checkpoint
checkpoint = torch.load(
    "Training_Code/checkpoint.pth",
    map_location="cpu"
)

# 3. Extract ONLY model weights
state_dict = checkpoint["model_state_dict"]

# 4. Load weights into model
model.load_state_dict(state_dict)


dummy = torch.randn(1, 3, 96, 96)

torch.onnx.export(
    model,
    dummy,
    "Training_Code/model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=12
)

print(os.path.getsize("Training_Code/model.onnx") / 1024, "KB")
print(model)