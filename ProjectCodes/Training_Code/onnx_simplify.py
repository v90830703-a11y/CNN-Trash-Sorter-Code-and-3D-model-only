import onnx
from onnxsim import simplify
import os
print(os.path.getsize("model.onnx"))

print(os.path.getsize("model.onnx.data"))

model = onnx.load("trash_model.onnx")

model_simp, check = simplify(model)

assert check, "Simplification failed"

onnx.save(model_simp, "trash_model_simplified.onnx")

print("Saved trash_model_simplified.onnx")

