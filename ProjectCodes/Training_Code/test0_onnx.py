import onnxruntime as ort
import numpy as np
from PIL import Image
from torchvision import transforms
import os


session = ort.InferenceSession("Training_Code/Tmodel.onnx")

transform = transforms.Compose([
    transforms.Resize((96, 96)),  # use your training size
    transforms.ToTensor(),
])

img = Image.open("Training_Code/test_img.jpg").convert("RGB")
x = transform(img).unsqueeze(0).numpy()

outputs = session.run(None, {session.get_inputs()[0].name: x})
print(outputs[0])
print("Pred:", np.argmax(outputs[0]))


logits = outputs[0][0]

exp = np.exp(logits - np.max(logits))
probs = exp / exp.sum()

print(probs)


#test onnx file size

# import onnx
# import numpy as np
# import os


# model = onnx.load("model.onnx")

# print(len(model.graph.initializer))
# print(os.path.abspath("model.onnx"))
# print(os.path.getsize("model.onnx"))

# model = onnx.load("model.onnx")

# total_params = 0

# for init in model.graph.initializer:
#     n = np.prod(init.dims)
#     total_params += n
#     print(init.name, init.dims, n)

# print("Total params:", total_params)