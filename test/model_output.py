import numpy as np
import ultralytics
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")

image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8,)
# turn image to tensor
image = torch.tensor(image).permute(2, 0, 1).float()/255
image = image.unsqueeze(0)

plt.imshow(image)
model

model_fp = model.model
out = model_fp(image)
out_org = model(image)


out[0].shape

len(out[1])
out[1][0].shape
out[1][1].shape
out[1][2].shape

