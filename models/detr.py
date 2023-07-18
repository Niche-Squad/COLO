import os
import json
from transformers import DetrForObjectDetection, DetrImageProcessor, DetrConfig

# DETR fine-tuning note
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb

# local imports
from .device import get_device


model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

model.train()
