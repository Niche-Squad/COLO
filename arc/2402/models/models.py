from transformers import (
    AutoFeatureExtractor,
    DetrFeatureExtractor,
    DetrForSegmentation,
    AutoModelForSemanticSegmentation,
)
from huggingface_hub import hf_hub_download
import json
import torch
import torch.nn as nn


def init_model(model_name, repo_lbs=None, json_lbs=None) -> nn.Module:
    """
    DESCRIPTION:
    Loading the pretrained models / Deep learning architecture.
    """
    if "mit" in model_name or "segformer" in model_name:
        # nvidia/mit-b0 or nvidia/segformer-b0-finetuned-ade-512-512
        id2label, label2id = get_labels(repo_id=repo_lbs, filename=json_lbs)
        return AutoModelForSemanticSegmentation.from_pretrained(
            model_name, id2label=id2label, label2id=label2id
        )
    elif "detr" in model_name:
        # facebook/detr-resnet-50-panoptic
        return DetrForSegmentation.from_pretrained(model_name)


def get_labels(repo_id, filename):
    """
    # https://huggingface.co/datasets/huggingface/label-files/tree/main
    repo_id = "huggingface/label-files"
    filename = "ade20k-id2label.json" # was "ade20k-hf-doc-builder.json"
    """
    id2label = json.load(open(hf_hub_download(repo_id, filename, repo_type="dataset")))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    return id2label, label2id


def get_features_ext(model_name):
    """
    DESCRIPTION:
    This function returns the feature of the pretrained model. ##
    """

    if "mit" in model_name or "segformer" in model_name:
        # nvidia/mit-b0 or nvidia/segformer-b0-finetuned-ade-512-512
        return AutoFeatureExtractor.from_pretrained(model_name, return_tensors=True)
    elif "detr" in model_name:
        # facebook/detr-resnet-50-panoptic
        return DetrFeatureExtractor.from_pretrained(model_name)
