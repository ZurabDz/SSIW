import torch
import numpy as np
import os
import os.path as osp
import cv2

def create_embs_from_names(labels, other_descriptions=None):
    import clip
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CLIP_TEXT_MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
    embs = []
    for name in labels:
        description = other_descriptions[name]

        text = clip.tokenize([description, ]).to(DEVICE)
        with torch.no_grad():
            text_features = CLIP_TEXT_MODEL.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        embs.append(text_features)
    embs = torch.stack(embs, dim=0).squeeze()
    return embs


def get_prediction(embs, gt_embs_list):
    prediction = []
    logits = []
    B, _, _, _ = embs.shape
    for b in range(B):
        score = embs[b,...]
        score = score.unsqueeze(0)
        emb = gt_embs_list
        emb /= emb.norm(dim=1, keepdim=True)
        score /= score.norm(dim=1, keepdim=True)
        score = score.permute(0, 2, 3, 1) @ emb.t()
        # [N, H, W, num_cls] You maybe need to remove the .t() based on the shape of your saved .npy
        score = score.permute(0, 3, 1, 2)  # [N, num_cls, H, W]
        prediction.append(score.max(1)[1])
        logits.append(score)
    if len(prediction) == 1:
        prediction = prediction[0]
        logit = logits[0]
    else:
        prediction = torch.cat(prediction, dim=0)
        logit = torch.cat(logits, dim=0)
    return logit