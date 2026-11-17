"""OPTIONAL extension: image-vs-text incongruity with CLIP.
If the farmer attaches a photo, estimate crop condition zero-shot with CLIP and
compare it with the sentiment polarity of the text. Positive text + damaged crop
=> raises sarcasm probability.
"""
from .incongruity import features

PROMPTS = ["a photo of a healthy green crop field", "a photo of a damaged, dry or diseased crop field"]
_clip = None

def damage_score(image):
    """Returns P(damaged) in [0,1] for a PIL image."""
    global _clip
    import torch
    from transformers import CLIPModel, CLIPProcessor
    if _clip is None:
        _clip = (CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
                 CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"))
    m, p = _clip
    inp = p(text=PROMPTS, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        return m(**inp).logits_per_image.softmax(-1)[0, 1].item()

def cross_modal_incongruity(text, image):
    pos = features(text)[0] > 0
    return damage_score(image) if pos else 0.0
