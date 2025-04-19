import torch
from torchvision import models, transforms
from config import DEVICE, FEATURE_LAYER


def extract_features(img):
    # img: HxWx3 RGB numpy
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    x = tf(img).unsqueeze(0).to(DEVICE)
    model = models.resnet50(pretrained=True).to(DEVICE)
    model.eval()
    feats = {}
    # Hook specified layer
    def hook(module, inp, out): feats['feat'] = out.detach()
    getattr(model, FEATURE_LAYER).register_forward_hook(hook)
    with torch.no_grad(): model(x)
    fmap = feats['feat'].squeeze(0).cpu().numpy()  # [C,h',w']
    return fmap