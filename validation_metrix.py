import torch

def binary_dice_score(logits, targets, eps=1e-6):
    """
    logits: [B, 1, H, W]
    targets: [B, 1, H, W]
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def binary_iou_score(logits, targets, eps=1e-6):
    """
    logits: [B, 1, H, W]
    targets: [B, 1, H, W]
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    total = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    union = total - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()