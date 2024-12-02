import torch.nn.functional as F
import torch.nn as nn
import torch

#class FocalLoss(nn.Module):
#    def __init__(self, alpha=0.75, gamma=3.0, num_classes=3):
#        super(FocalLoss, self).__init__()
#        self.alpha = alpha
#        self.gamma = gamma
#        self.num_classes = num_classes
#
#    def forward(self, inputs, targets):
#        # Apply softmax to the outputs to get class probabilities
#        prob = F.softmax(inputs, dim=1)
#        log_prob = F.log_softmax(inputs, dim=1)
#        
#        # One-hot encoding the targets
#        targets_one_hot = torch.zeros_like(inputs).scatter_(1, targets.unsqueeze(1), 1)
#        
#        # Compute the focal loss
#        loss = -self.alpha * targets_one_hot * (1 - prob) ** self.gamma * log_prob
#        return loss.sum(dim=1).mean()  # Mean over the batch

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()
    

    
class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, inputs, targets):
        # Flatten both inputs and targets to calculate intersection/union
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)

        dice_loss = 1 - (2 * intersection + self.epsilon) / (union + self.epsilon)
        return dice_loss
    
class HingeLoss(nn.Module):
    def forward(self, logits, targets):
        one_hot = F.one_hot(targets, num_classes=logits.shape[1])
        margin = 1.0
        loss = torch.clamp(margin - logits * one_hot, min=0).mean()
        return loss
    
class SymmetricCrossEntropy(nn.Module):
    def __init__(self, alpha=0.1, beta=1.0):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()
        cross_entropy = -torch.sum(one_hot * torch.log_softmax(logits, dim=1), dim=1).mean()
        reverse_kl = -torch.sum(probs * torch.log(one_hot + 1e-6), dim=1).mean()
        return self.alpha * cross_entropy + self.beta * reverse_kl
    
class ConcatenatedLosses(nn.Module):
    def __init__(self, losses: list, weights: list):
        super(ConcatenatedLosses, self).__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, inputs, targets):
        total_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss(inputs, targets)
    
        return total_loss