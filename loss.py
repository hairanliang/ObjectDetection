import torch
import torch.nn as nn

from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss() # (not dealing with multi-label loss here)
        self.sigmoid = nn.Sigmoid()

        # Constants (these can vary)

        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
    
    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # No_object loss
        
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj], (target[..., 0:1][noobj])), # Logical indexing! The 0:1 is just a trick to preserve the shape.
        )

        # Object Loss
        anchors = anchors.reshape(1, 3, 1, 1, 2) # original shape: 3(num_anchors)x2(width,height), need to reshape it so it works out when we do # p_w * exp(t_w)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1) # 1, 2 are x, y position (which we take sigmoid of) and 3, 4 are the width and height
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach() # Don't want the iou gradients to affect backpropagation
        object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj])) # The first value of predictions is the objectness score, and we only care about objectness score of the ones responsible for predicting. We compare our model's predicted objectness scores with the ground truth objectness, which is simply the IOU between ground truth and responsible bounding box.

        # Box Coordinate Loss

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3]) # x, y to be between [0,1]
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        ) # Instead of taking exponent in this case, just take the log of the target. 

        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj]) # Take the MSE loss of the predictions and the targets

        # Class Loss

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()), # QUESTION: Is the 5: a shape trick, or is it because we need to get 80 class probabilities. I think it's needed to get the 80 probabilities, but how does our model know to make the first 0-4 the x,y, w, h, o and then the last 80 to be the class confidences? Just cuz we train it that way?
        )
        
        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss 
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )
