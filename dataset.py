import config
import numpy as np
import os
import pandas as pd
import torch

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from utils import (
    iou_width_height as iou,
    non_max_suppression as nms
)

ImageFile.Load_TRUNCATED_IMAGES = True

class YoloDataset(Dataset):
    def __init__(
            self,
            csv_file,
            img_dir, label_dir,
            anchors,
            image_size=416,
            S=[13, 26, 52],
            C=20,
            transform=None,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S # grid sizes
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2]) # Adds 3 lists into one list and then converts this list into a tensor
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
    
    def __len__(self):
        return len(self.annotations) 
    
    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # it's 1 because the path is in the second column of the csv
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() # Shape is: (num_classes, 5), so axis=1 will make it correctly shift all bounding boxes to correct orientation --> per bounding box, right now it's [class, x, y, w, h], and they want [x, y, w, h, class] so use np.roll (there are literally 5 values and we are rolling class to the last value (none of the values are dimensions: it's the actual value for one label))
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB")) # some are grayscale, so need to change to RGB 

        if self.transform: # The following code is all using the Albumentations library, so I guess this is how it all gets formatted
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        
        # The targets shape is like this because per scale, we will have 3 possible anchor boxes. For each of these bounding boxes, there will be different targets, depending on if the bounding box is good/responsible for predicting (objectness=1, various values for x, y, w, h that should be predicted) or not (objectness=0, x, y, w, h set to 0 or some default value)
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] # [p_o, x, y, w, h, c] Setting every single, most importantly the objectness (p_o) value to 0 for the anchor_taken calculation

        # We begin by iterating over the ground truth bounding boxes for this particular image
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4], self.anchors)) # 2:4 just gets width and height (w, h) and with the anchors it calculates IOU for particular box and all 9 anchors using utility function
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box

            has_anchor = [False, False, False] # Must have anchor for each scale for each ground truth bounding box

            # finding out, for each scale, which of the anchors fit best for the current box
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale # 0, 1, 2
                # Finds which anchor on the scale 
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # 0, 1, 2
                S = self.S[scale_idx] # This gets an integer value
                # Everything in YOLO is relative to the cell, but in the labels, it's relative to the image
                i, j = int(S*y), int(S*x) # x = 0.5, S = 13 --> int(6.5) --> 6 (cell 6)
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0] # THIS IS FOR THE RARE CASE AN ANCHOR FITS TWO OBJECTS WELL. If this anchor has not been picked yet, it will equal 0. Once it's been picked up already, we will have set it to 1

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1 # Sets this to 1 because it has been seen and also this one will be used to predict(objectness=1)
                    x_cell, y_cell = S*x - j, S*y - i # both between [0, 1]
                    width_cell, height_cell = (
                        width*S, # S = 13, width=0.5 --> 6.5
                        height*S
                    )

                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )

                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True # I ADDED THIS: did he forget it?
                
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1 # We set -1 as a sign to ignore this prediction
        return image, tuple(targets)