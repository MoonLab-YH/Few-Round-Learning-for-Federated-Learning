from PIL import Image

import torch.nn.functional as F
import numpy as np
import os
import torch

def visualize(imgten, path, color=True, threshold = False, size=None, reverse = False):
    if color: # input should be [C,W,H]
        if imgten.size(0) == 3:
            imgten = imgten.permute([1,2,0])
        if size!=None:
            imgten = F.interpolate(imgten.unsqueeze(dim=0), size=(size,size), mode='bilinear', align_corners=True)
            imgnp = imgten[0].detach().cpu().numpy().transpose([1, 2, 0])
        else:
            imgnp = imgten.cpu().numpy()
        imgnp = np.interp(imgnp, (imgnp.min(), imgnp.max()), (0,255)).astype(np.uint8)
        # imgnp = 255 - imgnp
        img = Image.fromarray(imgnp)
        img.save(path)
    else: #grayscale, input should be [W,H]
        imgten = imgten.unsqueeze(dim=0).unsqueeze(dim=0).float()
        if size!= None:
            imgten = F.interpolate(imgten, size=(size,size), mode='bilinear', align_corners=True)
        imgnp = imgten[0,0].detach().cpu().numpy()
        imgnp = np.interp(imgnp, (imgnp.min(), imgnp.max()), (0,255)).astype(np.uint8)
        if threshold:
            imgnp[imgnp<threshold] = 0; imgnp[imgnp>=threshold] = 255
        if reverse:
            imgnp = 255 - imgnp
        img = Image.fromarray(imgnp)
        img.save(path)

def DeleteContent(path):
    eval_list = os.listdir(path)
    for i in eval_list:
        os.remove(os.path.join(path,i))



def ComputeCI(obsList):
    obsList = np.array(obsList)
    std = np.std(obsList)
    mean = np.mean(obsList)
    denom = np.sqrt(len(obsList))

    return mean, 1.96*std/denom