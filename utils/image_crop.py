import cv2
import torch
import numpy as np

from utils.general import (xywh2xyxy, xyxy2xywh)


def crop(xyxy, im, square=False, gain=1.02, pad=10, BGR=False):
    xyxy = torch.tensor(xyxy).view(-1, 4)

    b = xyxy2xywh(xyxy)

    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)

    b[:, 2:] = b[:, 2:] * gain + pad

    xyxy = xywh2xyxy(b).long()

    crop = im[int(xyxy[0, 1]):int(xyxy[0, 3]), int(xyxy[0, 0]):int(xyxy[0, 2]), ::(1 if BGR else -1)]

    # crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    medBlur = cv2.medianBlur(gray, 3)

    gaussBlur = cv2.GaussianBlur(medBlur, (5, 5), 0)

    return gaussBlur
