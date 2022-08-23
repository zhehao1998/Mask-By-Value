import cv2
import numpy as np
import copy

def showMask(mask):
    mask_original = cv2.imread(mask)
    mask_copy = copy.deepcopy(mask_original)
    mask_copy *= np.asarray((255/mask_original.max()), np.uint8)
    cv2.imshow("Binary mask", mask_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()