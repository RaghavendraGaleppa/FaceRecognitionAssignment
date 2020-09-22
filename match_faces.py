# A python program to match two faces

import sys
import cv2
from utils import CenterFaceAlign
from utils import FaceMatch
import torch

if len(sys.argv) < 3:
	raise ValueError(f"Usage: python3 match_faces.py file_1.jpg file_2.jpg")

file_1 = sys.argv[1]
file_2 = sys.argv[2]

img_1 = cv2.imread(file_1)
img_2 = cv2.imread(file_2)

face_align = CenterFaceAlign()
img_1 = face_align(img_1)/255.0
img_2 = face_align(img_2)/255.0

img_1 = torch.from_numpy(img_1)
img_2 = torch.from_numpy(img_2)

face_matcher = FaceMatch(PATH="model.ht")
face_matcher(img_1, img_2)
