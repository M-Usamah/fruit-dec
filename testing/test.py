import torch
import numpy as np
import cv2
from PIL import image
from torchvision import  transforms

model = torch.jit.load('scriptmodule.pt')
model.eval()

# g = open("ls.txt", "r")
# b = g.read()
# data_into_list = b.split("\n")
# g.close()

# print(data_into_list[1])


width = 800
height = 800

cam = cv2.VideoCapture(0)
cam.set(4,width)
cam.set(3,height)

size = (64,64)
trans = transforms.Compose(
    [
        transforms.Resize(size=size),
        # transforms.CenterCrop(50),
        # transforms.ColorJitter(brightness=.5,contrast=.4,hue=.6),
        transforms.RandomEqualize(),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)

def perProcessing(img):
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success, imgOriginal = cam.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(500,320))
    # img = perProcessing(img)
    cv2.imshow("Processed Image", img)
    # classImdex = int(model.eval())
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break