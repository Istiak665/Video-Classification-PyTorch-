import torch
import numpy as np
import joblib
import cv2
import cnn_models
import albumentations
from PIL import Image
from matplotlib import pyplot as plt

# Device check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Computation device: {device}\n")

# load the trained model and label binarizer from disk
print('Loading model and label binarizer...')
lb = joblib.load('../output/lb.pkl')
# print(lb.classes_)
# Import Model
model = cnn_models.CustomCNN().to(device)
# print(model)
# Load model parameters
model.load_state_dict(torch.load('equipments_activities_recognizerv1.pth'))
# Data augmentation
aug = albumentations.Compose([
    albumentations.Resize(224, 224),
    ])

# Capturing the Video using OpenCV
cap = cv2.VideoCapture('../input/test_videos/test_video_002.mp4')

if (cap.isOpened() == False):
    print('Error while trying to read video. Plese check again...')
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# define codec and create VideoWriter object
out = cv2.VideoWriter(('../output/test_video_002.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))


# # read until end of video
while (cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        model.eval()
        with torch.no_grad():
            # conver to PIL RGB format before predictions
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pil_image = aug(image=np.array(pil_image))['image']
            pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
            pil_image = torch.tensor(pil_image, dtype=torch.float).to(device)
            pil_image = pil_image.unsqueeze(0)

            outputs = model(pil_image)
            print(outputs.data)
            _, preds = torch.max(outputs.data, 1)
            print(f"predicted Label: {preds}")

        cv2.putText(frame, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 225, 255), 2)
        cv2.imshow('image', frame)
        out.write(frame)

        # Press "Esc" to exit
        if cv2.waitKey(33) == 27:
            break
        # Press 'q' to exist
        # if cv2.waitKey(27) & 0xFF == ord('q'):
        #     break
    else:
        break
# release VideoCapture()
cap.release()
# out.release()
# close all frames and video windows
cv2.destroyAllWindows()