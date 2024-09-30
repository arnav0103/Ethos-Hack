import os
import numpy as np
import cv2
from glob import glob
from mtcnn.mtcnn import MTCNN

img_sz = (1024,1024)
max_faces = 60
out_dir = './inputs/whole_imgs'
os.makedirs(out_dir, exist_ok=True)
detector = MTCNN()

def extract_frames(vid_dir, max_faces, out_dir):

    for vid_path in glob(os.path.join(vid_dir, '*.mp4')):
        cap = cv2.VideoCapture(vid_path)
        total_frms = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frms_to_ext = np.linspace(0, total_frms - 1, max_faces, dtype=int)
        faces_ext = 0

        for frm_idx in frms_to_ext:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frm_idx)
            ret, frm = cap.read()
            #frm = final_enhancement(frm)

            if not ret:
                break

            print(f'{out_dir}/{vid_dir[2::]}_frame_{vid_path.replace("\\","/").split("/")[-1].split(".")[0]}_{faces_ext:02d}.png')
            cv2.imwrite(f'{out_dir}/{vid_dir[2::]}_frame_{vid_path.replace("\\","/").split("/")[-1].split(".")[0]}_{faces_ext:02d}.png', frm)
            faces_ext += 1
        cap.release()
        print(f"Extracted {faces_ext} frames from {vid_path}.")


def final_enhancement(img):
    # Denoising 
    # TODO: Parameters should be tuned 
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    # Sharpening
    # TODO: Change parameters 
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    # Color enhancement
    # y_channel, u_channel, v_channel = cv2.split(img)
    # y_channel = cv2.equalizeHist(y_channel)
    # img = cv2.merge((y_channel, u_channel, v_channel))
    # img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)

    return img


vid_dir = 'video'
extract_frames('./' + str(vid_dir), max_faces,out_dir)
