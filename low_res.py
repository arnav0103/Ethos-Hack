import os 
import numpy as np
import cv2
from glob import glob

img_sz = (1024,1024)


def load_images(folder):
    images = []
    filenames = []
    for filename in glob(os.path.join(folder, '*.png')):
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
        filenames.append(filename)
    return np.array(images), filenames

def create_low_res(high_res_imgs_dir,out_dir):   
    os.makedirs(out_dir, exist_ok=True)
    print("loading")
    images, filenames = load_images(high_res_imgs_dir)
    print("loaded")

    for img in images:

        random_res = np.random.randint(64, 256)

        low_res = cv2.resize(img, (random_res,random_res))

        kernel_size = np.random.randint(3, 5)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size
        low_res = cv2.filter2D(low_res, -1, kernel)

        low_res = cv2.resize(low_res, img_sz)

        intensity = np.random.uniform(0.1, 0.2)
        noise = np.zeros_like(low_res, dtype=np.uint8)
        cv2.randu(noise, 0, 255)
        low_res = cv2.addWeighted(low_res, 1 - intensity, noise, intensity, 0)

        cv2.imwrite(f'{out_dir}/{os.path.basename(filenames.pop(0))}', low_res)
        
create_low_res('./highres_data', './lowres_data')



