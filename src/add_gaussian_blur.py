import cv2
import os

from tqdm import tqdm

src_dir = 'D:\Sem 6th\Project\image-deblurring-using-deep-learning\image-deblurring-using-deep-learning\input\sharp'
images = os.listdir(src_dir)
dst_dir = '../input/gaussian_blurred'

for i, img in tqdm(enumerate(images), total=len(images)):
    img = cv2.imread(f"{src_dir}/{images[i]}")
    # add gaussian blurring
    blur = cv2.GaussianBlur(img, (51, 51), 0) # kernel size of 51 * 51
    cv2.imwrite(f"{dst_dir}/{images[i]}", blur)

print('DONE')