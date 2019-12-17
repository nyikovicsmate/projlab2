import numpy as np
import cv2
import os

def _add_noise(img: np.ndarray) -> np.ndarray:
    """
    Adds noise to the grayscale image.
    :param img:
    :return: image with added noise
    """
    img = np.copy(img)
    # TODO better/more noises
    for i in range(img.shape[0]):
        if i % 2 == 0:
            img[i][0::2] = 1
        else:
            img[i][1::2] = 1
    return img

small_img_list = []
rescaled_img_list = []
predicted_img_list = []
noisy_img_list = []
orig_img_list = []
_train_img_path_list = os.listdir(os.path.join(os.getcwd(), "predictions"))
for img_path in _train_img_path_list:
    name, ext = img_path.split(".")
    idx, type_ = name.split("_")
    img = cv2.imread(os.path.join(os.getcwd(), "predictions",img_path))
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if type_ == "o":
        small_img_list.append(cv2.resize(img, (35,35), interpolation=cv2.INTER_NEAREST))
        rescaled_img_list.append(cv2.resize(small_img_list[-1], (70, 70), interpolation=cv2.INTER_CUBIC))
        orig_img_list.append(img)
        noisy_img_list.append(_add_noise(img))
    elif type_ == "p":
        predicted_img_list.append(img)

small_img_list = np.array(small_img_list, dtype=np.float32)
rescaled_img_list = np.array(rescaled_img_list, dtype=np.float32)
predicted_img_list = np.array(predicted_img_list, dtype=np.float32)
noisy_img_list = np.array(noisy_img_list, dtype=np.float32)
orig_img_list = np.array(orig_img_list, dtype=np.float32)
psnr_dir = os.path.join(os.getcwd(), "psnr")
if not os.path.exists(psnr_dir):
    os.mkdir(psnr_dir)
psnr_or_sum = 0
psnr_op_sum = 0
i=0
with open("psnr.txt","w") as f:
    for s,r,n,p,o in zip(small_img_list, rescaled_img_list, noisy_img_list, predicted_img_list, orig_img_list):
        cv2.imwrite(os.path.join(psnr_dir, f"{i}_r.png"), r)
        cv2.imwrite(os.path.join(psnr_dir, f"{i}_n.png"), n)
        cv2.imwrite(os.path.join(psnr_dir, f"{i}_s.png"), s)
        cv2.imwrite(os.path.join(psnr_dir, f"{i}_p.png"), p)
        cv2.imwrite(os.path.join(psnr_dir, f"{i}_o.png"), o)
        psnr_or = cv2.PSNR(o, r)
        psnr_op = cv2.PSNR(o, p)
        f.write(f"{psnr_or}    {psnr_op}\n")
        psnr_or_sum += psnr_or
        psnr_op_sum += psnr_op
        i += 1
    f.write("\n")
    f.write(f"{psnr_or_sum/i}    {psnr_op_sum/i}\n")



