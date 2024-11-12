import os
import numpy as np
from PIL import Image, ImageOps
import cv2
import clip
import torch
from torchvision.transforms import functional as F
import argparse
import glob
from model.AWM_Fuse import AWMFuse
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def main(args):
    # ANSI 转义码的前缀和后缀
    RESET = "\033[0m"  # 重置为默认颜色
    BOLD = "\033[1m"  # 粗体
    UNDERLINE = "\033[4m"  # 下划线
    # 颜色
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    save_path = args.save_path
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']
    clip_text = args.input_text
    blip_text_vi = args.blip_vi_text
    blip_text_ir = args.blip_ir_text
    visible_root = args.vi_path
    infrared_root = args.ir_path

    visible_path = [os.path.join(visible_root, i) for i in os.listdir(visible_root)
                  if os.path.splitext(i)[-1] in supported]
    infrared_path = [os.path.join(infrared_root, i) for i in os.listdir(infrared_root)
                  if os.path.splitext(i)[-1] in supported]



    clip_text_path = [os.path.join(clip_text, i) for i in os.listdir(clip_text)
                  if os.path.splitext(i)[-1] in supported]


    filepath_clip, filenames_clip = prepare_clip_path(clip_text)
    filepath_blip1, filenames_blip1 = prepare_blip_path(blip_text_vi)
    filepath_blip2, filenames_blip2 = prepare_blip_path(blip_text_ir)

    visible_path.sort()
    infrared_path.sort()

    print("Find the number of visible image: {},  the number of the infrared image: {}".format(len(visible_path), len(infrared_path)))
    assert len(visible_path) == len(infrared_path), "The number of the source images does not match!"

    print("Begin to run!")
    with torch.no_grad():
        model_clip, _ = clip.load("ViT-B/32", device=device)


        model = testMamba(model_clip=model_clip
                       ).to(device)
        model = torch.nn.DataParallel(model)
        model_weight_path = args.weights_path


        state_dict = torch.load(model_weight_path, map_location=device)



        model.load_state_dict(state_dict)
        print(f"{YELLOW}{UNDERLINE}Successful loading weight from: ", f"{RED}", model_weight_path, f"{RESET}")
        model.eval()

    for i in range(len(visible_path)):
        ir_path = infrared_path[i]
        vi_path = visible_path[i]
        clip_text_path = filepath_clip[i]
        blip_vi_path = filepath_blip1[i]
        blip_ir_path = filepath_blip2[i]

        with open(clip_text_path, 'r', encoding='utf-8') as file:
                clip_text = file.readlines()
                clip_text = ' '.join([line.strip() for line in clip_text])
        blip_text_vi = np.load(blip_vi_path, allow_pickle=True)
        blip_text_ir = np.load(blip_ir_path, allow_pickle=True)

        blip_text_vi = F.to_tensor(blip_text_vi)
        blip_text_ir = F.to_tensor(blip_text_ir)

        blip_text_vi = blip_text_vi.unsqueeze(1).to(device).permute(1,2,3,0)
        blip_text_ir = blip_text_ir.unsqueeze(1).to(device).permute(1,2,3,0)

        img_name = vi_path.replace("\\", "/").split("/")[-1]
        assert os.path.exists(ir_path), "file: '{}' dose not exist.".format(ir_path)
        assert os.path.exists(vi_path), "file: '{}' dose not exist.".format(vi_path)

        ir = Image.open(ir_path).convert(mode="RGB")
        vi = Image.open(vi_path).convert(mode="RGB")
        width, height = vi.size
        ir = F.to_tensor(ir)
        vi = F.to_tensor(vi)


        ir = ir.unsqueeze(0).to(device)
        vi = vi.unsqueeze(0).to(device)
        with torch.no_grad():
            clip_text = clip.tokenize(clip_text).to(device)
          
            i = model(ir, vi, clip_text, blip_text_vi, blip_text_ir)
            fused_img_Y = tensor2numpy(i)

            save_pic(fused_img_Y, save_path, img_name, width, height, 0)

        print("Save the {}".format(img_name))
    print("Finish! The results are saved in {}.".format(save_path))

def tensor2numpy(img_tensor):
    img = img_tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, [1, 2, 0])
    return img

def mergy_Y_RGB_to_YCbCr(img1, img2):
    Y_channel = img1.squeeze(0).detach().cpu().numpy()
    Y_channel = np.transpose(Y_channel, [1, 2, 0])
    img2 = img2.squeeze(0).cpu().numpy()
    img2 = np.transpose(img2, [1, 2, 0])
    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel, CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)
    return merged_img

def save_pic(outputpic, path, index : str, width, height, padding):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic = outputpic[:, :, ::-1]
    save_path = os.path.join(path, index).replace(".jpg", ".png")
    print(save_path)

    if not padding == 0:
        outputpic = outputpic[padding:-padding, padding:-padding, :]

    cv2.imwrite(save_path, outputpic)

def prepare_clip_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.txt"))
    data.sort()
    filenames.sort()
    return data, filenames

def prepare_blip_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.npy"))
    data.sort()
    filenames.sort()
    return data, filenames



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ir_path', default='/home/feecuin/datasets/CVPR25_Dataset/Test/Haze/ir', type=str)
    parser.add_argument('--vi_path', default='/home/feecuin/datasets/CVPR25_Dataset/Test/Haze/vi', type=str)
    parser.add_argument('--weights_path', type=str, default='/home/feecuin/CVPR25/version1/checkpoint/new_loss_from_remote/mamba_0015.pth', help='initial weights path')
    parser.add_argument('--save_path', type=str, default='./result/lxl_5_9_haze_10', help='output save image path')
    parser.add_argument('--input_text',default='/home/feecuin/datasets/CVPR25_Dataset/Test/Haze/hazy_captions',type=str)
    parser.add_argument('--blip_vi_text',default='/home/feecuin/datasets/CVPR25_Dataset/Test/Haze/vi_npy',type=str)
    parser.add_argument('--blip_ir_text',default='/home/feecuin/datasets/CVPR25_Dataset/Test/Haze/ir_npy',type=str)
    parser.add_argument('--device', default='cuda:0', help='device (i.e. cuda or cpu)')
    opt = parser.parse_args()
    main(opt)
