from __future__ import division

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import SCUT_ROOT, SCUTAnnotationTransform, SCUTDetection, BaseTransform
from data import SCUT_CLASSES as labelmap
import torch.utils.data as data

from ssd import build_ssd

from PIL import Image
import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt 
import fchd_utils as utils

SAVE_FLAG = 0
THRESH = 0.02
IM_RESIZE = False

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def detect(img_path):
    img = cv2.imread(img_path)
    height, width, channels = img.shape
    x = cv2.resize(img, (300, 300)).astype(np.float32)
    x -= (104, 117, 123)
    x = x.astype(np.float32)
    img = x[:, :, (2, 1, 0)]
    im =  torch.from_numpy(img).permute(2, 0, 1)
    h =  height
    w = width
    x = Variable(im.unsqueeze(0))
    if args.cuda:
        x = x.cuda()    
    file_id = utils.get_file_id(img_path)
    num_classes = len(labelmap) + 1                      # +1 for background
    net = build_ssd('test', 300, num_classes)            # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    detections = net(x).data
    dets = detections[0, 1, :]
    mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
    dets = torch.masked_select(dets, mask).view(-1, 5)
    boxes = dets[:, 1:]
    boxes[:, 0] *= w
    boxes[:, 2] *= w
    boxes[:, 1] *= h
    boxes[:, 3] *= h
    pred_bboxes_ = boxes  
    f = Image.open(img_path)
    f.convert('RGB')
    img_raw = f.copy()
    for i in range(pred_bboxes_.shape[0]):
        xmin, ymin, xmax, ymax = pred_bboxes_[i,:]
        utils.draw_bounding_box_on_image(img_raw,ymin, xmin, ymax, xmax)
    plt.axis('off')
    plt.imshow(img_raw)
    if SAVE_FLAG == 1:
        plt.savefig(os.path.join(opt.test_output_path, file_id+'.png'), bbox_inches='tight', pad_inches=0)
    else:
        plt.show()  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", type=str, help="test image path")
    parser.add_argument('--trained_model',
                        default='weights/SCUT.pth', type=str,
                        help='Trained state_dict file path to open')
    parser.add_argument('--save_folder', default='eval/', type=str,
                        help='File path to save results')
    parser.add_argument('--confidence_threshold', default=0.01, type=float,
                        help='Detection confidence threshold')
    parser.add_argument('--top_k', default=5, type=int,
                        help='Further restrict the number of predictions to parse')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda to train model')
    parser.add_argument('--scut_root', default=SCUT_ROOT,
                        help='Location of SCUT root directory')
    parser.add_argument('--cleanup', default=True, type=str2bool,
                        help='Cleanup and remove results files following eval')
    args = parser.parse_args()
    if torch.cuda.is_available():
        if args.cuda:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        if not args.cuda:
            print("WARNING: It looks like you have a CUDA device, but aren't using \
                  CUDA.  Run with --cuda for optimal eval speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    detect(args.img_path)
    # model_path = './checkpoints/sess:2/head_detector08120858_0.682282441835'

    # test_data_list_path = os.path.join(opt.data_root_path, 'brainwash_test.idl')
    # test_data_list = utils.get_phase_data_list(test_data_list_path)
    # data_list = []
    # save_idx = 0
    # with open(test_data_list_path, 'rb') as fp:
    #     for line in fp.readlines():
    #         if ":" not in line:
    #             img_path, _ = line.split(";")
    #         else:
    #             img_path, _ = line.split(":")

    #         src_path = os.path.join(opt.data_root_path, img_path.replace('"',''))
    #         detect(src_path, model_path, save_idx)
    #         save_idx += 1



