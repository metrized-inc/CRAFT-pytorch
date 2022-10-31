import sys
import os
import time
import argparse
import string

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

import cv2
from skimage import io
import numpy as np
import json
import zipfile
import pandas as pd
from collections import OrderedDict

from craft_utils.craft import CRAFT
import craft_utils.test_net as test_net
import craft_utils.imgproc as imgproc
import craft_utils.file_utils as file_utils
import craft_utils.craft_model_utils as craft_utils

from deep_text_recognition_utils.deep_text_recognition import run_str

def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")

def crop_images(image_folder, output_csv, cropped_words_output):
    data = pd.read_csv(output_csv)

    for image_num in range(data.shape[0]):
        image = cv2.imread(os.path.join(image_folder, data['image_name'][image_num]))
        image_name = data['image_name'][image_num].strip('.jpg')
        score_bbox = data['word_bboxes'][image_num].split('),')
        craft_utils.generate_words(image_name, score_bbox, image, cropped_words_output)


def main(args):
    # Create folders
    output_dir = args.output_folder
    output_csv = os.path.join(output_dir, 'data.csv')
    
    craft_result_output = os.path.join(output_dir, 'results')
    craft_mask_output = os.path.join(output_dir, 'masks')
    cropped_words_output = os.path.join(output_dir, 'cropped_words')
    if not os.path.exists(craft_result_output):
        os.makedirs(craft_result_output)
    if not os.path.exists(craft_mask_output):
        os.makedirs(craft_mask_output)
    if not os.path.exists(cropped_words_output):
        os.makedirs(cropped_words_output)


    # CUSTOMISE START
    start = args.input_folder

    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(start)
    image_names = []
    for num in range(len(image_list)):
        image_names.append(os.path.relpath(image_list[num], start))

    # Create dataframe to save bbox data
    data = pd.DataFrame(columns=['image_name', 'word_bboxes', 'pred_words', 'align_text'])
    data['image_name'] = image_names


    # load net
    net = CRAFT() # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(test_net.copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(test_net.copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()


    # LinkRefiner
    refine_net = None
    if args.refine:
        from craft_utils.refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(test_net.copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(test_net.copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()


    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text, det_scores = test_net.test_net(args, net, image, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, refine_net)
        
        bbox_score = {}
        
        for i, bbox in enumerate(bboxes):
            key = str(det_scores[i])
            bbox_score[key] = bbox

        data['word_bboxes'][k] = bbox_score

        # save score text
        filename, file_ext = os.path.splitext(os.path.basename(image_path))
        mask_file = craft_mask_output + "/res_" + filename + '_mask.jpg'
        cv2.imwrite(mask_file, score_text)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname = craft_result_output)

    data.to_csv(output_csv, sep = ',', na_rep='Unknown')
    print("elapsed time : {}s".format(time.time() - t))

    # Crop images 
    crop_images(start, output_csv, cropped_words_output)

    # Deep Scene Text Recognition
    run_str(args, cropped_words_output)

if __name__ == '__main__':

    # CRAFT
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--input_folder', default='data/', type=str, help='folder path to input images')
    parser.add_argument('--output_folder', default='output', type=str, help='output folder')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    # 4 Stage Deep Scene Recognition
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    args = parser.parse_args()

    """ STR vocab / character number configuration """
    if args.sensitive:
        args.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    args.num_gpu = torch.cuda.device_count()


    main(args)