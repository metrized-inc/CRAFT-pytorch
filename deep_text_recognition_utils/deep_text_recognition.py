import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from deep_text_recognition_utils.str_utils import CTCLabelConverter, AttnLabelConverter
from deep_text_recognition_utils.dataset import RawDataset, AlignCollate
from deep_text_recognition_utils.model import Model

import pandas as pd
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def run_str(args, cropped_words_output):

    # """Open csv file wherein you are going to write the Predicted Words"""
    # data = pd.read_csv(r'E:\Metrized\CRAFT-pytorch\content\Pipeline\data.csv')

    """ model configuration """
    if 'CTC' in args.Prediction:
        converter = CTCLabelConverter(args.character)
    else:
        converter = AttnLabelConverter(args.character)
    args.num_class = len(converter.character)

    if args.rgb:
        args.input_channel = 3
    model = Model(args)
    print('model input parameters', args.imgH, args.imgW, args.num_fiducial, args.input_channel, args.output_channel,
          args.hidden_size, args.num_class, args.batch_max_length, args.Transformation, args.FeatureExtraction,
          args.SequenceModeling, args.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % args.saved_model)
    model.load_state_dict(torch.load(args.saved_model, map_location=device))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=args.imgH, imgW=args.imgW, keep_ratio_with_pad=args.PAD)

    demo_data = RawDataset(root=cropped_words_output, opt=args)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([args.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, args.batch_max_length + 1).fill_(0).to(device)

            if 'CTC' in args.Prediction:
                preds = model(image, text_for_pred)

                # Select max probabilty (greedy decoding) then decode index to character
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                _, preds_index = preds.max(2)
                # preds_index = preds_index.view(-1)
                preds_str = converter.decode(preds_index.data, preds_size.data)

            else:
                preds = model(image, text_for_pred, is_train=False)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds.max(2)
                preds_str = converter.decode(preds_index, length_for_pred)

            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t {"predicted_labels":25s}\t confidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}')
            # log.write(f'{dashed_line}\n{head}\n{dashed_line}\n')

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                
                start = cropped_words_output
                path = os.path.relpath(img_name, start)

                folder = os.path.dirname(path)

                image_name=os.path.basename(path)

                file_name='_'.join(image_name.split('_')[:-8])

                txt_file=os.path.join(start, folder, file_name)                
                
                log = open(f'{txt_file}_log_demo_result_vgg.txt', 'a')
                if 'Attn' in args.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                print(f'{image_name:25s}\t {pred:25s}\t {confidence_score:0.4f}')
                log.write(f'{image_name:25s}\t {pred:25s}\t {confidence_score:0.4f}\n')

            log.close()