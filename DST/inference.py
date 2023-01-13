import os
import pdb
import sys
import json
import torch
import random
import argparse
import operator
import progressbar
import torch.nn as nn
from operator import itemgetter
import torch.nn.functional as F
from dst import ignore_none, default_cleaning, IGNORE_TURNS_TYPE2, paser_bs
from logger_conf import CreateLogger

def get_checkpoint_name(prefix):
    file_names = os.listdir(prefix)
    for name in file_names:
        if name.startswith('epoch'):
            return name

def zip_result(prediction):
    result = {}
    for turn in prediction:
        dial_id = turn['dial_id']
        turn_idx = turn['turn_num']
        try:
            result[dial_id][turn_idx] = turn
        except KeyError:
            result[dial_id] = {}
            result[dial_id][turn_idx] = turn
    return result


def add_dict_type(data):
    clean_tokens = ['<|endoftext|>', ]
    for file_name in data:
        for turn_id, turn_data in data[file_name].items():
            turn_target = turn_data['bspn']
            turn_pred = turn_data['bspn_gen']
            turn_target = paser_bs(turn_target)
            turn_pred = paser_bs(turn_pred)
            # clean
            for bs in turn_pred:
                if bs in clean_tokens + ['', ' '] or bs.split()[-1] == 'none':
                    turn_pred.remove(bs)

            new_turn_pred = []
            for bs in turn_pred:
                for tok in clean_tokens:
                    bs = bs.replace(tok, '').strip()
                    new_turn_pred.append(bs)
            turn_pred = new_turn_pred

            turn_pred, turn_target = ignore_none(turn_pred, turn_target)
            # MultiWOZ default cleaning
            turn_pred, turn_target = default_cleaning(turn_pred, turn_target)
            pred_belief_dict = {}
            for item in turn_pred:
                domain = item.split(' ')[0].replace("[","").replace(']',"").strip()
                slot = item.split(' ')[1].strip()
                value = ' '.join(item.split(' ')[2:]).strip()
                pred_belief_dict[f'{domain}-{slot}'] = value
            turn_data['pred_belief_dict'] = pred_belief_dict

def add_sys(data):
    clean_tokens = ['<|endoftext|>', ]
    for file_name in data:
        for turn_id, turn_data in data[file_name].items():
            turn_target = turn_data['bspn']
            turn_pred = turn_data['bspn_gen']
            turn_target = paser_bs(turn_target)
            turn_pred = paser_bs(turn_pred)
            # clean
            for bs in turn_pred:
                if bs in clean_tokens + ['', ' '] or bs.split()[-1] == 'none':
                    turn_pred.remove(bs)

            new_turn_pred = []
            for bs in turn_pred:
                for tok in clean_tokens:
                    bs = bs.replace(tok, '').strip()
                    new_turn_pred.append(bs)
            turn_pred = new_turn_pred

            turn_pred, turn_target = ignore_none(turn_pred, turn_target)
            # MultiWOZ default cleaning
            turn_pred, turn_target = default_cleaning(turn_pred, turn_target)
            pred_belief_dict = {}
            for item in turn_pred:
                domain = item.split(' ')[0].replace("[","").replace(']',"").strip()
                slot = item.split(' ')[1].strip()
                value = ' '.join(item.split(' ')[2:]).strip()
                pred_belief_dict[f'{domain}-{slot}'] = value
            turn_data['pred_belief_dict'] = pred_belief_dict
        
def parse_config():
    parser = argparse.ArgumentParser()
    # dataset configuration
    parser.add_argument('--train_path', type=str, help='The path where the data stores.')
    parser.add_argument('--dev_path', type=str, help='The path where the data stores.')
    parser.add_argument('--test_path', type=str, help='The path where the data stores.')
    parser.add_argument('--shuffle_mode', type=str, default='shuffle_session_level', 
        help="shuffle_session_level or shuffle_turn_level, it controls how we shuffle the training data.")
    parser.add_argument('--add_prefix', type=str, default='True', 
        help="True or False, whether we add prefix when we construct the input sequence.")
    parser.add_argument('--add_special_decoder_token', default='True', type=str, help='Whether we discriminate the decoder start and end token for different tasks.')
    # model configuration
    parser.add_argument('--model_name', type=str, help='t5-small or t5-base or t5-large')
    parser.add_argument('--pretrained_path', type=str, help='the path that stores pretrained checkpoint.')
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help='Batch size for each gpu.')  
    parser.add_argument("--number_of_gpu", type=int, default=8, help="Number of available GPUs.")  
    parser.add_argument("--output_save_path", type=str, help="directory to save the model output.")
    
    return parser.parse_args()

import argparse
if __name__ == '__main__':
    logger = CreateLogger('inference')
    if torch.cuda.is_available():
        logger.info('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            logger.info('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            logger.info('Using single GPU training.')
    else:
        pass
 
    args = parse_config()
    
    logger.info(args)

    device = torch.device('cuda')

    logger.info('Start loading data...')
    assert args.model_name.startswith('t5')

    ckpt_name = get_checkpoint_name(args.pretrained_path)
    pretrained_path = args.pretrained_path + '/' + ckpt_name

    from transformers import T5Tokenizer
    logger.info('Loading Pretrained Tokenizer...')
    tokenizer = T5Tokenizer.from_pretrained(pretrained_path)

    if args.add_prefix == 'True':
        add_prefix = True
    elif args.add_prefix == 'False':
        add_prefix = False
    else:
        raise Exception('Wrong Prefix Mode!!!')

    if args.add_special_decoder_token == 'True':
        add_special_decoder_token = True
    elif args.add_special_decoder_token == 'False':
        add_special_decoder_token = False
    else:
        raise Exception('Wrong Add Special Token Mode!!!')

    from dataclass import DSTMultiWozData
    data = DSTMultiWozData('_', args.model_name, tokenizer, args.train_path, args.dev_path, args.test_path, shuffle_mode=args.shuffle_mode, 
                          data_mode='train', train_data_ratio=0.005)

    logger.info('Start loading model...')
    assert args.model_name.startswith('t5')

    from modelling.T5Model import T5Gen_Model
    model = T5Gen_Model(pretrained_path, data.tokenizer, data.special_token_list, dropout=0.0, 
        add_special_decoder_token=add_special_decoder_token, is_training=True)

    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    model.eval()
    logger.info('Model loaded')

    # **********************************************************************
    # --- evaluation --- #
    from inference_utlis import batch_generate
    logger.info('Start evaluation...')
    with torch.no_grad():
        dev_batch_list = \
        data.build_all_evaluation_batch_list(eva_batch_size=args.number_of_gpu * args.batch_size_per_gpu, eva_mode='test')
        dev_batch_num_per_epoch = len(dev_batch_list)

        logger.info('Number of evaluation batches is %d' % dev_batch_num_per_epoch)
        all_dev_result = []
        for p_dev_idx in range(dev_batch_num_per_epoch):
            if p_dev_idx % 50 ==0:
                logger.info(f"Testing : {p_dev_idx}/{deb_batch_num_per_epoch}")
            one_inference_batch = dev_batch_list[p_dev_idx]
            dev_batch_parse_dict = batch_generate(model, one_inference_batch, data)
            for item in dev_batch_parse_dict:
                all_dev_result.append(item)
        from compute_joint_acc import compute_jacc
        all_dev_result = zip_result(all_dev_result)
        add_dict_type(all_dev_result)
        add_sys(all_dev_result)
        dev_score = compute_jacc(data=all_dev_result) * 100
        one_dev_str = 'test_joint_accuracy_{}'.format(round(dev_score,2))

        logger.info('Test Accuracy is {}'.format(dev_score))
        output_save_path = args.output_save_path + '/' + one_dev_str + '.json'
        import os
        if os.path.exists(args.output_save_path):
            pass
        else: # recursively construct directory
            os.makedirs(args.output_save_path, exist_ok=True)

        import json
        with open(output_save_path, 'w') as outfile:
            json.dump(all_dev_result, outfile, indent=4)
    logger.info('Evaluation Completed!')
