import argparse, os, shutil


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Experiment arguments
parser.add_argument('--split_path', default=f'', help='path to the file which contains the split of train and val.')
parser.add_argument('--split_k', default=3, type=int)
parser.add_argument('--exp_name', default='cervical', help='The name for this experiment')
parser.add_argument('--feature_dict_str', default='')
# Feature dict string is a string that follows this pattern: "dataset_key:path,dataset_key:path,..." when parsing, the code will check if a dataset_key is in the case_id.
parser.add_argument('--output_dir', default=f'output/', help='The output path.') 
parser.add_argument('--exp_id', default=0, type=int) 

# Training arguments
parser.add_argument('--epoch', type=int, default=10, help='The number of epoch.')
parser.add_argument('--lr', type=float, default=3e-3, help='learning rate.')
parser.add_argument('--lr_step', type=int, default=5, help='learning rate steps.')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate.')
parser.add_argument('--eval_step', type=int, default=100, help='batch size (gradient accum).')
parser.add_argument('--margin_inter', type=float, default=0.5, help='Margin between negative sample and positive.')
parser.add_argument('--margin_intra', type=float, default=0.1, help='Margin between negative samples.')
parser.add_argument('--sample_max_num', type=int, default=3000, help='The maximum sample number allowed within a bag given the GPU limitation.')
