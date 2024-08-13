from torch.utils.data import Dataset
import random
import numpy as np
import os, pickle
from tqdm import tqdm
from torch.autograd import Variable
import torch
from main_train import get_scores


class MyDataset(Dataset):
    def __init__(self, bags, isTest=False, sample_max_num=-1):
        self.bags = bags
        self.isTest = isTest
        self.sample_max_num = sample_max_num

    def __getitem__(self, index):
        examples = self.bags[index]
        if not self.isTest and self.sample_max_num > 0:
            examples = list(examples)
            random.shuffle(examples)
            examples = examples[:self.sample_max_num]
            examples = np.array(examples)
        return examples

    def __len__(self):
        return len(self.bags)



class MyTripletDataset(Dataset):
    def __init__(self, bags, labels, model, args, sample_max_num=-1):
        self.bags = bags
        self.labels = labels
        self.sample_max_num = sample_max_num

        self.model = model
        self.args = args

        neg_indices = [i for i, label in enumerate(labels) if label == 0]
        pos_indices = [i for i, label in enumerate(labels) if label == 1]

        self.triplet_pairs = {}
        for pos_idx in pos_indices:
            for n_i, neg_idx in enumerate(neg_indices):
                for a_i in range(n_i + 1, len(neg_indices)):
                    anc_idx = neg_indices[a_i]
                    if neg_indices == anc_idx:
                        continue
                    unique_id = f'{pos_idx}_{neg_idx}'
                    if unique_id not in self.triplet_pairs:
                        self.triplet_pairs[unique_id] = []
                    self.triplet_pairs[unique_id].append({
                        'pos': pos_idx,
                        'neg': neg_idx,
                        'anc': anc_idx
                    })
        
        self.__triplet_selection__()

    def __triplet_selection__(self, ):
        self.model.eval()
        score_dict = {}
        selected_triplets = []
        for idx, bag in tqdm(enumerate(self.bags), total=len(self.bags), desc='Generating scores'):
            bag = torch.from_numpy(bag)
            bag = bag.float()
            bag = Variable(bag, requires_grad=True).type(torch.cuda.FloatTensor)
            score = get_scores(self.model.forward(bag))
            score_dict[idx] = float(score)
        
        for idx, unique_id in tqdm(enumerate(self.triplet_pairs), total=len(self.triplet_pairs), desc='Triplet selection'):
            current_anc_list = []
            for u_idx, triplet_item in enumerate(self.triplet_pairs[unique_id]):
                n_score = score_dict[triplet_item['neg']]
                p_score = score_dict[triplet_item['pos']]
                anc_score = score_dict[triplet_item['anc']]

                loss_p_n = np.max([0.0, (n_score - p_score + self.args.margin_inter)])
                loss_p_anc = np.max([0.0, (anc_score - p_score + self.args.margin_inter)])
                loss_n_anc = np.max([0.0, (np.linalg.norm(anc_score - n_score) - self.args.margin_intra)])
                loss = loss_p_n + loss_p_anc + loss_n_anc
                if loss != 0:
                    current_anc_list.append(triplet_item)
            if len(current_anc_list) > 0:
                random.shuffle(current_anc_list)
                selected_triplets.append(current_anc_list[0])
        print('Selected triplets:', len(selected_triplets))
        self.triplet_pairs = selected_triplets

    def __select_patches_(self, examples):
        examples = list(examples)
        random.shuffle(examples)
        examples = examples[:self.sample_max_num]
        examples = np.array(examples)
        return examples

    def __getitem__(self, index):
        triplet_item = self.triplet_pairs[index]

        pos_example = self.bags[triplet_item['pos']]
        neg_example = self.bags[triplet_item['neg']]
        anc_example = self.bags[triplet_item['anc']]

        if self.sample_max_num > 0:
            pos_example = self.__select_patches_(pos_example)
            neg_example = self.__select_patches_(neg_example)
            anc_example = self.__select_patches_(anc_example)
        
        return {
            'pos': pos_example,
            'neg': neg_example,
            'anc': anc_example
        }

    def __len__(self):
        return len(self.triplet_pairs)


def read_features(patient_list, patient_labels, feature_path_dict, comb_key):
    bags = []
    labels = []

    filename_list = []
    
    cnt0 = 0
    cnt1 = 0
    
    for idx, patient_id in tqdm(enumerate(patient_list), total=len(patient_list)):
        label = patient_labels[patient_id]

        for feature_key in feature_path_dict:
            if feature_key in patient_id:
                fold_pt_dir = feature_path_dict[feature_key]
                break

        pkl_path = os.path.join(fold_pt_dir, patient_id + '.pkl')
        if not os.path.exists(pkl_path):
            print(pkl_path, 'does not exist!!!')
            continue
        with open(pkl_path, 'rb') as handle:
            pkl_file = pickle.load(handle)
            feature_dict = pkl_file['feature_dict']
            filenames = pkl_file['filenames']
        filename_list.append(filenames)

        feat_list = get_combined_feature(feat_dict=feature_dict, comb_key=comb_key)
        bags.append(np.array(feat_list))
        labels.append(label)
        
    bags = np.array(bags, dtype=object)
    labels = np.array(labels)
    return bags, labels, filename_list


def load_pkl(pkl_pth):
    with open(pkl_pth, 'rb') as handle:
        file = pickle.load(handle)
    return file


feat_comb_dict = {
    '5': ['5'], 
    '11': ['11'], 
    'cls_0': ['cls_0'], 
    'cls_0_avg': ['cls_0_avg'], 
    'cls_1': ['cls_1'],
    'cls_1_avg': ['cls_1_avg'],

    '5_11_concat': ['5', '11'],

    'cls0_cls1_concat': ['cls_0', 'cls_1'],

    'cls_block0_concat': ['cls_0', 'cls_0_avg'],
    'cls_block1_concat': ['cls_1', 'cls_1_avg'],

    'cls_block_all_concat': ['cls_0', 'cls_0_avg', 'cls_1', 'cls_1_avg'],

    'all_concat': ['5', '11', 'cls_0', 'cls_0_avg', 'cls_1', 'cls_1_avg'],


    # owkin/phiken
    'cls_11': ['cls_11'],
    'cls_11_avg': ['cls_11_avg'],
    'cls_block5_concat': ['cls_5', 'cls_5_avg'],
    'cls_block11_concat': ['cls_11', 'cls_11_avg'],
}


def get_combined_feature(feat_dict, comb_key):
    if comb_key not in feat_comb_dict:
        comb_list = [comb_key]
    else:
        comb_list = feat_comb_dict[comb_key]
    feat_list = []
    patch_len = len(feat_dict[comb_list[0]])
    for patch_id in range(patch_len):
        patch_feat_lst = []
        for feat_key in comb_list:
            assert feat_key in feat_dict
            patch_feat_lst.append(feat_dict[feat_key][patch_id])
        concat_feat = np.concatenate(patch_feat_lst, axis=0)
        feat_list.append(concat_feat)
    return feat_list

