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


def read_features(patient_list, patient_labels, feature_path_dict):
    bags = []
    labels = []

    filename_list = []
    
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
            feature_list = pkl_file['features']
            filenames = pkl_file['filenames']
        filename_list.append(filenames)

        bags.append(np.array(feature_list))
        labels.append(label)
        
    bags = np.array(bags, dtype=object)
    labels = np.array(labels)
    return bags, labels, filename_list


def load_pkl(pkl_pth):
    with open(pkl_pth, 'rb') as handle:
        file = pickle.load(handle)
    return file

