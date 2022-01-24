#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import sys
import os
from pathlib import Path
import pdb
import pickle
import copy

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
# BabyARC-fewshot dataset for classification:
from reasoning.experiments.concept_energy import get_dataset, ConceptDataset, ConceptFewshotDataset
from reasoning.pytorch_net.util import init_args, plot_matrices, get_device
from reasoning.fsl_baselines.babyarc_eval_fewshot import load_model, get_babyarc_dataloader


# In[ ]:


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label


EMBEDDING_DICT = {
    "c-Line->Eshape": {
        "Line": [
            [1,0,0,0, 0,0,0, 0,0, 0,0],
            [0,1,0,0, 0,0,0, 0,0, 0,0],
            [0,0,1,0, 0,0,0, 0,0, 0,0],
            [0,0,0,1, 0,0,0, 0,0, 0,0],
        ],
        "Parallel": [
            [0,0,0,0, 1,0,0, 0,0, 0,0],
            [0,0,0,0, 0,1,0, 0,0, 0,0],
            [0,0,0,0, 0,0,1, 0,0, 0,0],
        ],
        "VerticalMid": [
            [0,0,0,0, 0,0,0, 1,0, 0,0],
            [0,0,0,0, 0,0,0, 0,1, 0,0],
        ],
        "VerticalEdge": [
            [0,0,0,0, 0,0,0, 0,0, 1,0],
            [0,0,0,0, 0,0,0, 0,0, 0,1],
        ],
        "Eshape": [
            [1,1,1,1, 1,1,1, 1,0, 1,1],
        ],
        "Fshape": [
            [1,1,1,0, 1,0,0, 1,0, 1,0],
        ],
        "Ashape": [
            [1,1,1,1, 1,1,0, 1,1, 1,1],
        ],
    },
}


def get_label_embedding_from_c_label(c_label, mode):
    if mode == "c-Line->Eshape":
        label_dict = {
            "Line": [0,1,2,3],
            "Parallel": [4,5,6],
            "VerticalMid": [7,8],
            "VerticalEdge": [9,10],
            "Eshape": [11],
            "Fshape": [12],
            "Ashape": [13],
        }
    else:
        raise
    c_label_cand = label_dict[c_label]
    c_embedding_cand = EMBEDDING_DICT[mode][c_label]
    idx = np.random.choice(len(c_label_cand))
    c_label = c_label_cand[idx]
    c_embedding = c_embedding_cand[idx]
    return c_label, c_embedding


LABEL_TO_C_LABEL = {
    0: "Line",
    1: "Line",
    2: "Line",
    3: "Line",
    4: "Parallel",
    5: "Parallel",
    6: "Parallel",
    7: "VerticalMid",
    8: "VerticalMid",
    9: "VerticalEdge",
    10: "VerticalEdge",
    11: "Eshape",
    12: "Fshape",
    13: "Ashape",
}


# In[4]:


class DATA_LOADER(object):
    def __init__(self, dataset, aux_datasource, device='cuda'):

        print("The current working directory is")
        print(os.getcwd())
        folder = str(Path(os.getcwd()))
        if folder[-5:] == 'model':
            project_directory = Path(os.getcwd()).parent
        else:
            project_directory = folder

        print('Project Directory:')
        print(project_directory)
        data_path = str(project_directory) + '/data'
        print('Data Path')
        print(data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.device = device
        self.dataset = dataset
        self.auxiliary_data_source = aux_datasource

        self.all_data_sources = ['resnet_features'] + [self.auxiliary_data_source]

        if self.dataset in ['c-Line->Eshape', 'c-Eshape->RectE']:
            self.read_matdataset_concept(mode=self.dataset)
        else:
            if self.dataset == 'CUB':
                self.datadir = self.data_path + '/CUB/'
            elif self.dataset == 'SUN':
                self.datadir = self.data_path + '/SUN/'
            elif self.dataset == 'AWA1':
                self.datadir = self.data_path + '/AWA1/'
            elif self.dataset == 'AWA2':
                self.datadir = self.data_path + '/AWA2/'
            self.read_matdataset()

        self.index_in_epoch = 0
        self.epochs_completed = 0


    def next_batch(self, batch_size):
        #####################################################################
        # gets batch from train_feature = 7057 samples from 150 train classes
        #####################################################################
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.data['train_seen']['resnet_features'][idx]
        batch_label =  self.data['train_seen']['labels'][idx]
        batch_att = self.aux_data[batch_label]
        return batch_label, [ batch_feature, batch_att]


    def read_matdataset_concept(self, mode):
        """
        For the concept dataset, during training, the seen classes are
            "Line", "Parallel", "VerticalMid", "VerticalEdge". Their embeddings are
            [Line,Line,Line,Line, Parallel,Parallel,Parallel, VerticalMid,VerticalMid, VerticalEdge,VerticalEdge]  # starts at 0,4,7,9
            during training, if it is a Line e.g., one of the embedding will activate.

        During inference, for a compositional concept, the embeddings corresponding to the concepts will be activated.
            For example, for Eshape, it will be [1,1,1,1, 1,1,1, 1,0, 1,1]

        """
        if mode == 'c-Line->Eshape':
            concept_args = init_args({
                "dataset": "c-Line",
                "seed": 1,
                "n_examples": 44000,
                "canvas_size": 16,
                "rainbow_prob": 0.,
                "color_avail": "1,2",
                "w_type": "image+mask",
                "max_n_distractors": 2,
                "min_n_distractors": 0,
                "allow_connect": True,
            })
            concept_dataset, _ = get_dataset(concept_args, is_load=True)

            relation_args = init_args({
                "dataset": "c-Parallel+VerticalMid+VerticalEdge",
                "seed": 1,
                "n_examples": 44000,
                "canvas_size": 16,
                "rainbow_prob": 0.,
                "color_avail": "1,2",
                "w_type": "image+mask",
                "max_n_distractors": 3,
                "min_n_distractors": 0,
                "allow_connect": True,
            })
            relation_dataset, _ = get_dataset(relation_args, is_load=True)

            test_args = init_args({
                "dataset": "c-Eshape+Fshape+Ashape",
                "seed": 2,
                "n_examples": 400,
                "canvas_size": 16,
                "rainbow_prob": 0.,
                "w_type": "image+mask",
                "color_avail": "1,2",
                "min_n_distractors": 0,
                "max_n_distractors": 0,
                "allow_connect": True,
                "parsing_check": False,
            })
            test_dataset, _ = get_dataset(test_args, is_load=True)
        else:
            raise

        train_img = []
        train_mask = []
        train_label = []
        train_att = []

        test_seen_img = []
        test_seen_label = []

        test_unseen_img = []
        test_unseen_label = []
        test_unseen_att = []

        for data in concept_dataset:
            img, masks, c_label, _ = data  # img: [10,16,16]
            label, c_embedding = get_label_embedding_from_c_label(c_label, mode=mode)
            train_img.append(img)
            train_label.append(label)
            train_att.append(c_embedding)
            train_mask.append(torch.cat([torch.cat(masks), torch.zeros(masks[0].shape)]))

        for data in relation_dataset:
            img, masks, c_label, _ = data  # img: [10,16,16]
            label, c_embedding = get_label_embedding_from_c_label(c_label, mode=mode)
            train_img.append(img)
            train_label.append(label)
            train_att.append(c_embedding)
            train_mask.append(torch.cat(masks))

        for data in test_dataset:
            img, _, c_label, _ = data  # img: [10,16,16]
            label, c_embedding = get_label_embedding_from_c_label(c_label, mode=mode)
            test_unseen_img.append(img)
            test_unseen_label.append(label)
            test_unseen_att.append(c_embedding)

        train_img = torch.stack(train_img).to("cpu")  # [88000, 10, 16, 16]
        train_label = torch.LongTensor(train_label).to(self.device)  # [88000]
        train_att = torch.FloatTensor(train_att).to(self.device)  # [88000, 11]
        train_mask = torch.stack(train_mask).to(self.device)

        test_unseen_img = torch.stack(test_unseen_img).to("cpu")  # [400, 10, 16, 16]
        test_unseen_label = torch.LongTensor(test_unseen_label).to(self.device)  # [400]
        test_unseen_att = torch.FloatTensor(test_unseen_att).to(self.device)  # [400, 11]

        List = []
        for key, item in EMBEDDING_DICT[mode].items():
            List += item
        self.aux_data = torch.FloatTensor(List).to(self.device)
        
        if mode == 'c-Line->Eshape':
            model_args = init_args({
                'model': 'resnet12_ssl',
                'model_path': '/dfs/user/tailin/.results/fsl_baselines/backup/babyarc_resnet12_ssl_ground_lr_0.005_decay_0.0005_trans_2d_trial_1/model_cosmic-water-212.pth',
                'n_deconv_conv': 0,
                'lst_channels': [64, 160, 320, 640],
                'is_3d': False,
                'use_easy_aug': False,
                'task': 'ground',
                'training_ver': '',
                'fs': 'complex_v2',
                'data_root': '/raid/data/IncrementLearn/imagenet/Datasets/MiniImagenet/',
                'simclr': False,
                'n_aug_support_samples': 5,
                'num_workers': 3,
                'test_batch_size': 1,
                'batch_size': 64,
                'ft_batch_size': 1,
                'ft_epochs': 10,
                'ft_learning_rate': 0.02,
                'ft_weight_decay': 0.0005,
                'ft_momentum': 0.9,
                'ft_adam': False,
                'data_aug': True,
                'n_cls': 7
            })
        else:
            raise
        resnet_model = load_model(model_args).to("cpu")
        train_feature = resnet_model.encode(train_img).detach().to(self.device)
        test_seen_feature = []
        test_unseen_feature = resnet_model.encode(test_unseen_img).detach().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device) # [40]
        self.novelclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)  # [10]
        self.ntrain = train_feature.size()[0]  # 19832
        self.ntrain_class = self.seenclasses.size(0)  # 40
        self.ntest_class = self.novelclasses.size(0)  # 10
        self.train_class = self.seenclasses.clone()  # [40]
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()  #[0,...49]

        self.train_mapped_label = map_label(train_label, self.seenclasses)  # [19832]

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels']= train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[train_label]   # [19832, 85]
        self.data['train_seen']['masks'] = train_mask

        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature  # [4958, 2048]
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[test_unseen_label]  # [5685, 85]
        self.data['test_unseen']['labels'] = test_unseen_label  # [5685]

        self.novelclass_aux_data = self.aux_data[self.novelclasses]  # [3, 11]
        self.seenclass_aux_data = self.aux_data[self.seenclasses] # [11, 11]


    def read_matdataset(self):

        path= self.datadir + 'res101.mat'
        print('_____')
        print(path)
        matcontent = sio.loadmat(path) # keys: 'image_files', 'features', 'labels']
        feature = matcontent['features'].T  # [30475, 2048]
        label = matcontent['labels'].astype(int).squeeze() - 1  # [30475]

        path= self.datadir + 'att_splits.mat'
        matcontent = sio.loadmat(path)
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1  # (19832,)
        train_loc = matcontent['train_loc'].squeeze() - 1 # (16864,) --> train_feature = TRAIN SEEN
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1 #(7926,)--> test_unseen_feature = TEST UNSEEN
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1  # (4958,)
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1  # (5685,)


        if self.auxiliary_data_source == 'attributes':
            self.aux_data = torch.from_numpy(matcontent['att'].T).float().to(self.device)  # [50, 85]
        else:
            if self.dataset != 'CUB':
                print('the specified auxiliary datasource is not available for this dataset')
            else:

                with open(self.datadir + 'CUB_supporting_data.p', 'rb') as h:
                    x = pickle.load(h)
                    self.aux_data = torch.from_numpy(x[self.auxiliary_data_source]).float().to(self.device)


                print('loaded ', self.auxiliary_data_source)


        scaler = preprocessing.MinMaxScaler()

        train_feature = scaler.fit_transform(feature[trainval_loc])  # (19832, 2048)
        test_seen_feature = scaler.transform(feature[test_seen_loc])  # (4958, 2048)
        test_unseen_feature = scaler.transform(feature[test_unseen_loc])  # (5685, 2048)

        train_feature = torch.from_numpy(train_feature).float().to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature).float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature).float().to(self.device)

        train_label = torch.from_numpy(label[trainval_loc]).long().to(self.device)
        test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long().to(self.device)
        test_seen_label = torch.from_numpy(label[test_seen_loc]).long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device) # [40]
        self.novelclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)  # [10]
        self.ntrain = train_feature.size()[0]  # 19832
        self.ntrain_class = self.seenclasses.size(0)  # 40
        self.ntest_class = self.novelclasses.size(0)  # 10
        self.train_class = self.seenclasses.clone()  # [40]
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()  #[0,...49]

        self.train_mapped_label = map_label(train_label, self.seenclasses)  # [19832]

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels']= train_label
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[train_label]  # [19832, 85]


        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen'][self.auxiliary_data_source] = self.aux_data[test_unseen_label]
        self.data['test_unseen']['labels'] = test_unseen_label

        self.novelclass_aux_data = self.aux_data[self.novelclasses]
        self.seenclass_aux_data = self.aux_data[self.seenclasses]


    def transfer_features(self, n, num_queries='num_features'):
        """Only used for few-shot learning."""
        print('size before')
        print(self.data['test_unseen']['resnet_features'].size())
        print(self.data['train_seen']['resnet_features'].size())


        print('o'*100)
        print(self.data['test_unseen'].keys())
        for i,s in enumerate(self.novelclasses):

            features_of_that_class   = self.data['test_unseen']['resnet_features'][self.data['test_unseen']['labels']==s ,:]

            if 'attributes' == self.auxiliary_data_source:
                attributes_of_that_class = self.data['test_unseen']['attributes'][self.data['test_unseen']['labels']==s ,:]
                use_att = True
            else:
                use_att = False
            if 'sentences' == self.auxiliary_data_source:
                sentences_of_that_class = self.data['test_unseen']['sentences'][self.data['test_unseen']['labels']==s ,:]
                use_stc = True
            else:
                use_stc = False
            if 'word2vec' == self.auxiliary_data_source:
                word2vec_of_that_class = self.data['test_unseen']['word2vec'][self.data['test_unseen']['labels']==s ,:]
                use_w2v = True
            else:
                use_w2v = False
            if 'glove' == self.auxiliary_data_source:
                glove_of_that_class = self.data['test_unseen']['glove'][self.data['test_unseen']['labels']==s ,:]
                use_glo = True
            else:
                use_glo = False
            if 'wordnet' == self.auxiliary_data_source:
                wordnet_of_that_class = self.data['test_unseen']['wordnet'][self.data['test_unseen']['labels']==s ,:]
                use_hie = True
            else:
                use_hie = False


            num_features = features_of_that_class.size(0)

            indices = torch.randperm(num_features)

            if num_queries!='num_features':

                indices = indices[:n+num_queries]


            print(features_of_that_class.size())


            if i==0:

                new_train_unseen      = features_of_that_class[   indices[:n] ,:]

                if use_att:
                    new_train_unseen_att  = attributes_of_that_class[ indices[:n] ,:]
                if use_stc:
                    new_train_unseen_stc  = sentences_of_that_class[ indices[:n] ,:]
                if use_w2v:
                    new_train_unseen_w2v  = word2vec_of_that_class[ indices[:n] ,:]
                if use_glo:
                    new_train_unseen_glo  = glove_of_that_class[ indices[:n] ,:]
                if use_hie:
                    new_train_unseen_hie  = wordnet_of_that_class[ indices[:n] ,:]


                new_train_unseen_label  = s.repeat(n)

                new_test_unseen = features_of_that_class[  indices[n:] ,:]

                new_test_unseen_label = s.repeat( len(indices[n:] ))

            else:
                new_train_unseen  = torch.cat(( new_train_unseen             , features_of_that_class[  indices[:n] ,:]),dim=0)
                new_train_unseen_label  = torch.cat(( new_train_unseen_label , s.repeat(n)),dim=0)

                new_test_unseen =  torch.cat(( new_test_unseen,    features_of_that_class[  indices[n:] ,:]),dim=0)
                new_test_unseen_label = torch.cat(( new_test_unseen_label  ,s.repeat( len(indices[n:]) )) ,dim=0)

                if use_att:
                    new_train_unseen_att    = torch.cat(( new_train_unseen_att   , attributes_of_that_class[indices[:n] ,:]),dim=0)
                if use_stc:
                    new_train_unseen_stc    = torch.cat(( new_train_unseen_stc   , sentences_of_that_class[indices[:n] ,:]),dim=0)
                if use_w2v:
                    new_train_unseen_w2v    = torch.cat(( new_train_unseen_w2v   , word2vec_of_that_class[indices[:n] ,:]),dim=0)
                if use_glo:
                    new_train_unseen_glo    = torch.cat(( new_train_unseen_glo   , glove_of_that_class[indices[:n] ,:]),dim=0)
                if use_hie:
                    new_train_unseen_hie    = torch.cat(( new_train_unseen_hie   , wordnet_of_that_class[indices[:n] ,:]),dim=0)



        print('new_test_unseen.size(): ', new_test_unseen.size())
        print('new_test_unseen_label.size(): ', new_test_unseen_label.size())
        print('new_train_unseen.size(): ', new_train_unseen.size())
        #print('new_train_unseen_att.size(): ', new_train_unseen_att.size())
        print('new_train_unseen_label.size(): ', new_train_unseen_label.size())
        print('>> num novel classes: ' + str(len(self.novelclasses)))

        #######
        ##
        #######

        self.data['test_unseen']['resnet_features'] = copy.deepcopy(new_test_unseen)
        #self.data['train_seen']['resnet_features']  = copy.deepcopy(new_train_seen)

        self.data['test_unseen']['labels'] = copy.deepcopy(new_test_unseen_label)
        #self.data['train_seen']['labels']  = copy.deepcopy(new_train_seen_label)

        self.data['train_unseen']['resnet_features'] = copy.deepcopy(new_train_unseen)
        self.data['train_unseen']['labels'] = copy.deepcopy(new_train_unseen_label)
        self.ntrain_unseen = self.data['train_unseen']['resnet_features'].size(0)

        if use_att:
            self.data['train_unseen']['attributes'] = copy.deepcopy(new_train_unseen_att)
        if use_w2v:
            self.data['train_unseen']['word2vec']   = copy.deepcopy(new_train_unseen_w2v)
        if use_stc:
            self.data['train_unseen']['sentences']  = copy.deepcopy(new_train_unseen_stc)
        if use_glo:
            self.data['train_unseen']['glove']      = copy.deepcopy(new_train_unseen_glo)
        if use_hie:
            self.data['train_unseen']['wordnet']   = copy.deepcopy(new_train_unseen_hie)

        ####
        self.data['train_seen_unseen_mixed'] = {}
        self.data['train_seen_unseen_mixed']['resnet_features'] = torch.cat((self.data['train_seen']['resnet_features'],self.data['train_unseen']['resnet_features']),dim=0)
        self.data['train_seen_unseen_mixed']['labels'] = torch.cat((self.data['train_seen']['labels'],self.data['train_unseen']['labels']),dim=0)

        self.ntrain_mixed = self.data['train_seen_unseen_mixed']['resnet_features'].size(0)

        if use_att:
            self.data['train_seen_unseen_mixed']['attributes'] = torch.cat((self.data['train_seen']['attributes'],self.data['train_unseen']['attributes']),dim=0)
        if use_w2v:
            self.data['train_seen_unseen_mixed']['word2vec'] = torch.cat((self.data['train_seen']['word2vec'],self.data['train_unseen']['word2vec']),dim=0)
        if use_stc:
            self.data['train_seen_unseen_mixed']['sentences'] = torch.cat((self.data['train_seen']['sentences'],self.data['train_unseen']['sentences']),dim=0)
        if use_glo:
            self.data['train_seen_unseen_mixed']['glove'] = torch.cat((self.data['train_seen']['glove'],self.data['train_unseen']['glove']),dim=0)
        if use_hie:
            self.data['train_seen_unseen_mixed']['wordnet'] = torch.cat((self.data['train_seen']['wordnet'],self.data['train_unseen']['wordnet']),dim=0)

#d = DATA_LOADER()

