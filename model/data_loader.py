#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

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

        if self.dataset == 'concept':
            self.read_matdataset_concept()
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


    def read_matdataset_concept(self):

        path= self.datadir + 'res101.mat'
        print('_____')
        print(path)
        pdb.set_trace()
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
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[train_label]


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


    def read_matdataset(self):

        path= self.datadir + 'res101.mat'
        print('_____')
        print(path)
        pdb.set_trace()
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
        self.data['train_seen'][self.auxiliary_data_source] = self.aux_data[train_label]


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

