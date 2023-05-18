import glob
import numpy
import random
from functools import reduce
import matplotlib.pyplot as plt
from operator import concat
from torchvision import transforms as TR
from torchvision import datasets, transforms
#from augmentations import *
import torchvision
from tabulate import tabulate
import numpy as np
import torch
import yaml
from tqdm import tqdm
import importlib
import torch.nn.functional as F
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Experiment(object):

    def __init__(self, config_file):


        self.config_file = config_file
        with open(config_file) as cf_file:
            self.config = yaml.safe_load( cf_file.read())

        self.label_words = self.config['stage_1']['num_words']-1
        self.label_objects = self.label_words + self.config['stage_1']['num_objects']
        self.label_faces = self.label_objects + self.config['stage_1']['num_faces']
        self.label_expertise = self.label_faces + self.config['stage_2']['expert_dataset']['num_classes']
        self.expert_category = self.config['stage_2']['expert_dataset']['name']
        self.num_words = int(self.config['stage_1']['num_words'])
        self.num_faces = int(self.config['stage_1']['num_faces'])
        self.num_objects = int(self.config['stage_1']['num_objects'])

        print("inovked utils")

    def visualize_example_batch(self, dataloader, batch_size):
        '''
        Visualize example images from the first batch of the passed in dataloader
        '''
        examples = enumerate(dataloader)
        batch_idx, (data, targets) = next(examples)
        fig = plt.figure(figsize=(15, 15))
        for i in range(64):
            plt.subplot(8,8,i+1)
            # plt.tight_layout()
            plt.imshow(data[i].permute(1,2,0), interpolation='none')
            plt.title("ground truth: {}".format(targets[i]))
            plt.xticks([])
            plt.yticks([])


    def __visualize_gates(self, cumulative_gates):

        means = []
        for k in cumulative_gates.keys():
            means.append(np.mean(cumulative_gates[k],axis=0))

        table = [["Dataset","Gate1","Gate2","Gate3"],
                 ["Words",means[0][0],means[0][1],means[0][2]],
                 ["Objects",means[1][0],means[1][1],means[1][2]],
                 ["Faces",means[2][0],means[2][1],means[2][2]]]

        if len(cumulative_gates.keys()) == 4:
            table = [["Dataset","Gate1","Gate2","Gate3"],
                 ["Words",means[0][0],means[0][1],means[0][2]],
                 ["Objects",means[1][0],means[1][1],means[1][2]],
                 ["Faces",means[2][0],means[2][1],means[2][2]],
                 ["Expert Dataset",means[3][0],means[3][1],means[3][2]] ]
        print(tabulate(table))

    def __gate_values(self, gate, labels, expertise_task=False):
        dataset_1 = gate[labels<=self.label_words].detach().cpu().numpy()
        dataset_2 = gate[(self.label_words<labels) & (labels<=self.label_objects)].detach().cpu().numpy()
        dataset_3 = gate[(self.label_objects<labels) & (labels<=self.label_faces)].detach().cpu().numpy()
        current_gates = {"dataset1":dataset_1,"dataset2":dataset_2,"dataset3":dataset_3}

        if expertise_task:
            dataset_4 = gate[self.label_faces<labels].detach().cpu().numpy()
            current_gates = {"dataset1":dataset_1,"dataset2":dataset_2,"dataset3":dataset_3,"dataset4":dataset_4}
        return current_gates

    def __append_values(self, current_gates, cumulative_gates):
        for k in cumulative_gates.keys():
            cumulative_gates[k] = np.concatenate((cumulative_gates[k],current_gates[k]),axis=0)
        return cumulative_gates

    def __cv_loss(self, gate):
        x = gate.sum(0)
        return x.float().var()/(x.float().mean()**2 + 1e-10)


    def __get_accuracy(self, pred, y, accuracies, expertise_task=False):

        words_ind = (y<=self.label_words)
        objects_ind = ((y>self.label_words) & (y<=self.label_objects))
        faces_ind = ((y>self.label_objects) & (y<=self.label_faces))

        correct_words = (pred[words_ind] == y[words_ind]).sum()
        correct_objects = (pred[objects_ind] == y[objects_ind]).sum()
        correct_faces = (pred[faces_ind] == y[faces_ind]).sum()

        accuracies['words'][0] += correct_words
        accuracies['objects'][0] += correct_objects
        accuracies['faces'][0] += correct_faces

        accuracies['words'][1] += words_ind.sum()
        accuracies['objects'][1] += objects_ind.sum()
        accuracies['faces'][1] += faces_ind.sum()

        if expertise_task:
            expert_ind = (y > self.label_faces)
            correct_expert = (pred[expert_ind] == y[expert_ind]).sum()
            accuracies['Expert Dataset'][0] += correct_expert
            accuracies['Expert Dataset'][1] += expert_ind.sum()

    def __print_accuracies(self, accuracies, expertise_task=False, acc_type='Accuracies'):
        acc_mnist = accuracies['words'][0]/accuracies['words'][1]
        acc_cifar = accuracies['objects'][0]/accuracies['objects'][1]
        acc_faces = accuracies['faces'][0]/accuracies['faces'][1]
        num = accuracies['words'][0] + accuracies['objects'][0] + accuracies['faces'][0]
        den = accuracies['words'][1] + accuracies['objects'][1] + accuracies['faces'][1]
        acc_total = num/den

        if expertise_task:
            acc_expert = accuracies['Expert Dataset'][0]/accuracies['Expert Dataset'][1]
            num += accuracies['Expert Dataset'][0]
            den += accuracies['Expert Dataset'][1]
            acc_total = num/den
            print(acc_type,"- WORDS:%f OBJECTS:%f FACES:%f %s:%f TOTAL:%f"%(acc_mnist,acc_cifar,acc_faces,self.expert_category,
                                                                                acc_expert,acc_total))
            return

        print(acc_type,"- WORDS:%f OBJECTS:%f FACES:%f TOTAL:%f"%(acc_mnist,acc_cifar,acc_faces,acc_total))


    def __get_sublevel_acc(self, preds, y, accuracies, expertise_task=False):
        preds = torch.where(preds<=self.label_words,0.0,preds.double())
        preds = torch.where((self.label_words<preds) & (preds<=self.label_objects),1.0,preds.double())
        preds = torch.where((self.label_objects<preds) & (preds<=self.label_faces),2.0,preds.double())

        y = torch.where(y<=self.label_words,0.0,y.double())
        y = torch.where((self.label_words<y) & (y<=self.label_objects),1.0,y.double())
        y = torch.where((self.label_objects<y) & (y<=self.label_faces),2.0,y.double())

        correct_mnist = ((preds==0.0) & (y==0.0)).sum()
        correct_cifar = ((preds==1.0) & (y==1.0)).sum()
        correct_faces = ((preds==2.0) & (y==2.0)).sum()

        accuracies['words'][0] += correct_mnist
        accuracies['objects'][0] += correct_cifar
        accuracies['faces'][0] += correct_faces

        accuracies['words'][1] += (y==0.0).sum()
        accuracies['objects'][1] += (y==1.0).sum()
        accuracies['faces'][1] += (y==2.0).sum()

        if expertise_task:
            preds = torch.where(self.label_faces<preds,3.0,preds.double())
            y = torch.where(self.label_faces<y,3.0,y.double())
            correct_expertise = ((preds==3.0) & (y==3.0)).sum()
            accuracies['Expert Dataset'][0] += correct_expertise
            accuracies['Expert Dataset'][1] += (y==3.0).sum()

    def __validation(self, model, dataloader):
        model.eval()

        num_correct = 0
        num_samples = 0
        val_accuracies = {'words':[0,0],'objects':[0,0],'faces':[0,0]}
        with torch.no_grad():

            for t, (x, y) in enumerate(dataloader):
                x = x.to(device)
                y = y.to(device)
                gate1,gate2,scores = model(x)

                _, preds = scores.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)
                self.__get_accuracy(preds, y, val_accuracies)

            overall_val_acc = float(num_correct) / num_samples
            return overall_val_acc,val_accuracies

    def training(self, dataloader_train, dataloader_val, batch_multiplier = 10):
        model_file = 'architectures.' + self.config['stage_1']['model_file']
        module_network = importlib.import_module(model_file)
        model = module_network.Expert_network(self.config_file)
        model = model.to(device)

        epochs = self.config['stage_1']['training']['epochs']
        learning_rate = self.config['stage_1']['training']['learning_rate']
        num_correct = 0
        num_samples = 0
        initialization = np.zeros((1,3))
        initial_weight = self.config['stage_1']['training']['initial_cv_weight']
        final_weight = self.config['stage_1']['training']['final_cv_weight']

        loss_weights = self.config['stage_1']['training']['class_weights']
        class_weight = torch.cat((torch.tensor([loss_weights[0] for i in range(self.num_words)]),
                                  torch.tensor([loss_weights[1] for i in range(self.num_objects)]),
                                  torch.tensor([loss_weights[2] for i in range(self.num_faces)])),dim=0).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weight, reduction='mean')

        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-3)
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)

        cur_accuracy = 0.0
        for e in tqdm(range(epochs)):

            weight_cv_loss = (initial_weight-final_weight)*(1-(e/epochs))+final_weight

            cumulative_gates1 = {"dataset1":initialization,"dataset2":initialization,"dataset3":initialization}
            cumulative_gates2 = cumulative_gates1.copy()

            accuracies = {'words':[0,0],'objects':[0,0],'faces':[0,0]}
            sub_accuracies = {'words':[0,0],'objects':[0,0],'faces':[0,0]}

            c_loss = []
            v_loss = []
            count = batch_multiplier

            for t, (x, y) in enumerate(dataloader_train):
                x = x.to(device)
                #y = y.type(torch.LongTensor)
                y = y.to(device)
                
                if count == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier
                
                #gate1,gate2,scores = model(x,temp1,temp2)
                gate1,gate2,scores = model(x)

                current_gates1 = self.__gate_values(gate1, y)
                current_gates2 = self.__gate_values(gate2, y)
                cumulative_gates1 =  self.__append_values(current_gates1, cumulative_gates1)
                cumulative_gates2 =  self.__append_values(current_gates2, cumulative_gates2)

                variance_loss = self.__cv_loss(gate1) + self.__cv_loss(gate2)
                classification_loss = loss_fn(scores, y)
                loss = (classification_loss + weight_cv_loss*variance_loss) / batch_multiplier
                c_loss.append(classification_loss.item())
                v_loss.append(variance_loss.item())

                loss.backward()
                
                count -= 1

                _, preds = scores.max(1)
                self.__get_accuracy(preds, y, accuracies)
                self.__get_sublevel_acc(preds, y, sub_accuracies)

                '''
                for p in model.parameters():
                    p.data.clamp_(-0.5, 0.5)
                '''


            print(" ------epoch:%d------\n"%(e))
            print("-------gate layer:%d-------"%(1))
            self.__visualize_gates(cumulative_gates1)
            print("-------gate layer:%d-------"%(2))
            self.__visualize_gates(cumulative_gates2)
            self.__print_accuracies(accuracies)
            self.__print_accuracies(sub_accuracies, acc_type='Sub Accuracies')
            print("Closs:",np.mean(c_loss),"Vloss",np.mean(v_loss))
            print("-------Validation-------")
            overall_val_acc,val_accuracies = self.__validation(model, dataloader_val)
            print("Overall Validation Accuracy",overall_val_acc)
            self.__print_accuracies(val_accuracies)
            print("----------------------------")
            #torch.save({'weights': model.state_dict()},'model_7_weight_annealing.pt')


            if overall_val_acc>cur_accuracy:
                cur_accuracy = overall_val_acc
                torch.save({'weights': model.state_dict(),'config':self.config},'saved_models/'+self.config['stage_1']['model_save_name'])

            print("\n")

    def training_stage2(self, dataloader_train, dataloader_val):



        # model_file = 'architectures.' + self.config['stage_2']['model_file']
        # module_network = importlib.import_module(model_file)
        # # model = module_network.stage2_model
        # # model = model.to(device)

        model_file = self.config['stage_2']['model_file']
        module_network = importlib.import_module('architectures.' + model_file)
        # stage_2_set_up = module_network.Expert_Network_stage_2(self.config_file)
        module_network = getattr(module_network, model_file)
        stage_2_set_up = module_network(self.config_file)
        model = stage_2_set_up.get_stage_2_model()
        model = model.to(device)

        epochs = self.config['stage_2']['training']['epochs']
        learning_rate = self.config['stage_2']['training']['learning_rate']
        initial_temp = self.config['stage_2']['training']['initial_temp']
        final_temp = self.config['stage_2']['training']['final_temp']

        num_correct = 0
        num_samples = 0

        initialization = np.zeros((1,3))

        num_expert_classes = self.config['stage_2']['expert_dataset']['num_classes']

        loss_weights = self.config['stage_2']['training']['class_weights']
        class_weight = torch.cat((torch.tensor([loss_weights[0] for i in range(self.num_words)]),
                                  torch.tensor([loss_weights[1] for i in range(self.num_words)]),
                                  torch.tensor([loss_weights[2] for i in range(self.num_words)]),
                                  torch.tensor([loss_weights[3] for i in range(num_expert_classes)])),dim=0).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weight, reduction='mean')

        optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=1e-4)
        #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
        cur_accuracy = 0.0
        for e in tqdm(range(epochs)):
            model.train()

            temp = (initial_temp-final_temp)*(1-(e/epochs))+final_temp

            cumulative_gates1 = {"dataset1":initialization,"dataset2":initialization,
                                 "dataset3":initialization,"dataset4":initialization}
            cumulative_gates2 = cumulative_gates1.copy()

            accuracies = {'words':[0,0],'objects':[0,0],'faces':[0,0],'Expert Dataset':[0,0]}
            sub_accuracies = {'words':[0,0],'objects':[0,0],'faces':[0,0],'Expert Dataset':[0,0]}

            for t, (x, y) in enumerate(dataloader_train):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                gate1,gate2,scores = model(x,temperature=temp)

                current_gates1 = self.__gate_values(gate1, y, expertise_task=True)
                current_gates2 = self.__gate_values(gate2, y, expertise_task=True)
                cumulative_gates1 =  self.__append_values(current_gates1, cumulative_gates1)
                cumulative_gates2 =  self.__append_values(current_gates2, cumulative_gates2)
                loss = loss_fn(scores, y)

                loss.backward()
                optimizer.step()


                _, preds = scores.max(1)
                self.__get_accuracy(preds, y, accuracies, expertise_task=True)
                self.__get_sublevel_acc(preds, y, sub_accuracies, expertise_task=True)


            print(" ------epoch:%d------\n"%(e))
            print("-------gate layer:%d-------"%(1))
            self.__visualize_gates(cumulative_gates1)
            print("-------gate layer:%d-------"%(2))
            self.__visualize_gates(cumulative_gates2)
            self.__print_accuracies(accuracies, expertise_task=True)
            self.__print_accuracies(sub_accuracies, expertise_task=True, acc_type='Sub Accuracies')
            valid_accuracy = self.__validation(model, dataloader_val)
            print("Validation Accuracy",valid_accuracy)


            if valid_accuracy>cur_accuracy:
                cur_accuracy = valid_accuracy
                torch.save({'weights': model.state_dict(),'config':self.config},'saved_models/'+self.config['stage_2']['model_save_name'])
            print("\n")

    def test(self, dataloader_test):

        #Getting model architecture from config
        model_file = 'architectures.' + self.config['stage_1']['model_file']
        module_network = importlib.import_module(model_file)
        model = module_network.Expert_network(self.config_file)
        model = model.to(device)

        #loading model
        model.load_state_dict(torch.load('saved_models/'+self.config['stage_1']['model_save_name'],map_location=device.type)['weights'])
        model.eval()

        initialization = np.zeros((1,3))



        with torch.no_grad():

            cumulative_gates1 = {"dataset1":initialization,"dataset2":initialization,"dataset3":initialization}
            cumulative_gates2 = cumulative_gates1.copy()
            accuracies = {'words':[0,0],'objects':[0,0],'faces':[0,0]}
            sub_accuracies = {'words':[0,0],'objects':[0,0],'faces':[0,0]}

            for t, (x, y) in enumerate(dataloader_test):
                x = x.to(device)
                y = y.to(device)
                gate1,gate2,scores = model(x)

                current_gates1 = self.__gate_values(gate1, y)
                current_gates2 = self.__gate_values(gate2, y)
                cumulative_gates1 =  self.__append_values(current_gates1, cumulative_gates1)
                cumulative_gates2 =  self.__append_values(current_gates2, cumulative_gates2)

                _, preds = scores.max(1)
                self.__get_accuracy(preds, y, accuracies)
                self.__get_sublevel_acc(preds, y, sub_accuracies)

            print("-------gate layer:%d-------"%(1))
            self.__visualize_gates(cumulative_gates1)
            print("-------gate layer:%d-------"%(2))
            self.__visualize_gates(cumulative_gates2)
            self.__print_accuracies(accuracies)
            self.__print_accuracies(sub_accuracies, acc_type='Sub Accuracies')
            print("\n")

    def test_stage2(self, dataloader_test):

        #Getting model architecture from config
        # model_file = 'architectures.' + self.config['stage_2']['model_file']
        # module_network = importlib.import_module(model_file)
        # model = module_network.stage2_model
        # model = model.to(device)


        model_file = self.config['stage_2']['model_file']
        module_network = importlib.import_module('architectures.' + model_file)
        # stage_2_set_up = module_network.Expert_Network_stage_2(self.config_file)
        module_network = getattr(module_network, model_file)
        stage_2_set_up = module_network(self.config_file)
        model = stage_2_set_up.get_stage_2_model()
        model = model.to(device)

        #loading model
        model.load_state_dict(torch.load('saved_models/'+self.config['stage_2']['model_save_name'],map_location=device.type)['weights'])
        model.eval()

        initialization = np.zeros((1,3))



        with torch.no_grad():

            cumulative_gates1 = {"dataset1":initialization,"dataset2":initialization,
                                 "dataset3":initialization,"dataset4":initialization}
            cumulative_gates2 = cumulative_gates1.copy()
            accuracies = {'words':[0,0],'objects':[0,0],'faces':[0,0],'Expert Dataset':[0,0]}
            sub_accuracies = {'words':[0,0],'objects':[0,0],'faces':[0,0],'Expert Dataset':[0,0]}

            for t, (x, y) in enumerate(dataloader_test):
                x = x.to(device)
                y = y.to(device)
                gate1,gate2,scores = model(x)

                current_gates1 = self.__gate_values(gate1, y, expertise_task=True)
                current_gates2 = self.__gate_values(gate2, y, expertise_task=True)
                cumulative_gates1 =  self.__append_values(current_gates1, cumulative_gates1)
                cumulative_gates2 =  self.__append_values(current_gates2, cumulative_gates2)

                _, preds = scores.max(1)
                self.__get_accuracy(preds, y, accuracies, expertise_task=True)
                self.__get_sublevel_acc(preds, y, sub_accuracies, expertise_task=True)

            print("-------gate layer:%d-------"%(1))
            self.__visualize_gates(cumulative_gates1)
            print("-------gate layer:%d-------"%(2))
            self.__visualize_gates(cumulative_gates2)
            self.__print_accuracies(accuracies, expertise_task=True)
            self.__print_accuracies(sub_accuracies, acc_type='Sub Accuracies', expertise_task=True)
            print("\n")

