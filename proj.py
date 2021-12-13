import torch 
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from torchvision import models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import random
import os
import math
import copy
import sys
sys.path.append("/home/simnz/CISC455")
from utils.getData import processing


class getID(Dataset):
  '''
  This class is a Dataset class, returns the batch index
  '''
  def __init__(self, total_len):
      self.tmp = total_len
      self.total_len = self.tmp
  
  def __len__(self):
      return self.total_len
    
  def __getitem__(self, ind):
      return torch.Tensor([ind])


class MyClassifier(nn.Module):
  '''
  This class defines the CNN architecture
  '''
  def __init__(self, pop_list, num_classes):
      super().__init__()
      self.pop_list = pop_list
      self.num_classes = num_classes
      self.conv1_dim = int(pop_list[0]) # convolution dim
      self.conv2_dim = int(pop_list[1])
      self.conv3_dim = int(pop_list[2])
      self.kernal1_dim = int(pop_list[3]) # kernel size
      self.kernal2_dim = int(pop_list[4])
      self.kernal3_dim = int(pop_list[5])
      self.lin1_fea = int(self.pop_list[12]) # linear feature
      self.lin2_fea = int(self.pop_list[13])
      self.adaptive_num = 5
      # adaptive pooling layer for consistency output regardless the input and output dim
      self.adaptive_pool = nn.AdaptiveAvgPool2d((self.adaptive_num,self.adaptive_num))
      
      # feature extractor
      net = []
      #net.append(nn.BatchNorm2d(3))
      net.append(nn.Conv2d(in_channels=3, out_channels=32, padding=1, kernel_size=3, stride=1))
      net.append(nn.BatchNorm2d(32))
      net.append(self.get_acfun(self.pop_list[6]))
      net.append(nn.MaxPool2d(kernel_size=2))
      net.append(nn.Conv2d(in_channels=32, out_channels=self.conv1_dim, padding=1, kernel_size=self.kernal1_dim, stride=1))
      net.append(nn.BatchNorm2d(self.conv1_dim))
      net.append(self.get_acfun(self.pop_list[7]))
      net.append(nn.Conv2d(in_channels=self.conv1_dim, out_channels=self.conv2_dim, padding=1, kernel_size=self.kernal2_dim, stride=1))
      net.append(nn.BatchNorm2d(self.conv2_dim))
      net.append(self.get_acfun(self.pop_list[8]))
      net.append(nn.MaxPool2d(kernel_size=2))
      net.append(nn.Conv2d(in_channels=self.conv2_dim, out_channels=self.conv3_dim, padding=1, kernel_size=self.kernal3_dim, stride=1))
      net.append(nn.BatchNorm2d(self.conv3_dim))
      net.append(self.get_acfun(self.pop_list[9]))
      net.append(nn.MaxPool2d(kernel_size=2))
      net.append(self.adaptive_pool)
      
      self.features = nn.Sequential(*net)
      
      # classifier
      net2 = []
      net2.append(nn.Linear(in_features=self.conv3_dim*self.adaptive_num**2, out_features=self.lin1_fea))
      net2.append(self.get_acfun(pop_list[10]))
      net2.append(self.get_drop(self.pop_list[14]))
      net2.append(nn.Linear(in_features=self.lin1_fea, out_features=self.lin2_fea))
      net2.append(self.get_acfun(pop_list[11]))
      net2.append(self.get_drop(self.pop_list[15]))
      net2.append(nn.Linear(in_features=self.lin2_fea, out_features=self.num_classes))
      net2.append(nn.Softmax(dim=1))
      
      self.classifier = nn.Sequential(*net2)
      
  def get_acfun(self, ac_name):
      '''
      Helper function, return activation functions
      '''
      if ac_name == "tanh":
        return nn.Tanh()
      elif ac_name == "relu":
        return nn.ReLU(inplace=True)
      else:
        return nn.LeakyReLU(inplace=True)
    
  def get_drop(self, drop_rate):
      '''
      Helper function, return dropout layer
      '''
      drop_rate = float(drop_rate)
      return nn.Dropout(drop_rate)
    
  def forward(self, x):
      '''
      forward pass of nn
      '''
      x = self.features(x)
      x = x.view(x.size(0), -1)
      x = self.classifier(x)
      return x

def get_train_val(train_ratio, total_len):
    '''
    this function splits the data to train and validation
    train_ratio: proportion of training image
    total_len: total number of images
    '''
    torch.manual_seed(1)
    train_num = math.ceil(total_len * train_ratio)
    train_set, val_set = random_split(getID(total_len), lengths=(train_num, total_len - train_num))
    return train_set, val_set


def initialization(num_filter, size_kernel, ac_funs, linear_fea, drop_rate, num_layers, ac_num, lin_num, drop_num, init_pop_size):

    '''
    this function is the initialize parents for GA

    num_filter: [32, 64, 128, 256, 512]
    kernal: [3,5,7]
    activation: [tanh, relu, leakyrelu]
    linear_fea = [512, 256, 128]
    drop_rate = [0.1, 0.2, 0.3, 0.5]
    init_pop_size: 50
    num_layers: number of convolution layers
    ac_num: number of activation layers
    lin_num: number of fully connected layers
    drop_num: number of drop out layers
    return: 2d array
    '''
    # randomly select each parameter
    filter_pop = np.random.choice(num_filter, (init_pop_size, num_layers))
    kernel_pop = np.random.choice(size_kernel, (init_pop_size, num_layers))
    acfunc_pop = np.random.choice(ac_funs, (init_pop_size, ac_num))
    linear_fea_pop = np.random.choice(linear_fea, (init_pop_size, lin_num))
    drop_rate_pop = np.random.choice(drop_rate, (init_pop_size, drop_num))
    # concate together
    total_pop = np.concatenate((filter_pop, kernel_pop, acfunc_pop, linear_fea_pop, drop_rate_pop), axis = 1)
    # store index range of each parameter
    conv_ind = list(range(0, num_layers))
    kernel_ind = list(range(num_layers, 2*num_layers))
    active_ind = list(range(2*num_layers, 2*num_layers+ac_num))
    linear_fea_ind = list(range(2*num_layers+ac_num, 2*num_layers+ac_num+lin_num))
    drop_rate_ind = list(range(2*num_layers+ac_num+lin_num, 2*num_layers+ac_num+lin_num+drop_num))
    # print(total_pop[0,:])
    # print(total_pop.shape)
    # print(conv_ind, kernel_ind, active_ind, linear_fea_ind, drop_rate_ind)
    return total_pop, [conv_ind, kernel_ind, active_ind, linear_fea_ind, drop_rate_ind]


def initializae_network(single_pop, num_classes):
    '''
    This function is for initializing the network given the parameters
    single_pop: single parent in total_pop
    num_classes: number of classes for the task
    '''
    net = MyClassifier(single_pop, num_classes)
    return net

def compute_loss(model, optimizer, images, target, loss_fn, train_mode):
    '''
    this function returns the loss for each training epoch
    model: cnn model
    optimizer: model optimizer
    images: validation data
    target: target label
    loss_fn: loss function
    train_mode: binary, whether the model in training or validation stage
    '''
    predicted = model(images)
    #print(predicted)
    #predicted_lab = torch.argmax(predicted, dim=1).float()
    #print(predicted)
    # compute loss through loss function
    loss = loss_fn(predicted, target.long())

    # train mode, update loss and optimizer
    if train_mode:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    else: # in validation mode, return additional prediction result
        return loss.item(), torch.argmax(predicted, dim=1).float()

    return loss.item()

def train(net, train_ind, train_img, train_lab, val_ind, optimizer, epochs, weights):
    '''
    this function is for model training
    net: cnn model
    train_ind: training data index(s)
    train_img: training data
    train_lab: training label
    val_ind: validation data index(s)
    optimizer: model optimizer
    epochs: epochs
    weights: class weights
    '''
    # weighted cross entropy loss
    loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(weights).to("cuda:1"), reduction="mean")
    lowest_loss = np.inf
    # store validation accuracy
    total_val_acc = []
    # begin training
    for epoch in range(epochs):
      net.zero_grad()
      loss = 0
      #net.train()
      for batch_idx in train_ind:
        net.train()
        # get the training index
        batch_idx = batch_idx.view(-1).long()
        # get the actual training data per batch
        image_batch, label_batch = train_img[batch_idx], train_lab[batch_idx]
        # compute loss
        train_loss = compute_loss(net, optimizer, image_batch, label_batch, loss_fn, train_mode=True)
        loss += train_loss

      # compute average loss
      loss /= len(train_ind)
      #print("epoch: {}, train loss: {}".format(epoch+1, loss))
      loss_eval = 0
      
      # begin validating
      with torch.no_grad():
        net.eval()
        # record predicted labels and total labels
        actual_labels = torch.empty(0).long()
        predicted_labels = torch.empty(0).float()
        for batch_idx in val_ind:
          # get the validation index
          batch_idx = batch_idx.view(-1).long()
          # get the validation data per batch
          image_batch, label_batch = train_img[batch_idx], train_lab[batch_idx]
          #print("val_ind: ", label_batch)
          # store actual labels in this batch
          actual_labels = torch.cat((actual_labels, label_batch.long().cpu()))
          # store the predicted labels in this batch and record the validation loss
          eval_loss, predicted = compute_loss(net, optimizer, image_batch, label_batch, loss_fn, train_mode=False)
          predicted_labels = torch.cat((predicted_labels, predicted.cpu()))
          #print("val_ind predict: ", predicted)
          #val_acc = torch.sum(torch.squeeze(predicted).float() == all_label).item() / float(all_label.size()[0])
          loss_eval += eval_loss
        # get the validation loss
        loss_eval /= len(val_ind)
        # compute the validation accuracy
        val_acc = torch.sum(predicted_labels == actual_labels.float()).item() / len(actual_labels)
        total_val_acc.append(val_acc)
        #print("eval loss: ", loss_eval)
        #print("validation accuracy: ", val_acc)
    return np.mean(total_val_acc) # return the average avalidation accuracy across batches


def get_fitness(total_pop, train_ind, train_img, train_lab, val_ind, epochs, lr, num_classes, weights):
    '''
    this function returns the fitness score for each parent
    total_pop: initial population
    train_ind: training data index(s)
    train_img: training data
    train_lab: training label
    val_ind: validation data index(s)
    epochs: epochs
    lr: learning rate
    num_classes: number of classes for the dataset
    weights: class weights
    '''
    # initialize fitness as a list
    fitness = [0] * len(total_pop)
    for i in range(len(total_pop)):
      # initialize the cnn model for each parent
      net = initializae_network(total_pop[i], num_classes)
      # set the optimizer
      optimizer = torch.optim.Adam(net.parameters(), lr=lr)
      # get the validation acc.
      acc = train(net.to("cuda:1"), train_ind, train_img, train_lab, val_ind, optimizer, epochs, weights)
      fitness[i] = acc
    return fitness

def tournament(total_pop, fitness, tournament_size, matpool_size):
    '''
    this function utilizes the tournament selection to select parent
    total_pop: total population -> 2d array
    fitness: population fitness
    tournament_size: how many parents to compete in each round
    matpool_size: how many parents want to select for producing offspring
    this function is adopted from my assignment1
    '''
    selected_to_mate = []
    index_counter = list(range(len(fitness)))
    hashmap = dict(zip(index_counter, fitness)) # {index:fitness} map
    #print(hashmap)
    
    while len(selected_to_mate) < matpool_size:
      hash_ind = list(hashmap.keys()) # get parent index
      ts_ind = np.random.choice(hash_ind, tournament_size, replace = False) # randomly select (tournament_size) parents, return parent index
      #print(ts_ind)
      # find corresponding fitness score
      fitness_sub = [hashmap[i] for i in ts_ind]
      # select max fitness for the parent to be selected
      temp_map = dict(zip(ts_ind, fitness_sub))
      # winner of the tournament will have the max validation score
      possible_max = np.argwhere(fitness_sub == np.max(fitness_sub)).tolist()
      if len(possible_max) == 1:
        max_fitness = max(fitness_sub)
      else: # random break ties if there are multiple optimal solution
        random_select = np.random.randint(0, len(possible_max))
        max_fitness = fitness_sub[random_select]
      #select_parent = fitness.index(max_fitness)
      # find corresponding parent
      select_parent = ts_ind[list(temp_map.values()).index(max_fitness)]
      selected_to_mate.append(select_parent)
      # remove selected parent from hashmap
      del hashmap[select_parent]
    
    return selected_to_mate

def crossover(p1, p2, index_counter):
    '''
    this function performs the crossover operation between two parents

    p1, p2: two parents selected from tournament -> 1d array
    index_counter: counter to record conv num range, kernal num range, etc.
    
    crossover: [conv num]:0, [kernal num]:1, [activation func]:2, [linear func]:3, [drop rate]:4
    Two parents will crossover with each other based on the number
    if 1 is drawn, then two parents will only cross kernal_num part
    For simplicity, we only select 2 parts 
    '''
    # randomly select two crossover points
    which_parts = np.random.choice(index_counter, 2, replace=False).tolist()
    first_point = which_parts[0]
    second_point = which_parts[1]
    #print(which_parts)
    p1_new = copy.deepcopy(p1)
    #print(p1_new)
    p2_new = copy.deepcopy(p2)
    #print(p2_new)
    # cross_range1 = index_counter[first_point] # list contain index 
    # cross_range2 = index_counter[second_point]
    # p1_new[first_point[0]:first_point[-1]+1], p2_new[first_point[0]:first_point[-1]+1] = p2_new[first_point[0]:first_point[-1]+1], p1_new[first_point[0]:first_point[-1]+1]
    # get the corresponding range
    temp = p1_new[first_point[0]:first_point[-1]+1]
    # swap
    p1_new[first_point[0]:first_point[-1]+1] = p2_new[first_point[0]:first_point[-1]+1]
    p2_new[first_point[0]:first_point[-1]+1] = temp

    # similar to above, now swap the second part
    temp1 = p1_new[second_point[0]:second_point[-1]+1]
    p1_new[second_point[0]:second_point[-1]+1] = p2_new[second_point[0]:second_point[-1]+1]
    p2_new[second_point[0]:second_point[-1]+1] = temp1
    # p1_final = p2_new[first_point[0]:first_point[-1]+1].append(i for i in p1_new[first_point[-1]+1:])
    # p2_final = p1_new[first_point[0]:first_point[-1]+1].append(i for i in p2_new[first_point[-1]+1:])
    return p1_new, p2_new

def multipoint_greedy_mutation(off1, index_counter, kernal_num, lin_fea_num, activ_num, drop_num):
    '''
    this function is the mutation operation of GA
    off: offspring
    first four activation functions is for convolutional layers
    index_counter: list of parameters index
    kernal_num: number of kernels
    lin_fea_num: number of FC layers
    activ_num: number of activation funcs
    drop_num: number of drop rate
    '''
    # first mutate convolution filters
    off = copy.deepcopy(off1)
    # get convolutional layer range
    conv_num = index_counter[0]
    #for i in range(conv_num[0], conv_num[-1]+1):
    # the follow code block rearrange the conv dim to increasing order
    if int(off[1]) <= int(off[0]) and int(off[1]) <= int(off[2]):
      if int(off[0]) >= int(off[2]):
        off[0], off[2] = off[2], off[0]
      off[0] = str(int(off[0]) - np.random.randint(0, 16))
      off[2] = str(int(off[2]) + np.random.randint(0, 32)) 
      off[1] = str(np.random.randint(off[0], off[2]))
    elif int(off[0]) > int(off[1]) > int(off[2]):
      off[0], off[2] = off[2], off[0]
    elif int(off[1]) > int(off[0]) and int(off[1]) > int(off[2]):
      if int(off[0]) > int(off[2]): # if conv1 > conv3
        off[0], off[2] = off[2], off[0] # switch
      off[0] = str(int(off[0]) - np.random.randint(0, 16))
      off[2] = str(int(off[2]) + np.random.randint(0, 32))
      off[1] = str(np.random.randint(int(off[0]), int(off[2])))
    
    # now mutation one of conv, kernel, linear features, activations and drop rate
    next_change = np.random.randint(0, len(index_counter))
    print("mutation ind: ", next_change)
    # still consider mutate convolution layer
    if next_change == 0:
      # now the conv dim is in the increasing order
      index_change =  np.random.choice(index_counter[next_change]) # return one of [0,1,2]
      # randomly mutate
      off[int(index_change)] = str(int(off[int(index_change)]) + np.random.randint(0, 32))
    
    elif next_change == 1: # kernal size
      index_change = np.random.choice(index_counter[next_change]) # return one of [3,4,5]
      kernal_copy = copy.deepcopy(kernal_num)
      kernal_copy.remove(int(off[int(index_change)]))
      off[int(index_change)] = str(np.random.choice(kernal_copy))
    
    elif next_change == 2: # activation functions
      index_change = np.random.choice(index_counter[next_change]) # return one of [6~11]
      acfun_copy = copy.deepcopy(activ_num)
      acfun_copy.remove(off[int(index_change)])
      off[int(index_change)] = str(np.random.choice(acfun_copy))
    
    elif next_change == 3: # linear feature, rearrange to decreasing order
      linear_fea_range = index_counter[next_change] # [12,13]
      if int(off[linear_fea_range[-1]]) > int(off[linear_fea_range[0]]):
        # if last linear feature > first linear feature
        off[linear_fea_range[-1]] = str(np.random.randint(128, int(off[linear_fea_range[0]]))) if int(off[linear_fea_range[0]]) != 128 else "128"
      else: # if last linear feature < first linear feature
        indicator = np.random.random()
        if indicator < 0.5: # 50% mutate first linear feature
          off[linear_fea_range[0]] = str(int(off[linear_fea_range[0]]) + np.random.randint(0, 32))
        else: # 50% mutate last feature
          off[linear_fea_range[-1]] = str(int(off[linear_fea_range[-1]]) + np.random.randint(0, 32))
    
    else: # drop out rate
      index_change = np.random.choice(index_counter[4]) # next_change = 4, return one of [14,15]
      #drop_copy = copy.deepcopy(drop_num)
      #drop_copy.remove(int(off[int(index_change)]))
      off[int(index_change)] = str(round(np.random.uniform(0, min(drop_num))))
    
    return off

def survivor_selection(off, off_fit, cur_pop, cur_fit):
    '''
    this function performs the survivor selection operation in GA
    off: offspring, 2d-array
    off_fit: fitness of offspring
    cur_pop: current total population, 2d-array
    cur_fit: current fitness of total population
    replace the worest n parents with n offspring
    this code is adopted from my assignment1
    '''
    population = []
    fitness = []
    cur_pop_new = copy.deepcopy(cur_pop)
    cur_fit_new = copy.deepcopy(cur_fit)
    off_new = copy.deepcopy(off)
    off_fit_new = copy.deepcopy(off_fit)
    pa_and_child = cur_pop_new + off_new
    pa_and_child_fit = cur_fit_new + off_fit_new
    # sort in descending order
    zip_hash = sorted(zip(pa_and_child_fit, pa_and_child), reverse = True)
    # get the top "cur_pop" parents
    zip_hash_popsize = zip_hash[:len(cur_pop)]
    population = [x[1] for x in zip_hash_popsize]
    fitness = [x[0] for x in zip_hash_popsize]
    return population, fitness

def freeze_layer(model):
    '''
    this function is to freeze the pretrained weights
    '''
    for param in model.parameters():
      param.requires_grad = False

def pretrained(train_loader, x_train, x_lab, val_loader, epochs, lr, class_weight):
    '''
    this function is for computing the validation acc for pretrained model
    '''
    model_ft = models.vgg16(pretrained=True)
    optimizer = torch.optim.Adam(model_ft.parameters(), lr=lr)
    freeze_layer(model_ft)
    num_ftrs = model_ft.classifier[6].in_features
    model_ft.classifier[6] = nn.Linear(num_ftrs,2)
    #print(model_ft)
    #return model_ft
    acc = train(model_ft.to("cuda:1"), train_loader, x_train, x_lab, val_loader, optimizer, epochs, class_weight)
    return acc

def run():
    '''
    This is the main function to perform evolution
    '''
    # np.random.seed(41)
    # random.seed(41)
    file = "/home/simonz/CISC455/y_train_covid.csv"
    num_filter = [32, 64, 128, 256]
    size_kernel = [3,5,7]
    ac_funs = ["tanh", "relu", "leakyrelu"]
    linear_fea = [512, 256, 128]
    drop_rate = [0.1,0.2,0.3,0.5]
    # network structure
    num_layers = 3 # kernel size is also 3
    ac_num = 6
    lin_num = 2
    drop_num = 2
    # parent selection num
    mating_pool_size = 10
    # tournament size
    tournament_size = 5
    # initial pop size
    pop_size = 50
    gen = 0
    gen_limit = 100
    crossover_rate = 0.60
    num_classes = 2
    epochs = 20
    lr = 0.0005

    # compute class weights
    _, class_weight = processing(file)

    # load data
    x_train = torch.load("/home/simonz/CISC455/train_dat455.pt")
    x_lab = torch.load("/home/simonz/CISC455/train_lab455.pt")
    
    # split data
    total_len = x_train.shape[0]
    #torch.manual_seed(2021)
    train_set, val_set = get_train_val(0.80, total_len)
    #torch.manual_seed(torch.initial_seed())
    x_train = x_train.to("cuda:1")
    x_lab = x_lab.to("cuda:1")

    # get the dataloader
    train_loader = DataLoader(train_set, batch_size=16, num_workers=0, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=16, num_workers=0, shuffle=True, drop_last=True)
    
    # initialization
    population, index_counter = initialization(num_filter, size_kernel, ac_funs, linear_fea, drop_rate, num_layers, ac_num, lin_num, drop_num, pop_size)
    print("population: ", population)
    #print(population)
    #fitness = []
    # calc initial fitness
    population = population.tolist()
    fitness = get_fitness(population, train_loader, x_train, x_lab, val_loader, epochs, lr, num_classes, class_weight)
    print("fitness: ", fitness)
    assert len(population) == len(fitness),"population and fitness must have same size!"
    print("generation", gen, ": best fitness", max(fitness), "\taverage fitness", sum(fitness)/len(fitness))
    
    #print(fitness)
    #print(len(population), len(fitness))
    #print(fitness)
    
    '''
    code blocks shown below are adopted from Assignment 1 provided by Dr. Hu
    
    '''
    while gen < gen_limit:
      parents_ind = tournament(population, fitness, tournament_size, mating_pool_size)
      #print(select_in_tour)
      random.shuffle(parents_ind)
      offspring = []
      offspring_fit = []
      i = 0
      while len(offspring) < mating_pool_size:
          
          if np.random.random() < crossover_rate: # crossover or not
              off1, off2 = crossover(population[parents_ind[i]], population[parents_ind[i+1]], index_counter)
              print("crossover performed")
              #off1, off2 = one_point_crossover(population[parents_ind[i]], population[parents_ind[i+1]], max_dis, min_len, max_len)
              #print("one point crossover executed:")
              #print(off1, off2)
          else:
              off1 = population[parents_ind[i]].copy()
              off2 = population[parents_ind[i+1]].copy()
          
          off1 = multipoint_greedy_mutation(off1, index_counter, size_kernel, linear_fea, ac_funs, drop_rate)
          off2 = multipoint_greedy_mutation(off2, index_counter, size_kernel, linear_fea, ac_funs, drop_rate)
          #print("mutations:")
          #print(off1, off2)
          
          offspring.append(off1)
          offspring_fit.append(get_fitness([off1], train_loader, x_train, x_lab, val_loader, epochs, lr, num_classes, class_weight)[0])
          offspring.append(off2)
          offspring_fit.append(get_fitness([off2], train_loader, x_train, x_lab, val_loader, epochs, lr, num_classes, class_weight)[0])
          
          i = i + 2
      
      #offspring_fit = [fit[0] for fit in offspring_fit]
      #print(offspring)
      #print(offspring_fit)
      # off, off_fit, cur_pop, cur_fit
      population, fitness = survivor_selection(offspring, offspring_fit, population, fitness)
      print("population after one generation: ", (len(population), len(population[1])))
      print("fitness after one generation: ", len(fitness))
      #print(len(population), len(fitness))
      
      gen += 1
      if gen % 1 == 0:
          print("generation", gen, ": best fitness", max(fitness), "average fitness", sum(fitness)/len(fitness))
    
    # display result
    k = 0
    global_max = []
    #global global_min
    for i in range(0, pop_size):
        if fitness[i] == max(fitness):
            print("best solution", k, population[i], fitness[i])
            global_max.append([population[i],fitness[i]])
            k += 1
    
    
    pretrain_acc = pretrained(train_loader, x_train, x_lab, val_loader, epochs, lr, class_weight)
    print("pretrained model acc: ", pretrain_acc)
    

if __name__ == "__main__":
    run()
