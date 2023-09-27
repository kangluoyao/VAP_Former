'''The following module trains the weights of the neural network model.'''
import os
import datetime
import uuid
from tqdm import tqdm

import torch
import torch.nn    as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from data_declaration import Task
from loader_helper    import LoaderHelper


from vapformer.model_components import thenet
from evaluation import evaluate_model
import matplotlib.pyplot as plt
import numpy as np
import random
from torchmetrics.classification import BinaryAUROC

def load_cam_model(path):
    model = torch.load(path)
    return model

def load_prompt_model(path):
    model = thenet(
            input_size=[37 * 45 * 37, 18 * 22 * 18, 9 * 11 * 9, 4 * 5 * 4],
            dims=[32, 64, 128, 256], 
            depths=[3, 3, 3, 3], 
            num_heads=8,
            in_channels=1
            )
    pretextmodel = torch.load(path).state_dict()
    model2_dict = model.state_dict()
    state_dict = {k:v for k,v in pretextmodel.items() if k in model2_dict.keys()}
    model2_dict.update(state_dict)
    model.load_state_dict(model2_dict)
    model.double()
    return model

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    print("Running on the GPU.")
else:
    DEVICE = torch.device("cpu")
    print("Running on the CPU")

def save_weights(model_in, uuid_arg, epoch, fold=1, task: Task = None):
    '''The following function saves the weights file into required folder'''
    root_path = ""

    if task == Task.NC_v_AD:
        root_path = "../weights/NC_v_AD/"     + uuid_arg + "/"
    else:
        root_path = "../weights/sMCI_v_pMCI/" + uuid_arg + "/"

    if os.path.exists(root_path) == False:
        os.mkdir(root_path) #otherwise it already exists

    while True:

        s_path = root_path + "fold_{}_epoch{}_weights-{date:%Y-%m-%d_%H:%M:%S}".format(fold, epoch, date=datetime.datetime.now()) # pylint: disable=line-too-long

        if os.path.exists(s_path):
            print("Path exists. Choosing another path.")
        else:
            torch.save(model_in, s_path)
            break
def save_best_weights(model_in, uuid_arg,task: Task = None):
    '''The following function saves the weights file into required folder'''
    root_path = ""

    if task == Task.NC_v_AD:
        root_path = "../weights/NC_v_AD/"     + uuid_arg + "/"
    else:
        root_path = "../weights/sMCI_v_pMCI/" + uuid_arg + "/"

    if os.path.exists(root_path) == False:
        os.mkdir(root_path) #otherwise it already exists

    

    s_path = root_path + "best_weight" # pylint: disable=line-too-long

        
    torch.save(model_in, s_path)
   

def load_model():
    '''Function for loaded camull net from a specified weights path'''
    pth = "./weights/best_weight"
    model = load_prompt_model(pth)
    # model = load_cam_model(pth)
    print('load from', pth)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model.to(DEVICE)

    return model

def build_arch():
    '''Function for instantiating the pytorch neural network object'''
    net = thenet(
                input_size=[37 * 45 * 37, 18 * 22 * 18, 9 * 11 * 9, 4 * 5 * 4],
                dims=[32, 64, 128, 256], 
                depths=[3, 3, 3, 3], 
                num_heads=8,
                in_channels=1
                )
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    net.to(DEVICE)
    net.double()

    return net


def evaluate(model_in, test_dl, thresh=0.5, param_count=False):
        
    correct = 0; total = 0
    model_in.eval()
    total_label = torch.tensor([]).to(DEVICE)
    total_pre = torch.tensor([]).to(DEVICE)
    
    TP = 0.000001; TN = 0.000001; FP = 0.000001; FN = 0.000001
    
    with torch.no_grad():
        
        for i_batch, sample_batched in enumerate(test_dl):
            
            batch_X  = sample_batched['mri'].to(DEVICE)
            batch_clinical = sample_batched['clin_t'].to(DEVICE)
            batch_y  = sample_batched['label'].to(DEVICE)

            net_out = model_in(batch_X,batch_clinical)
            total_label = torch.cat((total_label,batch_y),1)
            total_pre = torch.cat((total_pre,net_out),1)


            for i in range(len(batch_X)):
                
                real_class = batch_y[i].item()

                
                predicted_class = 1 if net_out[i] > thresh else 0      
                
                if (predicted_class == real_class):
                    correct += 1
                    if (real_class == 0):
                        TN += 1
                    elif (real_class == 1):
                        TP += 1
                else:
                    if (real_class == 0):
                        FP += 1
                    elif (real_class == 1):
                        FN += 1
                    
                    
                total += 1

    metric = BinaryAUROC(thresholds=None)
    auc = metric(total_pre, total_label).item()
    
    sensitivity = round((TP / (TP + FN)), 5)
    specificity = round((TN / (TN + FP)), 5)
    accuracy = round((sensitivity+specificity)/2, 5)

    
    
    return accuracy, sensitivity, specificity, auc

def mixup_criterion(criterion, pred, y_a, y_b, lam, pow=2): 
    y = lam ** pow * y_a + (1 - lam) ** pow * y_b 
    return criterion(pred, y) 

def mixup_data(v, q, a, alpha=1, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda without organ constraint'''
    lam = np.random.beta(alpha, alpha)

    batch_size = v.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_v = lam * v + (1 - lam) * v[index, :]
    mixed_q = lam * q + (1 - lam) * q[index, :]

    a_1, a_2 = a, a[index]
    return mixed_v, mixed_q, a_1, a_2, lam

def train_loop(model_in, train_dl, test_dl, epochs, uuid_, k_folds, task):
    '''Function containing the neural net model training loop'''

    optimizer = optim.AdamW(model_in.parameters(), lr=0.00001, weight_decay=5e-4)
    scheduler_warm = lr_scheduler.StepLR(optimizer,step_size=1, gamma=1.4)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=1)

    loss_function = nn.BCELoss()
    loss_fig = []
    eva_fig = []

    model_in.train()
    best_auc = 0
    nb_batch = len(train_dl)
    log_path = "../train_log/" + uuid_ + ".txt"

    if (os.path.exists(log_path)):
        filein     = open(log_path, 'a')
    else:
        filein     = open(log_path, 'w')

    # Train
    for i in range(1,1+epochs):
        loss = 0.0
        model_in.train()
        for _, sample_batched in enumerate(tqdm(train_dl)):

            batch_x = sample_batched['mri'].to(DEVICE)
            batch_clinical = sample_batched['clin_t'].to(DEVICE)
            batch_y = sample_batched['label'].to(DEVICE)

            model_in.zero_grad()
            outputs = model_in(batch_x,batch_clinical)
            
            batch_loss = loss_function(outputs, batch_y)
            batch_loss.backward()
            optimizer.step()


            loss += float(batch_loss) / nb_batch

        tqdm.write("Epoch: {}/{}, train loss: {}".format(i, epochs, round(loss, 5)))
        filein.write("Epoch: {}/{}, train loss: {}\n".format(i, epochs, round(loss, 5)))
        loss_fig.append(round(loss, 5))
        accuracy, sensitivity, specificity, auc = evaluate(model_in, test_dl)
        eva_fig.append(accuracy)
        tqdm.write("Epoch: {}/{}, evaluation loss: {}".format(i, epochs,(accuracy, sensitivity, specificity, auc)))
        filein.write("Epoch: {}/{}, evaluation loss: {}\n".format(i, epochs,(accuracy, sensitivity, specificity, auc)))

        if i % 10 == 0 and i != 0:
        # if i >= 0:
            save_weights(model_in, uuid_, epoch = i, fold=k_folds, task=task)
            plt.plot(range(i),loss_fig,label='loss')
            plt.plot(range(i),eva_fig,label='BA')
            plt.savefig("../figures/"+uuid_+'eva.png')   

        # elif i >= (epochs - 10):
        #     plt.plot(range(i),loss_fig,label='loss')
        #     plt.plot(range(i),eva_fig,label='BA')
        #     plt.savefig("../figures/"+uuid_+'eva.png') 
        #     save_weights(model_in, uuid_, epoch = i, fold=k_folds, task=task)     

        if auc >= best_auc:
           save_best_weights(model_in, uuid_, task=task)
           best_auc = auc

        if epochs <= 5:
            scheduler_warm.step() 
        else:            
            scheduler.step(loss)

        # DAFT
        # scheduler_daft.step()

    
    


def train_camull(ld_helper, k_folds=1, model=None, epochs=40):
    '''The function for training the camull network'''
    task = ld_helper.get_task()
    uuid_ = "{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())

    print(uuid_)
    model_cop = model

    for k_ind in range(k_folds):

        if model_cop is None:
            model = build_arch()
        else:
            model = model_cop

        train_dl = ld_helper.get_train_dl(k_ind)
        test_dl = ld_helper.get_test_dl(k_ind)
        train_loop(model, train_dl, test_dl, epochs, uuid_, k_folds=k_ind+1, task=task)
               

        print("Completed fold {}/{}.".format(k_ind, k_folds))

    return uuid_




def main():
    '''Main function of the module.'''
    # setup_seed(2023)

    #NC v AD
    # ld_helper = LoaderHelper(task=Task.NC_v_AD)
    # model_uuid = train_camull(ld_helper, epochs=50)

    #transfer learning for pMCI v sMCI
    ld_helper = LoaderHelper(task=Task.sMCI_v_pMCI)
    model = load_model()
    model_uuid  = train_camull(ld_helper, model=model, epochs=50)
    evaluate_model(DEVICE, model_uuid, ld_helper)

main()
