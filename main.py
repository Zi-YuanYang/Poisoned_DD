import os
import torch
import argparse
import copy
import numpy as np
import torch.nn as nn
import json
import datetime
import scipy.io as sio 

from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug, augment

from utils_dd import evaluate_synset_benign, set_seed, evaluate_synset_finger, evaluate_synset_backdoor,evaluate_synset_for_backed_recons

from models.fing_network import *

### Training Model is ConvNet
parser = argparse.ArgumentParser(description='Parameter Processing')
parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
parser.add_argument('--model', type=str, default='ConvNet', help='model')
parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
parser.add_argument('--num_eval', type=int, default=10, help='the number of evaluating randomly initialized models')  ## 多少个随机种子，搞成3
parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data') ## 改成 100 之前是 300
parser.add_argument('--Iteration', type=int, default=10, help='training iterations') ### 100
parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for downstream network')
parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
parser.add_argument('--init', type=str, default='noise', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
parser.add_argument('--data_path', type=str, default='data', help='dataset path')

parser.add_argument('--dis_metric', type=str, default='cos', help='distance metric')

### Check them Every Time
parser.add_argument('--seed', type=int, default=42,
                    help='the seed')

parser.add_argument('--save_path', type=str, default='./bd_result', help='path to save results')
parser.add_argument('--mode_pro', type=str, default='finger', help='path to save results')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')  ## CIFAR10
parser.add_argument('--img_path', type=str, default='/storage/dataset/DD_Dataset/DCDSA_Synthetic_Sets/res_DC_CIFAR10_ConvNet_1ipc.pt', help='distance metric')
parser.add_argument('--label_path', type=str, default='ours', help='distance metric')
parser.add_argument('--round',type=int, default=1)
parser.add_argument('--phase',type=str, default='Train')
parser.add_argument('--fing_lr',type=float, default=0.01, help='the learning rate of the fingerprinting network')

parser.add_argument('--fing_size',type=int, default=128, help='the length of fingerprint')
parser.add_argument('--med_size',type=int, default=512, help='the length of fingerprint')    
parser.add_argument('--fing_train_epoc',type=int, default=100, help='the length of fingerprint')
parser.add_argument('--mu',type=float, default=100, help='the weight of fingerprint in combination with the input')


parser.add_argument('--task_weight', type=float, default=0,
                    help='the weight of dd performance')
parser.add_argument('--pred_level_weight', type=float, default=0.01,
                    help='the weight of prediction-level reconstruction')
parser.add_argument('--fing_recon_weight', type=float, default=1,
                    help='the weight of fingerprint reconstruction')

parser.add_argument('--test_num_fingerprint', type=int, default=3,
                    help='the weight of fingerprint reconstruction')
parser.add_argument('--parameter_size', type=int, default=512,
                    help='the size of intermediate feature')
parser.add_argument("--gpu_id",type=str, default='0')

parser.add_argument("--recon_step",type=int,default=200)
parser.add_argument("--num_recon_imgs",type=int,default=1)
parser.add_argument("--tri_val",type=int,default=10)
parser.add_argument("--tri_size",type=int,default=1)
parser.add_argument("--alpha",type=float,default=0.6)

args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
torch.cuda.set_device(int(args.gpu_id)) 

def get_daparam_fin(dataset):

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    return dc_aug_param

def get_phis(phi_dimension, batch_size ,eps = 1e-8):
    phi_length = phi_dimension
    b = batch_size
    phi = torch.empty(b,phi_length).uniform_(0,1)
    return torch.bernoulli(phi) + eps


class BernoulliFingerprintSampler(nn.Module):
    def __init__(self, size):
        super(BernoulliFingerprintSampler, self).__init__()
        self.size = size
        self.eps = 1e-8

    def forward(self):
        return torch.bernoulli(torch.full((self.size,), 0.5))

def feature_matching_loss(p,q):
    return F.kl_div(F.log_softmax(p,dim=-1),F.log_softmax(q,dim=-1))

def add_backdoor(img, trigger_value=0, trigger_size=5):

    img = img.clone()  
    _, _, H, W = img.shape
    img[:, :, H-trigger_size:H, W-trigger_size:W] = trigger_value
    return img

def epoch__(epo,model_train_pool,img,lab,criterion):
    
    networks = []
    optimizers = []
    for model_name in model_train_pool:
        net_eval = get_network(model_name, args.channel, args.num_classes, args.image_size).to(args.device)
        networks.append(net_eval)  

        optimizer = torch.optim.SGD(net_eval.parameters(), lr=float(args.lr_net), momentum=0.9, weight_decay=0.0005)
        optimizers.append(optimizer)   
    for i in range(epo):
        for net_ind in range(len(networks)):
            img = img.to(args.device)
            lab = lab.to(args.device)
            output, _ = networks[net_ind](img)
            loss = criterion(output, lab)
            optimizers[net_ind].zero_grad()
            loss.backward()
            optimizers[net_ind].step()

    return networks


def reconstruct_images_for_all_classes(model, num_classes, img_shape, channel, device, num_images_per_class=10,recon_step=500):

    model.eval()
    reconstructed_images = {label: [] for label in range(num_classes)} 
    model = model.cuda()

    images = [] 
    labels = []  

    for target_label in range(num_classes):
        print(f"Reconstructing images for class {target_label}...")
        
        for img_idx in range(num_images_per_class):
            reconstructed_img = torch.randn((1, channel, img_shape,img_shape), device=device, requires_grad=True)
            optimizer = torch.optim.Adam([reconstructed_img], lr=0.1)

            for step in range(recon_step): 
                optimizer.zero_grad()
                output, _ = model(reconstructed_img)
                loss = -output[:, target_label].mean() 
                loss.backward()
                optimizer.step()

            images.append(reconstructed_img.detach().squeeze(0)) 
            labels.append(target_label)

    images_tensor = torch.stack(images) 
    labels_tensor = torch.tensor(labels, device=device)
    return images_tensor, labels_tensor


def main():

    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(args.device)
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    print(args.dsa)

    args.save_path = os.path.join(args.save_path,args.mode_pro)

    model_train_pool = [args.model]
    model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']

    accs_all_exps_clean_task = dict() # record clean performances of all experiments
    asrs_backdoor = dict() # record clean performances of all experiments
    asrs_clean = dict() # record clean performances of all experiments    
    for key in model_eval_pool:
        accs_all_exps_clean_task[key] = {}
        asrs_backdoor[key] = {}
        asrs_clean[key] = {}
        
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset,
                                                                                                         args.data_path)
    args.channel = channel
    args.image_size = im_size
    args.num_classes = num_classes

    if args.method.lower() == 'mtt':
        img_path = args.img_path
        image_syn = torch.load(img_path + 'images_best.pt')
        label_syn = torch.load(img_path+'labels_best.pt')
    else:
        data = torch.load(args.img_path, map_location='cpu')['data']
        image_syn = data[args.round][0]
        label_syn = data[args.round][1]

    image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(
        label_syn.detach())  
    images_train = image_syn_eval.to(args.device)
    labels_train = label_syn_eval.to(args.device)
    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    args.dc_aug_param = get_daparam_fin(args.dataset) 

    networks = []
    optimizers = [] 
    set_seed(args.seed)
    for model_name in model_train_pool:
        net_eval = get_network(model_name, channel, num_classes, im_size).to(args.device)
        networks.append(net_eval)  

        optimizer = torch.optim.SGD(net_eval.parameters(), lr=float(args.lr_net), momentum=0.9, weight_decay=0.0005)
        optimizers.append(optimizer)   
    ### Set Learning Schedule
    lr_schedule = [args.epoch_eval_train//2+1]
    lr_train_netowrks = float(args.lr_net)

  ### Loss Function
    criterion = nn.CrossEntropyLoss().to(args.device) ### Task Loss
    recon_fing_loss = nn.BCEWithLogitsLoss().to(args.device) ### Recons Loss

    save_dir = "model_checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    ### Train the fingerprint injection network and decoder
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


    for model_name in model_train_pool:
        net_eval2 = get_network(model_name, channel, num_classes, im_size).to(args.device)
        optimizer2 = torch.optim.SGD(net_eval2.parameters(), lr=float(args.lr_net), momentum=0.9, weight_decay=0.0005)
        break

    net_id = 0
    image_ttemp, label_ttemp = copy.deepcopy(image_syn.detach()), copy.deepcopy(
        label_syn.detach())  # avoid any unaware modification
    for i in range(args.epoch_eval_train):

        image_ttemp = image_ttemp.to(args.device)
        label_ttemp = label_ttemp.to(args.device)
        output, _ = net_eval2(image_ttemp)
        loss_real = criterion(output, label_ttemp)  
        optimizer2.zero_grad()
        loss_real.backward()
        optimizer2.step()
    
    recons_imgs, recon_labels = reconstruct_images_for_all_classes(net_eval2, num_classes, im_size[0], args.channel, args.device, args.num_recon_imgs,args.recon_step)

    image_syn_upda, label_syn_upda = copy.deepcopy(image_syn.detach()), copy.deepcopy(
        label_syn.detach())  # avoid any unaware modification

    image_syn_upda = image_syn_upda.to(args.device).detach().requires_grad_(True)
    optimizer_syn = torch.optim.Adam([image_syn_upda], lr=0.1)
    label_syn_upda = label_syn_upda.to(args.device).detach()


    for i in range(args.Iteration + 1):
            
        for i_batch, datum in enumerate(trainloader):

            img = datum[0].float().to(args.device)
            lab = datum[1].long().to(args.device)

            if i != 0:
                networks=None
                networks = epoch__(i,model_train_pool,img,lab,criterion)
            
            for net_id in range(len(model_train_pool)):
                print('123')

                backed_imgs = add_backdoor(recons_imgs,trigger_value=args.tri_val, trigger_size=args.tri_size)
                backdoor_labels = torch.zeros((backed_imgs.shape[0],), dtype=torch.long, device=args.device)  

                for epoch in range(args.fing_train_epoc+1):
                    
                    net_parameters = list(networks[net_id].parameters())

                    output, _ = networks[net_id](img)
                    loss_real = criterion(output, lab)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    backdoor_output, _ = networks[net_id](backed_imgs)
                    loss_backdoor = criterion(backdoor_output, backdoor_labels)
                    gw_backdoor = torch.autograd.grad(loss_backdoor, net_parameters)
                    gw_backdoor = list((_.detach().clone() for _ in gw_backdoor))

                    output_syn, _ = networks[net_id](image_syn_upda)
                    loss_syn = criterion(output_syn, label_syn_upda) 
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss_grad_real = match_loss(gw_syn, gw_real, args) 
                    loss_grad_backdoor = match_loss(gw_syn, gw_backdoor, args) 

                    loss_total = (1-args.alpha) * loss_grad_real + args.alpha * loss_grad_backdoor 
                    # loss_total = loss_grad_backdoor

                    optimizer_syn.zero_grad()
                    loss_total.backward()
                    optimizer_syn.step()
                
                    print('%s | epoch = %04d fing_roung = %04d loss_total = %.6f  loss_grad_real = %.6f loss_grad_backdoor = %.6f' % (get_time(), i,epoch,loss_total.item(), loss_grad_real.item(), loss_grad_backdoor.item()))

    for idx,model_eval in enumerate(model_eval_pool):
    
        set_seed(args.seed)
    
        for epo in range(50,args.epoch_eval_train+1,50):

            accs = []
            asrs = []
            clean_asrs = []

            for it_eval in range(args.num_eval):  
                net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)  # get a random model
                image_sy, label_sy = copy.deepcopy(image_syn_upda.detach()), copy.deepcopy(
                    label_syn_upda.detach())  # avoid any unaware modification

                _, acc_train, acc_test, asr_clean, asr_avg = evaluate_synset_for_backed_recons(it_eval, net_eval, image_sy, label_sy, backed_imgs, testloader,
                                    args, epo)

                # accs.append(acc_test)
                accs.append(acc_test)
                asrs.append(asr_avg)
                clean_asrs.append(asr_clean)

            if epo not in accs_all_exps_clean_task[model_eval]: 
                accs_all_exps_clean_task[model_eval][epo] = []
            if epo not in asrs_backdoor[model_eval]:  
                asrs_backdoor[model_eval][epo] = []
            if epo not in asrs_clean[model_eval]:  
                asrs_clean[model_eval][epo] = []
            
            accs_all_exps_clean_task[model_eval][epo].append(accs)
            asrs_backdoor[model_eval][epo].append(asrs)
            asrs_clean[model_eval][epo].append(clean_asrs)


    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    str_alpha = str(args.alpha)
    str_num_recon_imgs = str(args.num_recon_imgs)
    str_tri_size = str(args.tri_size)
    str_tri_val = str(args.tri_val)
    output_dir = f"./results_backdoor/{args.method}/{args.model}/{args.dataset}/alpha_{str_alpha}_recon_num_{str_num_recon_imgs}_tri_size_{str_tri_size}_tri_value_{str_tri_val}/BACKED/{current_time}"
    os.makedirs(output_dir, exist_ok=True)

    # save the image and label
    image_save_path = os.path.join(output_dir, "image_syn_upda.pt")
    label_save_path = os.path.join(output_dir, "label_syn_upda.pt")

    torch.save(image_syn_upda, image_save_path)
    torch.save(label_syn_upda, label_save_path)

    ## Store Setting
    args.dsa_param = 'deafult'
    args_dict = vars(args)
    output_setting_file = os.path.join(output_dir, "setting.json")

    with open(output_setting_file, 'w') as f:
        json.dump(args_dict, f, indent=4)

    ## Store Clean 
    all_exps_data = {
    'clean_task': accs_all_exps_clean_task
    }

    output_res_file = os.path.join(output_dir, "clean_exps_data.json")

    with open(output_res_file, 'w') as f:
        json.dump(all_exps_data, f)

    ## Store Attack 
    all_exps_data = {
    'attack': asrs_backdoor
    }

    output_res_file = os.path.join(output_dir, "attack_exps_data.json")

    with open(output_res_file, 'w') as f:
        json.dump(all_exps_data, f)

    ## Store Attack Clean
    all_exps_data = {
    'attack_clean': asrs_clean
    }

    output_res_file = os.path.join(output_dir, "attack_clean_exps_data.json")

    with open(output_res_file, 'w') as f:
        json.dump(all_exps_data, f)


if __name__ == '__main__':

    main()
