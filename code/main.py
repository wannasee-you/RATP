import argparse
import torch
import numpy as np
import utils1
from dataset import Dateset_mat
import os
import logging
from model import Encoder_Special, UD_constraint
from utils1 import data_loder
from lightly.loss import NTXentLoss
import warnings
from sklearn.metrics import pairwise_distances
import torch.nn.functional as F

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", default=r'./dataset/flickr', type=str)
parser.add_argument("--lr", type=float, default=0.000001)
parser.add_argument("--num_epochs", type=int, default=200)
parser.add_argument("--fea_dim", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--cluster_num", type=int, default=6)
parser.add_argument("--iters", type=int, default=10)
parser.add_argument("--seed", type=int, default=4)

config = parser.parse_args()
config.max_ACC = 0
Dataset = Dateset_mat(config.dataset_root)
dataset = Dataset.getdata()
label1 = np.array(dataset[2]) - 1
all_label = np.squeeze(label1)

max_ACC = 0
NTX_loss = NTXentLoss()
log_file_name = "log_seed_" + str(config.seed) + ".txt"

def run_S():
    os.makedirs(config.dataset_root, exist_ok=True)
    log_file = os.path.join(config.dataset_root, log_file_name)
    logging.basicConfig(
        level=logging.INFO,
        # format="%(asctime)s - %(levelname)s - %(message)s", 
        handlers=[
            logging.FileHandler(log_file, mode='a', encoding='utf-8'),
            logging.StreamHandler()  
        ]
    )
    logger = logging.getLogger(__name__)
    all_img = (torch.tensor(dataset[0], dtype=torch.float32)).to(device)
    all_txt = torch.tensor(dataset[1], dtype=torch.float32).to(device)
    img1, txt1, label = dataset[0], dataset[1], dataset[2]
    all_label = np.squeeze(label)

    print("clustering number: ", config.cluster_num)
    data = data_loder(config.batch_size)
    data.get_data(img1, txt1, label)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    global_best_metrics = {
        'max_acc': 0.0,
        'max_nmi': 0.0,
        'loss': 0.0,
        'best_iter': 0,
        'best_epoch': 0
    }
    for iter_outer in range(1, config.iters+1):
        model = Encoder_Special(all_img.size(1), all_txt.size(1), config.fea_dim, config.cluster_num, config.batch_size).to(device)
        optimiser_S = torch.optim.Adam(model.parameters(), lr=config.lr)
        print("iter:", iter_outer)
        epoch_metrics = []
        for epoch in range(config.num_epochs):
            model.train()
            model.zero_grad()
            for img, txt, label in data:
                img, txt = img.to(device), txt.to(device)
                for iter_inner in range(iter_outer+1):
                    img_fea, txt_fea, img_cluster, txt_cluster, loss_vae = model(img, txt, epoch)
                    C = generate_consensus_matrix(img_cluster, txt_cluster)
                    loss1 = 1 * NTX_loss(img_fea, txt_fea) + 1 * NTX_loss(img_cluster, txt_cluster)
                    UDC_img = UD_constraint(img_cluster).to(device)
                    UDC_txt = UD_constraint(txt_cluster).to(device)
                    loss2 = criterion(img_cluster, UDC_img) + criterion(txt_cluster, UDC_txt)
                    loss_con = consensus_loss(img_fea, txt_fea, img_cluster, txt_cluster, C)
                    if iter_inner==0:
                        loss = 1 * (loss1 + loss2) + 1 * loss_vae + 0.1 * loss_con
                        loss.backward()
                        optimiser_S.step()
                    else:
                        loss_ = 0.1 * loss_con 
                        loss_.backward()
                        optimiser_S.step()
                    
            print("epoch:",epoch)
            total_loss = loss + config.iters * loss_
            print("loss1 %.4f loss2 %.4f loss_vae %.4f loss_con %.4f total_loss %.4f "% (loss1, loss2, loss_vae, loss_con, total_loss))
            acc, nmi = get_S_ACC(model, all_img, all_txt, all_label, epoch)
            epoch_metrics.append((acc, nmi, total_loss))
            print("S: acc %.4f nmi %.4f "% (acc, nmi))
        all_acc = [metric[0]for metric in epoch_metrics]
        all_nmi = [metric[1]for metric in epoch_metrics]
        all_loss = [metric[2]for metric in epoch_metrics]
        max_acc_idx = all_acc.index(max(all_acc))
        c_max_acc = all_acc[max_acc_idx]
        c_corresponding_nmi = all_nmi[max_acc_idx]
        c_best_epoch = max_acc_idx + 1
        c_loss = all_loss[max_acc_idx]
        print(f"\n===== iter {iter_outer} 最优结果 =====")
        print(f"ACC: {c_max_acc:.4f}，NMI: {c_corresponding_nmi:.4f}, loss: {c_loss:.4f}")
        print(f"（epoch: {c_best_epoch}）\n")
        logger.info(f"\n===== iter {iter_outer} 最优结果 =====")
        logger.info(f"ACC: {c_max_acc:.4f}，NMI: {c_corresponding_nmi:.4f}, loss: {c_loss:.4f}, epoch: {c_best_epoch}）\n")
        logger.info(f"{'='*50}\n")
        if c_max_acc > global_best_metrics['acc']:
            global_best_metrics.update({
            'max_acc': c_max_acc,
            'max_nmi': c_corresponding_nmi,
            'loss': c_loss,
            'best_iter': iter_outer,
            'best_epoch': c_best_epoch
        })
        elif c_max_acc == global_best_metrics['max_acc']:
         if c_corresponding_nmi > global_best_metrics['max_nmi']:
            global_best_metrics.update({
                'max_nmi': c_corresponding_nmi,
                'loss': c_loss,
                'best_iter': iter_outer,
                'best_epoch': c_best_epoch
            })
    print(f"\n===== 全局最优结果 =====")
    print(f"ACC: {global_best_metrics['max_acc']:.4f}")
    print(f"NMI: {global_best_metrics['max_nmi']:.4f}")
    print(f"循环次数（iter_outer）: {global_best_metrics['best_iter']}")
    print(f"epoch: {global_best_metrics['best_epoch']}")
    logger.info(f"\n===== 全局最优结果 =====")
    logger.info(f"ACC: {global_best_metrics['max_acc']:.4f}，NMI: {global_best_metrics['max_nmi']:.4f}, loss: {global_best_metrics['loss']:.4f}, iters: {global_best_metrics['best_iter']:.4f}, epoch: {global_best_metrics['best_epoch']}）\n")
    logger.info(f"{'='*50}\n")
    # for handler in logger.handlers:
    #     handler.close()
    # logger.handlers.clear()
            
def generate_consensus_matrix(labels1, labels2):
    """生成共识矩阵 C"""
    labels1 = torch.argmax(labels1, dim=1)
    labels2 = torch.argmax(labels2, dim=1)
    n = len(labels1)
    C = np.zeros((n, n), dtype=np.float32)
    same1 = (labels1.unsqueeze(0) == labels1.unsqueeze(1))
    same2 = (labels2.unsqueeze(0) == labels2.unsqueeze(1))
    C = torch.where(same1 & same2, 1.0, torch.where(~same1 & ~same2, -1.0, 0.0))
    return C

def consensus_loss(fea1, fea2, prob1, prob2, C):
    device = fea1.device
    C = torch.tensor(C, device=device)
    fea1_norm = F.normalize(fea1, p=2, dim=1)
    fea2_norm = F.normalize(fea2, p=2, dim=1)
    dist1 = 1 - torch.matmul(fea1_norm, fea1_norm.t())  
    dist2 = 1 - torch.matmul(fea2_norm, fea2_norm.t())
    mask_same = (C == 1.0)
    mask_diff = (C == -1.0)
    loss_same = torch.mean(dist1[mask_same] + dist2[mask_same])
    loss_diff = torch.mean(dist1[mask_diff] + dist2[mask_diff])
    loss_consensus = loss_diff / loss_same
    return loss_consensus

def get_S_ACC(model, data1, data2, label, epoch):
    model.eval()
    with torch.no_grad():
        _, _,  x_out1, x_out2, _ = model(data1, data2, epoch)
    pre_label = np.array(x_out1.cpu().detach().numpy())
    pre_label = np.argmax(pre_label, axis=1)

    acc1 = utils1.metrics.acc(pre_label, label)
    nmi1 = utils1.metrics.nmi(pre_label, label)

    pre_label = np.array(x_out2.cpu().detach().numpy())
    pre_label = np.argmax(pre_label, axis=1)

    acc2 = utils1.metrics.acc(pre_label, label)
    nmi2 = utils1.metrics.nmi(pre_label, label)
    return max(acc1, acc2),max(nmi1, nmi2)


if __name__ == '__main__':
    run_S()