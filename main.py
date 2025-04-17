import os
import torch
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
from model import Model
from main_utils import num_class, input_length, parse_the_args, read_data, read_mask_set
from sklearn.metrics import precision_recall_fscore_support
from captum.attr import (
    Saliency,
    FeatureAblation
)
                                                                                                                                                                                                                                                                                    

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, label, keys=None):
        super(Dataset, self).__init__()
        self.x = x
        self.label = label
        self.keys = keys

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.keys is not None:
            return self.x[idx], self.label[idx], self.keys[idx]
        else:
            return self.x[idx], self.label[idx]


def paired_collate_fn(insts):
    if len(list(zip(*insts))) == 2:
        x, label = list(zip(*insts))
        x = torch.FloatTensor(np.array(x))
        label = torch.LongTensor(np.array(label))
        return x, label
    elif len(list(zip(*insts))) == 3:
        x, label, key = list(zip(*insts))
        x = torch.FloatTensor(np.array(x))
        label = torch.LongTensor(np.array(label))
        key = np.array(key)
        return x, label, key


# retrain base models with explanation supervision
def train_with_exp_supervision(train_path, val_path, train_mask_path):
    f = open('results/results_with_exp_supervision.txt', 'w')
    f.write('train_l train_pre train_recall train_f1 val_l val_pre val_recall val_f1\n')
    f.flush()

    model = Model(num_class=num_class[args.dataset], input_len=input_length[args.dataset]).cuda()
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    criterion = nn.CrossEntropyLoss()

    x_train, y_train, keys_train = read_data(train_path, args.dataset)
    x_val, y_val, keys_val = read_data(val_path, args.dataset)

    train_loader = torch.utils.data.DataLoader(
        Dataset(x=x_train, label=y_train, keys=keys_train),
        collate_fn=paired_collate_fn,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        Dataset(x=x_val, label=y_val),
        collate_fn=paired_collate_fn,
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    mask_set = read_mask_set(train_mask_path)
    
    print('================= training with explanation supervision =================')
    for epoch in trange(args.epochs, mininterval=2, desc='Training Process ', leave=False):
        # train
        model.train()

        total_losses = []
        total_pred = []
        total_label = []

        for idx, (x_batch, label_batch, keys_batch) in enumerate(train_loader):
            x_batch, label_batch = x_batch.cuda(), label_batch.cuda()

            pred_batch = model(x_batch)
            task_loss = criterion(pred_batch, label_batch)

            exp_losses = []
            for each_x, each_label, each_key in zip(x_batch, label_batch, keys_batch):
                if each_key in mask_set.keys():
                    exp_mask = torch.FloatTensor(np.array(mask_set[each_key])).cuda()

                    each_x.requires_grad = True
                    if args.exp_method == 'inputgrad':
                        pred_logits = model(each_x.unsqueeze(0))
                        model_exp = torch.autograd.grad(pred_logits[0][each_label], each_x, create_graph = True)[0]
                    elif args.exp_method == 'erasure':
                        saliency = FeatureAblation(model)
                        model_exp = saliency.attribute(each_x, target=each_label)

                    # EBPG explanation loss
                    model_exp_pos = model_exp[model_exp > 0]
                    model_exp_masked_pos = model_exp * exp_mask
                    model_exp_masked_pos = model_exp_masked_pos[model_exp_masked_pos > 0]
                    if model_exp_pos.sum() == 0:
                        continue
                    exp_loss = model_exp_masked_pos.sum() / model_exp_pos.sum()
                    exp_loss = -exp_loss
                    exp_losses.append(exp_loss)

            if exp_losses:
                exp_losses = torch.stack(exp_losses)
                exp_losses = exp_losses.mean()

                total_loss = task_loss + args.exp_lambda * exp_losses
            else:
                total_loss = task_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            pred_batch = F.softmax(pred_batch, dim=-1).max(1)[1]
            total_pred.extend(pred_batch.long().tolist())
            total_label.extend(label_batch.tolist())
            total_losses.append(total_loss.item())

        train_loss = sum(total_losses)/len(total_losses)
        train_pre, train_recall, train_f1, _ = precision_recall_fscore_support(total_label, total_pred, average='macro', zero_division=1)

        scheduler.step()

        # validation
        model.eval()

        total_loss = []
        total_pred = []

        for idx, (x_batch, label_batch) in enumerate(val_loader):
            x_batch, label_batch = x_batch.cuda(), label_batch.cuda()

            pred_batch = model(x_batch)
            loss_batch = criterion(pred_batch, label_batch)

            pred_batch = F.softmax(pred_batch, dim=-1).max(1)[1]
            total_pred.extend(pred_batch.long().tolist())
            total_loss.append(loss_batch.item())

        val_loss = sum(total_loss)/len(total_loss)
        val_pre, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, total_pred, average='macro', zero_division=1)

        f.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' %(train_loss, train_pre, train_recall, train_f1, val_loss, val_pre, val_recall, val_f1))
        f.flush()
    f.close()


# train base models without explanation supervision
def train_model(train_path, val_path):
    f = open('results/results_baseline.txt', 'w')
    f.write('train_l train_pre train_recall train_f1 val_l val_pre val_recall val_f1\n')
    f.flush()
    
    model = Model(num_class=num_class[args.dataset], input_len=input_length[args.dataset]).cuda()
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
    criterion = nn.CrossEntropyLoss()

    x_train, y_train, keys_train = read_data(train_path, args.dataset)
    x_val, y_val, keys_val = read_data(val_path, args.dataset)

    train_loader = torch.utils.data.DataLoader(
        Dataset(x=x_train, label=y_train),
        collate_fn=paired_collate_fn,
        batch_size=args.train_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        Dataset(x=x_val, label=y_val),
        collate_fn=paired_collate_fn,
        batch_size=args.val_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print('================= training base model =================')
    for epoch in trange(args.epochs, mininterval=2, desc='Training Process ', leave=False):
        # train
        model.train()

        total_loss = []
        total_pred = []
        total_label = []

        for idx, (x_batch, label_batch) in enumerate(train_loader):
            x_batch, label_batch = x_batch.cuda(), label_batch.cuda()
            
            pred_batch = model(x_batch)
            loss_batch = criterion(pred_batch, label_batch)
            
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

            pred_batch = F.softmax(pred_batch, dim=-1).max(1)[1]
            total_pred.extend(pred_batch.long().tolist())
            total_label.extend(label_batch.tolist())
            total_loss.append(loss_batch.item())

        train_loss = sum(total_loss)/len(total_loss)
        train_pre, train_recall, train_f1, _ = precision_recall_fscore_support(total_label, total_pred, average='macro',zero_division=1)

        scheduler.step()
        
        # validation
        model.eval()

        total_loss = []
        total_pred = []
        for idx, (x_batch, label_batch) in enumerate(val_loader):
            x_batch, label_batch = x_batch.cuda(), label_batch.cuda()

            pred_batch = model(x_batch)
            loss_batch = criterion(pred_batch, label_batch)
        
            pred_batch = F.softmax(pred_batch, dim=-1).max(1)[1]
            total_pred.extend(pred_batch.long().tolist())
            total_loss.append(loss_batch.item())

        val_loss = sum(total_loss)/len(total_loss)
        val_pre, val_recall, val_f1, _ = precision_recall_fscore_support(y_val, total_pred, average='macro',zero_division=1)

        f.write('%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f\n' %(train_loss, train_pre, train_recall, train_f1, val_loss, val_pre, val_recall, val_f1))
        f.flush()
    f.close()


# test model
def test_model(test_path, test_mask_path, model_name):
    model = Model(num_class=num_class[args.dataset], input_len=input_length[args.dataset]).cuda()
    model.load_state_dict(torch.load(model_name))
    model.eval()

    saliency = Saliency(model)
    criterion = nn.CrossEntropyLoss()

    x_test, y_test, keys_test = read_data(test_path, args.dataset)
    test_loader = torch.utils.data.DataLoader(
        Dataset(x=x_test, label=y_test, keys=keys_test),
        collate_fn=paired_collate_fn,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    mask_set = read_mask_set(test_mask_path)

    total_loss = []
    total_pred = []
    ious = []

    for batch in tqdm(test_loader, mininterval=2, desc='Testing Process ', leave=False):
        x_batch, label_batch, keys_batch = batch
        x_batch, label_batch = x_batch.cuda(), label_batch.cuda()

        pred_batch = model(x_batch)
        loss_batch = criterion(pred_batch, label_batch)

        pred_batch = F.softmax(pred_batch, dim=-1).max(1)[1]
        total_pred.extend(pred_batch.long().tolist())
        total_loss.append(loss_batch.item())

        for each_x, each_label, each_key in zip(x_batch, label_batch, keys_batch):
            if each_key in mask_set.keys():
                exp_mask = torch.FloatTensor(np.array(mask_set[each_key])).cuda()

                model_exp = saliency.attribute(each_x, each_label, abs=False)
                model_exp = model_exp.view(input_length[args.dataset])

                # min-max
                model_exp = model_exp - torch.min(model_exp)
                model_exp = model_exp / torch.max(model_exp)

                # binarize
                model_exp = torch.where(model_exp > 0.5, torch.tensor(1.).cuda(), torch.tensor(0.).cuda())

                # iou
                intersection = torch.logical_and(model_exp, exp_mask)
                union = torch.logical_or(model_exp, exp_mask)
                iou = intersection.sum() / union.sum()
                ious.append(iou.cpu().detach())

    test_loss = sum(total_loss)/len(total_loss)
    test_pre, test_recall, test_f1, _ = precision_recall_fscore_support(y_test, total_pred, average='macro')

    print('  Test loss: %f' %test_loss)
    print('  Precision: %f' %test_pre)
    print('  Recall: %f' %test_recall)
    print('  F1-score: %f' %test_f1)
    print('  IoU %f' %np.array(ious).mean())

args = parse_the_args()

if __name__ == '__main__':
    train_path = os.path.join('datasets', args.dataset, 'train_data.csv')
    val_path = os.path.join('datasets', args.dataset, 'val_data.csv')
    test_path = os.path.join('datasets', args.dataset, 'test_data.csv')

    train_mask_path = os.path.join('datasets', args.dataset, 'train_mask.csv')
    test_mask_path = os.path.join('datasets', args.dataset, 'test_mask.csv')

    if args.train_model:
        # train base models without explanation supervision
        train_model(train_path, val_path)
    elif args.train_with_exp_loss:
        # retrain base models with explanation supervision
        train_with_exp_supervision(train_path, val_path, train_mask_path)
    elif args.test_model:
        # test model
        model_name = os.path.join('models', args.dataset, 'model_demo.pth')
        test_model(test_path, test_mask_path, model_name)
    else:
        print('Please specify the operation you want to perform!')
        exit(0)



