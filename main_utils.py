import argparse
import csv
import numpy as np

num_class = {'cicids2017':15,'ton_iot':10}
input_length = {'cicids2017':784,'ton_iot':920}

def parse_the_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_model', dest='train_model', action='store_true', help='Choose to train base models')
    parser.add_argument('--train_with_exp_loss', dest='train_with_exp_loss', action='store_true', help='Choose to retrain base models with explanation supervision')
    parser.add_argument('--test_model', dest='test_model', action='store_true', help='Choose to test models')
    parser.add_argument('--dataset', type=str, required=True, choices=['cicids2017', 'ton_iot'], help="Dataset to use")
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--train_batch', type=int, default=256, help='Batch size for training')
    parser.add_argument('--val_batch', type=int, default=256, help='Batch size for validation')
    parser.add_argument('--test_batch', type=int, default=256, help='Batch size for testing')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--exp_method', type=str, choices=['inputgrad', 'erasure'], help='Explanation method to use')
    parser.add_argument('--exp_lambda', type=float, default=0.5, help='Weight for explanation loss')
    args = parser.parse_args()

    return args


def read_data(file_path, dataset):
    print('Loading data %s' % file_path)
    if dataset == 'cicids2017':
        type = {'BENIGN':0, 'FTP-Patator':1, 'SSH-Patator':2, 'DoS GoldenEye':3, 'DoS Hulk':4, 'DoS Slowhttptest':5, 'DoS slowloris':6, 
            'Heartbleed':7, 'Web Attack - Brute Force':8, 'Web Attack - Sql Injection':9, 'Web Attack - XSS':10, 'Infiltration':11,
            'Bot':12, 'PortScan':13, 'DDoS':14}
        input_len = input_length['cicids2017']
    elif dataset == 'ton_iot':
        type = {'normal':0, 'backdoor':1, 'ddos':2, 'dos':3, 'injection':4, 'mitm':5, 'password':6, 
            'ransomware':7, 'scanning':8, 'xss':9}
        input_len = input_length['ton_iot']
    else:
        raise ValueError("Invalid dataset. Choose either 'cicids2017' or 'ton_iot'.")

    x = []
    y = []
    keys = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        head_row = next(reader)
        for row in reader:
            # use (timestamp, src_ip, dst_ip, src_port, dst_port) as key to identify the packet
            key = str(row[input_len])+'-'+str(row[input_len+1])+'-'+str(row[input_len+2])+'-'+str(row[input_len+3])+'-'+str(row[input_len+4])
            keys.append(key)
            x.append([int(row[i]) for i in range(input_len)])
            y.append(int(type[row[-1]]))
    f.close()

    return x, y, keys


def read_mask_set(mask_path):
    mask_set = {}
    with open(mask_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            mask_set[row[0]] = np.array(row[1:]).astype(np.float32)
    f.close()

    return mask_set