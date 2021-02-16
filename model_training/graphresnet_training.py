import os
import sys
from datetime import datetime
import torch
import random
import logging
logging.basicConfig()

sys.path.append(os.getcwd())
from graphnets_config import config
from models.graph_models import Graph_resnet, GATNet, Graph_resnet_GAT
from utils.graph_utils import create_adj_matrix, blockify_A, create_coordinate_channel, \
    create_edge_index_from_adjacency_matrix
from utils.training_gcn_utils import trainNet
from utils.videoloader import trafic4cast_dataset

if __name__ == "__main__":
    device = torch.device(config['device_num'])

    dataset_train = trafic4cast_dataset(split_type='training', **config['dataset'],
                                        reduce=True, filter_test_times=True)
    dataset_val = trafic4cast_dataset(split_type='validation', **config['dataset'],
                                      reduce=True, filter_test_times=True)

    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True,
                                               **config['dataloader'])
    val_loader = torch.utils.data.DataLoader(dataset_val, shuffle=True,
                                             **config['dataloader'])

    batch_size = config['dataloader']['batch_size']
    n_features = config['n_features']

    coords = create_coordinate_channel(b=batch_size)

    adj, nn_ixs, G, mask = create_adj_matrix(city=config['dataset']['cities'][0],
                                             mask_threshold=config['mask_threshold'])
    if batch_size > 1:
        adj = blockify_A(adj, batch_size)

    edge_index = create_edge_index_from_adjacency_matrix(adj)

    edge_index = edge_index.to(device)

    nb_of_models = config['nb_of_models']

    for i in range(nb_of_models):

        # # set parameters as used in the paper:
        # nh = 60
        # heads = 4
        # p = 0.6
      

        # # Print all of the hyper parameters of the training iteration:
        # print("===== HYPERPARAMETERS =====")
        # print("batch_size=", batch_size)
        # print("epochs =", config['num_epochs'])
        # print("learning_rate =", config['optimizer']['lr'])
        # print("mask_threshold =", config['mask_threshold'])
        # print("nh =", nh)
        # print("heads =", heads)
        # print("p =", p)
        # print("=" * 30)

        # log_folder = config['log_folder']
        # log_dir = log_folder + 'GATNet' \
        #           + '_nh=' + str(nh) \
        #           + '_heads=' + str(heads) \
        #           + '_p=' + str(p) \
        #           + '_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-") \
        #           + '-'.join(config['dataset']['cities'])

        # config['model'] = {"in_channels": 38,
        #                    "GATNet": {"nh": nh, "heads": heads, "p": p, "n_classes": 9, "n_input": n_features}}

                # set parameters as used in the paper:
        nh = 20
        depth = 4
        K = 4
        K_mix = 2
        inout_skipconn = True
        p = 0
        bn = True

        # Print all of the hyper parameters of the training iteration:
        print("===== HYPERPARAMETERS =====")
        print("batch_size=", batch_size)
        print("epochs =", config['num_epochs'])
        print("learning_rate =", config['optimizer']['lr'])
        print("mask_threshold =", config['mask_threshold'])
        print("nh =", nh)
        print("depth =", depth)
        print("K =", K)
        print("K_mix =", K_mix)
        print("inout_skipconn =", inout_skipconn)
        print("p =", p)
        print("bn =", bn)
        print("=" * 30)

        log_folder = config['log_folder']
        log_dir = log_folder + 'KipfNetresd2_GAT' + '_depth=' + str(depth) \
                  + '_nh=' + str(nh) \
                  + '_K=' + str(K) + '_Kmix=' + str(K_mix) \
                  + '_skipconn' + str(inout_skipconn) \
                  + '_p=' + str(p) \
                  + '_bn=' + str(bn) \
                  + '_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-") \
                  + '-'.join(config['dataset']['cities'])


        # model = GATNet(n_input=n_features, n_classes=9, n_hidden=8, heads=8, p=0.6).to(device)

        model = Graph_resnet_GAT(num_features=n_features, num_classes=9, nh=nh, depth=depth, K=K, K_mix=K_mix,
                              inout_skipconn=inout_skipconn, p=p, bn=bn).to(device)



        #try:
        trainNet(model, train_loader, val_loader, device,
                     adj, nn_ixs, edge_index, coords=coords, config=config, log_dir=log_dir)

        #except RuntimeError:
        #    print('Out of Memory error for ', nh, heads, p)

        del model
        # torch.cuda.empty_cache()
