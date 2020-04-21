import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
from collections import defaultdict
from load_mslr import get_time
from metrics import NDCG
from utils import (
    # eval_cross_entropy_loss,
    # eval_ndcg_at_k,
    get_device,
    get_ckptdir,
    init_weights,
    load_train_vali_data,
    get_args_parser,
    save_to_ckpt,
)


class PointwiseRankNet(nn.Module):
    def __init__(self, net_structures, double_precision=False):
        """
        :param net_structures: list of int for RankNet FC width
        """
        super(PointwiseRankNet, self).__init__()
        self.fc_layers = len(net_structures)
        for i in range(len(net_structures) - 1):
            layer = nn.Linear(net_structures[i], net_structures[i+1])
            if double_precision:
                layer = layer.double()
            setattr(self, 'fc' + str(i + 1), layer)

        last_layer = nn.Linear(net_structures[-1], 1)
        if double_precision:
            last_layer = last_layer.double()
        setattr(self, 'fc' + str(len(net_structures)), last_layer)
        self.criterion = nn.MSELoss()

    def forward(self, input1):
        for i in range(1, self.fc_layers):
            fc = getattr(self, 'fc' + str(i))
            input1 = F.relu(fc(input1))

        fc = getattr(self, 'fc' + str(self.fc_layers))
        output1 = fc(input1)
        return torch.sigmoid(output1)

    def calculate_loss(self, output1, label1):
        loss = self.criterion(output1, label1)
        return loss

    def dump_param(self):
        for i in range(1, self.fc_layers + 1):
            print("fc{} layers".format(i))
            fc = getattr(self, 'fc' + str(i))

            with torch.no_grad():
                weight_norm, weight_grad_norm = torch.norm(
                    fc.weight).item(), torch.norm(fc.weight.grad).item()
                bias_norm, bias_grad_norm = torch.norm(
                    fc.bias).item(), torch.norm(fc.bias.grad).item()
            try:
                weight_ratio = weight_grad_norm / \
                    weight_norm if weight_norm else float(
                        'inf') if weight_grad_norm else 0.0
                bias_ratio = bias_grad_norm / \
                    bias_norm if bias_norm else float(
                        'inf') if bias_grad_norm else 0.0
            except Exception:
                import ipdb
                ipdb.set_trace()

            print(
                '\tweight norm {:.4e}'.format(
                    weight_norm), ', grad norm {:.4e}'.format(weight_grad_norm),
                ', ratio {:.4e}'.format(weight_ratio),
                'weight type {}, weight grad type {}'.format(fc.weight.type(), fc.weight.grad.type()))
            print(
                '\tbias norm {:.4e}'.format(
                    bias_norm), ', grad norm {:.4e}'.format(bias_grad_norm),
                ', ratio {:.4e}'.format(bias_ratio),
                'bias type {}, bias grad type {}'.format(
                    fc.bias.type(), fc.bias.grad.type())
            )

#############################
# Train RankNet with Different Algorithms
#############################


def train_rank_net(
        start_epoch=0, additional_epoch=100, lr=0.0001, optim="adam",
        train_algo='baseline',
        double_precision=False, standardize=False,
        small_dataset=False, debug=False,
        output_dir="/tmp/ranking_output/"):
    """

    :param start_epoch: int
    :param additional_epoch: int
    :param lr: float
    :param optim: str
    :param train_algo: str
    :param double_precision: boolean
    :param standardize: boolean
    :param small_dataset: boolean
    :param debug: boolean
    :return:
    """
    print("start_epoch:{}, additional_epoch:{}, lr:{}".format(
        start_epoch, additional_epoch, lr))
    writer = SummaryWriter(output_dir)

    precision = torch.float64 if double_precision else torch.float32

    # get training and validation data:
    data_fold = 'Fold1'
    train_loader, df_train, valid_loader, df_valid = load_train_vali_data(
        data_fold, small_dataset)
    if standardize:
        df_train, scaler = train_loader.train_scaler_and_transform()
        df_valid = valid_loader.apply_scaler(scaler)

    net, ckptfile = get_train_inference_net(
        train_algo, train_loader.num_features, start_epoch, double_precision)
    device = get_device()
    net.to(device)

    # initialize to make training faster
    net.apply(init_weights)

    if optim == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif optim == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    print(optimizer)

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.75)

    losses = []

    for i in range(start_epoch, start_epoch + additional_epoch):

        net.zero_grad()
        net.train()

        if train_algo == 'baseline':
            epoch_loss = baseline_pairwise_training_loop(
                i, net, optimizer, train_loader,
                precision=precision, device=device, debug=debug)
        else:
            raise NotImplementedError()
        
        losses.append(epoch_loss)
        scheduler.step()
        print('=' * 20 + '\n', get_time(),
              'Epoch{}, loss : {}'.format(i, losses[-1]), '\n' + '=' * 20)

        # save to checkpoint every 5 step, and run eval
        if i % 5 == 0 and i != start_epoch:
            save_to_ckpt(ckptfile, i, net, optimizer, scheduler)
            net.load_state_dict(net.state_dict())
            eval_model(net, device, df_valid,
                       valid_loader, i, writer)

    # save the last ckpt
    save_to_ckpt(ckptfile, start_epoch + additional_epoch,
                 net, optimizer, scheduler)

    # final evaluation
    net.load_state_dict(net.state_dict())
    ndcg_result = eval_model(
        net, device, df_valid, valid_loader, start_epoch + additional_epoch, writer)

    # save the final model
    torch.save(net.state_dict(), ckptfile)
    print(
        get_time(),
        "finish training " + ", ".join(
            ["NDCG@{}: {:.5f}".format(k, ndcg_result[k]) for k in ndcg_result]
        ),
        '\n\n'
    )


def get_train_inference_net(train_algo, num_features, start_epoch, double_precision):
    ranknet_structure = [num_features*3, 128, 64]

    if train_algo == 'baseline':
        # inference always use single precision
        net = PointwiseRankNet(ranknet_structure)
        ckptfile = get_ckptdir('ranknet', ranknet_structure)
    else:
        raise NotImplementedError()

    if start_epoch != 0:
        load_from_ckpt(ckptfile, start_epoch, net)

    return net, ckptfile


def baseline_pairwise_training_loop(
        epoch, net, optimizer, train_loader, batch_size=100000,
        precision=torch.float32, device="cpu", debug=False):
    minibatch_loss = []
    minibatch = 0
    count = 0

    for x_i, y_i, x_j, y_j in train_loader.generate_query_pair_batch(batchsize=batch_size):
        if x_i is None or x_i.shape[0] == 0:
            continue

        x_i_concat, y_i_concat = np.concatenate(
            [x_i, x_j, x_i-x_j], axis=1), y_i-y_j
        x_j_concat, y_j_concat = np.concatenate(
            [x_j, x_i, x_j-x_i], axis=1), y_j-y_i

        inputs = np.concatenate([x_i_concat, x_j_concat], axis=0)
        labels = np.concatenate([y_i_concat, y_j_concat], axis=0)

        inputs = torch.tensor(inputs, dtype=precision, device=device)
        labels = torch.tensor(labels, dtype=precision, device=device)
        labels = torch.sigmoid(labels)
        # print(labels.min(), labels.max())

        net.zero_grad()
        outputs = net(inputs)
        loss = net.calculate_loss(outputs, labels)

        loss.backward()
        count += 1
        if count % 25 == 0 and debug:
            net.dump_param()
        optimizer.step()

        minibatch_loss.append(loss.item())

        minibatch += 1
        if minibatch % 100 == 0:
            print(get_time(), 'Epoch {}, Minibatch: {}, loss : {}'.format(
                epoch, minibatch, loss.item()))

    return np.mean(minibatch_loss)


def eval_model(inference_model, device, df_valid, valid_loader, epoch, writer=None):
    """
    :param torch.nn.Module inference_model:
    :param str device: cpu or cuda:id
    :param pandas.DataFrame df_valid:
    :param valid_loader:
    :param int epoch:
    :return:
    """
    inference_model.eval()  # Set model to evaluate mode
    batch_size = 1000000

    with torch.no_grad():
        eval_mse_loss(inference_model, device, valid_loader, epoch, writer)
        ndcg_result = eval_ndcg_at_k(inference_model, device, df_valid, valid_loader, [10, 30], epoch, writer)
    return ndcg_result


def load_from_ckpt(ckpt_file, epoch, model):
    ckpt_file = ckpt_file + '_{}'.format(epoch)
    if os.path.isfile(ckpt_file):
        print(get_time(), 'load from ckpt {}'.format(ckpt_file))
        ckpt_state_dict = torch.load(ckpt_file)
        model.load_state_dict(ckpt_state_dict['model_state_dict'])
        print(get_time(), 'finish load from ckpt {}'.format(ckpt_file))
    else:
        print('ckpt file does not exist {}'.format(ckpt_file))


def eval_mse_loss(model, device, loader, epoch, writer=None, phase="Eval", sigma=1.0):
    print(get_time(), "{} Phase evaluate pairwise mse loss".format(phase))
    model.eval()
    with torch.set_grad_enabled(False):
        total_cost = []
        pairs_in_compute = 0
        for X, Y in loader.generate_batch_per_query(loader.df):
            num_inputs = len(X)
            
            inputs = []
            labels = []
            num_pairs = 0
            for i in range(num_inputs-1):
                for j in range(i+1, num_inputs):
                    inputs_ = np.concatenate([X[i], X[j], X[i]-X[j]], axis=0)
                    labels_ = Y[i]-Y[j]
                    inputs.append(inputs_)
                    labels.append(labels_)
                    num_pairs += 1

            # skip negative sessions, no relevant info:
            if len(inputs)==0:
                continue
            inputs = np.stack(inputs)
            labels = np.stack(labels)
            labels = labels.reshape(-1,1).astype(np.float32)

            pairs_in_compute += num_pairs

            inputs_tensor = torch.tensor(inputs, device=device)
            labels_tensor = torch.tensor(labels, device=device)
            labels_tensor = torch.sigmoid(labels_tensor)
            outputs = model(inputs_tensor)
            loss = model.calculate_loss(outputs, labels_tensor)
            total_cost.append(loss.item())

    avg_cost = sum(total_cost)/len(total_cost)
    print(
        get_time(),
        "Epoch {}: {} Phase pairwise mse loss {:.6f}, total_paris {}".format(
            epoch, phase, avg_cost, pairs_in_compute
        ))
    if writer:
        writer.add_scalars('loss/mse_loss', {phase: avg_cost}, epoch)

def apply_solver(pairwise_scores, ofe_score_min_cap=0, ofe_score_max_cap=1):
    # num_docs = pairwise_scores.shape[0]
    ofe_tree_sum_clipped = pairwise_scores.clip(ofe_score_min_cap, ofe_score_max_cap)
    ofe_tree_sum_norm = 6 * (ofe_tree_sum_clipped-ofe_score_min_cap)/(ofe_score_max_cap-ofe_score_min_cap) - 3
    ofe_score = 1/(1+np.exp(-ofe_tree_sum_norm))
    ofe_score_diag = ofe_score.copy()
    np.fill_diagonal(ofe_score_diag, 0.5)
    scores = ofe_score_diag.sum(axis=1)/10
    return scores, ofe_score_diag, ofe_score 

def eval_ndcg_at_k(
        inference_model, device, df_valid, valid_loader, k_list, epoch,
        writer=None, phase="Eval"
):
    inference_model.eval()
    with torch.no_grad():
        print("Eval Phase evaluate NDCG @ {}".format(k_list))
        ndcg_metrics = {k: NDCG(k) for k in k_list}
        for X, Y in valid_loader.generate_batch_per_query(df_valid): 
            num_inputs = len(X)
            inputs = []
            labels = []
            num_pairs = 0
            for i in range(num_inputs-1):
                for j in range(i+1, num_inputs):
                    inputs_ = np.concatenate([X[i], X[j], X[i]-X[j]], axis=0)
                    labels_ = Y[i]-Y[j]
                    inputs.append(inputs_)
                    labels.append(labels_)
                    num_pairs += 1
            if len(inputs)==0:
                continue
            inputs = np.stack(inputs)
            inputs_tensor = torch.tensor(inputs, device=device)
            outputs = inference_model(inputs_tensor)

            score_matrix = np.eye(num_inputs) * 0.5
            cnt = 0
            for i in range(num_inputs-1):
                for j in range(i+1, num_inputs):
                    score_matrix[i, j] = outputs[cnt].item()
                    cnt += 1

            score, pairwise_prob_diag, pairwise_prob = apply_solver(score_matrix)
            relavance = Y.reshape(-1)

            session_ndcgs = defaultdict(list)
            result_df = pd.DataFrame({'relavance': relavance, 'score': score})
            result_df.sort_values('score', ascending=False)
            rel_rank = result_df.relavance.values
            for k, ndcg in ndcg_metrics.items():
                if ndcg.maxDCG(rel_rank) == 0:
                    continue
                ndcg_k = ndcg.evaluate(rel_rank)
                if not np.isnan(ndcg_k):
                    session_ndcgs[k].append(ndcg_k)

    ndcg_result = {k: np.mean(session_ndcgs[k]) for k in k_list}
    ndcg_result_print = ", ".join(
        ["NDCG@{}: {:.5f}".format(k, ndcg_result[k]) for k in k_list])
    print(get_time(), "{} Phase evaluate {}".format(phase, ndcg_result_print))
    if writer:
        for k in k_list:
            writer.add_scalars("metrics/NDCG@{}".format(k),
                               {phase: ndcg_result[k]}, epoch)
    return ndcg_result


if __name__ == "__main__":
    parser = get_args_parser()
    # add additional args for RankNet
    parser.add_argument("--train_algo", default='baseline')
    args = parser.parse_args()
    train_rank_net(
        args.start_epoch, args.additional_epoch, args.lr, args.optim,
        args.train_algo,
        args.double_precision, args.standardize,
        args.small_dataset, args.debug,
        output_dir=args.output_dir,
    )
