from ogb.graphproppred import Evaluator
import time
import torch

from nn.GNNs import *
# from Trainer.loss_func import *

def get_optimizer(name, params, lr, lamda=0):
    if name == 'sgd':
        return torch.optim.SGD(params, lr=lr, weight_decay=lamda, momentum=0.9)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(params, lr=lr, weight_decay=lamda)
    elif name == 'adagrad':
        return torch.optim.Adagrad(params, lr=lr, weight_decay=lamda)
    elif name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=lamda)
    elif name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=lamda)
    elif name == 'adamax':
        return torch.optim.Adamax(params, lr=lr, weight_decay=lamda)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))

def get_scheduler(optimizer, name, epochs):
    if name == 'Mstep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//3*2, epochs//4*3], gamma = 0.1)
    elif name == 'Expo':
        gamma = (1e-6)**(1.0/epochs)
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = gamma, last_epoch=-1)
    elif name == 'Cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs, eta_min = 1e-16, last_epoch=-1)
    else:
        raise ValueError('Wrong LR schedule!!')

class Trainer_ogbg(torch.nn.Module):
    def __init__(self, args):
        super(Trainer_ogbg, self).__init__()
        self.args = args
        self.net = GNNs(args).to(self.args.device)
        self.optimizer = get_optimizer(
            args.optimizer,
            self.net.parameters(),
            args.lr,
            args.lamda
        )

        self.scheduler = get_scheduler(self.optimizer, args.lrschedular, args.epochs)

        if "classification" in args.task_type:
            self.criterion = torch.nn.BCEWithLogitsLoss()
        else:
            self.criterion = torch.nn.MSELoss()
        self.evaluator = Evaluator(args.dataset)

    def update(self, dataset):
        loss_sum, acc_sum, counts = 0., 0., 0.
        labels, preds = [], []
        for k in range(dataset.batch['train']):
            self.train()
            self.optimizer.zero_grad()
            self.zero_grad()

            data = dataset.sample('train')
            if data.x.size(0) == 1 or data.batch.size(0) == 1:
                pass
            else:
                x = self.net(data)[0]
                is_labeled = data.y == data.y
                loss = self.criterion(x[is_labeled], data.y[is_labeled].to(torch.float))
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()
                labels.append(data.y.cpu())
                preds.append(x.detach().cpu())
        loss_sum /= dataset.batch['train']
        labels = torch.cat(labels)
        preds = torch.cat(preds)
        acc = self.evaluator.eval({'y_true': labels, 'y_pred': preds})[dataset.eval_metric]
        torch.cuda.empty_cache()
        return {'loss': loss_sum, 'acc': acc.item()}

    @torch.no_grad()
    def evaluation(self, dataset, phase):
        logits, labels = [], []
        for i in range(dataset.batch[phase]):
            self.eval()
            data = dataset.sample(phase)
            logits.append(self.net(data)[0].cpu())
            labels.append(data.y.cpu())
            self.zero_grad()

        logits = torch.cat(logits)
        labels = torch.cat(labels)
        is_labeled = labels == labels
        loss = self.criterion(logits[is_labeled], labels[is_labeled].to(torch.float))
        acc = self.evaluator.eval({'y_true': labels, 'y_pred': logits})[dataset.eval_metric]
        torch.cuda.empty_cache()
        return {'acc': acc.item(), 'loss': loss.item()}

    @torch.no_grad()
    def node_embedding(self, dataset):
        splits = torch.arange(dataset.num_graphs).split(self.args.bsize)
        node_emb = []
        for gids in splits:
            self.eval()
            data = dataset.sample('test', gids)
            node_emb.append(self.net(data)[1].cpu())
            self.zero_grad()
        node_emb = torch.cat(node_emb).numpy()
        return node_emb