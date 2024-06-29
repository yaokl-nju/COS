import gc
import itertools
import joblib
import numpy as np
import os.path as osp
import time

from datasets.ogbg_dataset import ogbg_dataset
from parse_conf import *
from Trainer.GNN_Trainer_ogbg import *
from Trainer.early_stopping_ogb import *
warnings.filterwarnings("ignore")
##########################################
def write_to_file(args, curves, repeat):
    results = np.stack(curves, axis=1)
    mean = np.mean(results, axis=1)
    std = np.std(results, axis=1)
    result_str = 'cs {} - repeat {} - drop {} - {}\n'.format(str(args.cs), str(repeat), str(args.drop), args.dataset)
    for i, epoch in enumerate(range(args.eval_step, args.epochs + 1, args.eval_step)):
        result_str += 'epoch={:3d}, acc={:.4f} ± {:.4f}\n'.format(epoch, mean[i], std[i])
    file = open('curves/{}.txt'.format(args.dataset), mode = 'a')
    file.write(result_str + '\n')
    file.close()
    return result_str

dataset = ogbg_dataset(args, name=args.dataset, root=args.root)
args.task_type= dataset.task_type
args.num_tasks = dataset.num_tasks

setup_seed(args.seed)
lamdas = [args.lamda]
lrs = [args.lr]
drops = [args.drop]

results, curves = [], []
for drop, lr, lamda in itertools.product(drops, lrs, lamdas):
    args.lamda = lamda
    args.lr = lr
    args.drop = drop
    acc_te_i = []

    for i in range(args.repeat):
        t_total = time.time()
        loss_val_j, acc_val_j, curves_i = [], [], []

        model = Trainer_ogbg(args)
        early_stopping = EarlyStopping(model, **stopping_args)
        dataset.reset_iter()

        print("lamda={}, lr={}, drop={}, cs={}".format(str(lamda), str(lr), str(drop), str(args.cs)))

        train_time = []
        op_time = 0.0
        for j in range(args.epochs):
            start_time = time.time()
            temp_tr = model.update(dataset)
            temp_val = model.evaluation(dataset, 'valid')
            temp_te = model.evaluation(dataset, 'test')
            train_time.append(time.time() - start_time)

            if (j + 1) % args.eval_step == 0:
                print('epoch :{:4d}, loss:{:.4f}, acc_tr:{:.4f}, loss_v:{:.4f}, acc_v:{:.4f}'
                      .format(j, temp_tr['loss'], temp_tr['acc'], temp_val['loss'], temp_val['acc']),
                      'acc_t {:.4f}'.format(temp_te['acc']),
                      'time: {:.1f}s'.format(np.mean(train_time)),
                      'ratio: {}%'.format(str(int(dataset.ncount_new * 100 / dataset.ncount_old)))
                      )
                curves_i.append(temp_te['acc'])

            loss_val_j.append(temp_val['loss'])
            acc_val_j.append(temp_val['acc'])

            iter_results = {'loss': np.mean(loss_val_j[-5:]), 'acc': np.mean(acc_val_j[-5:])}
            if (j + 1) >= args.early_stopping:
                stop_vars = [iter_results[key] for key in early_stopping.stop_vars]
                if early_stopping.check(stop_vars, j):
                    break

            if (j + 1) % 5 == 0:
                gc.collect()

            if args.lrscheduler:
                model.scheduler.step()

        model.load_state_dict(early_stopping.best_state)
        start_time = time.time()
        temp_te = model.evaluation(dataset, 'test')
        best_val_j = early_stopping.remembered_vals
        print("lamda={}, lr={}, drop={}".format(str(lamda), str(lr), str(drop)))
        print("Test results:",
              "best epoch={}".format(str(early_stopping.best_epoch)),
              "acc_v={:.4f}".format(best_val_j[0]),
              "acc_t={:.4f}".format(temp_te['acc']),
              'time:{:.1f}s '.format(time.time() - start_time)
              )

        if args.save:
            node_emb = model.node_embedding(dataset)
            path = osp.join(dataset.root, 'processed/GINEmb_L{}.pt'.format(str(args.layer_num)))
            joblib.dump(node_emb, path, compress=3)
        del model
        gc.collect()

        acc_te_i.append(temp_te['acc'])
        print("repeat: {:4d}".format(i + 1), "Now test_acc mean={:.4f}, std={:.4f}".
              format(np.mean(acc_te_i), np.std(acc_te_i)))

        curves.append(np.array(curves_i))
        curves_str = write_to_file(args, curves, i + 1)

        if i == args.repeat - 1:
            print('\n' + curves_str)

    results.append([np.mean(acc_te_i), lamda, lr, drop, np.std(acc_te_i)])
    idx = np.argsort(np.array(results)[:, 0])[-1]
    print("\nBest acc={:.4f} std={:.4f} lamda={}, lr={}, drop={}, cs={}\n".
          format(results[idx][0], results[idx][-1], results[idx][1], results[idx][2], results[idx][3], str(args.cs)))

print(args)
msg = "lamda={:f} cs={}, test_acc={:.4f} test_std={:.4f}" \
    .format(results[0][1], str(args.cs), results[0][0], results[0][-1])
print(msg, '\n\n')