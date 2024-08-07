from typing import List
import copy
import operator
from enum import Enum, auto
import numpy as np

from torch.nn import Module


class StopVariable(Enum):
    LOSS = auto()
    ACCURACY = auto()
    NONE = auto()


class Best(Enum):
    RANKED = auto()
    ALL = auto()


stopping_args = dict(
        stop_varnames=[StopVariable.ACCURACY],
        patience=100, max_epochs=1000, remember=Best.RANKED)


class EarlyStopping:
    def __init__(
            self, model: Module, stop_varnames: List[StopVariable],
            patience: int = 10, max_epochs: int = 200, remember: Best = Best.ALL):
        self.model = model
        self.comp_ops = []
        self.stop_vars = []
        self.best_vals = []
        for stop_varname in stop_varnames:
            if stop_varname is StopVariable.LOSS:
                self.stop_vars.append('loss')
                self.comp_ops.append(operator.le)
                self.best_vals.append(np.inf)
            elif stop_varname is StopVariable.ACCURACY:
                self.stop_vars.append('acc')
                self.comp_ops.append(operator.ge)
                self.best_vals.append(-np.inf)
        self.remember = remember
        # self.remembered_vals = copy.copy(self.best_vals)
        self.remembered_vals = [-np.inf]
        self.max_patience = patience
        self.patience = self.max_patience
        self.max_epochs = max_epochs
        self.best_epoch = None
        self.best_state = None
        self.best_remmbered_test = [np.inf, -np.inf, -np.inf]

    def check(self, values: List[np.floating], epoch: int) -> bool:
        # ===  将目前最好的评价指标与当前输出的评价指标进行比较
        checks = [self.comp_ops[i](val, self.best_vals[i])
                  for i, val in enumerate(values[:1])]
        if any(checks):
            # ===  记录最好的评价指标
            self.best_vals = np.choose(checks, [self.best_vals, values[:1]])
            self.patience = self.max_patience

            comp_remembered = [
                    self.comp_ops[i](val, self.remembered_vals[i])
                    for i, val in enumerate(values[:1])]
            if self.remember is Best.ALL:
                # ===  要求当前的所有评价指标必须比历史最好的所有指标都要好
                if all(comp_remembered):
                    self.best_epoch = epoch
                    self.remembered_vals = copy.copy(values[:1])
                    # self.best_remmbered_test = copy.copy(values[2:])
                    self.best_state = {
                            key: value.cpu() for key, value
                            in self.model.state_dict().items()}
            elif self.remember is Best.RANKED:
                # ===  只要当前的评价指标有一项比历史最好的要好就可以更新
                for i, comp in enumerate(comp_remembered):
                    if comp:
                        # if not(self.remembered_vals[i] == values[i]):
                        self.best_epoch = epoch
                        self.remembered_vals = copy.copy(values[:1])
                        # self.best_remmbered_test = copy.copy(values[2:])
                        self.best_state = {
                                key: value.cpu() for key, value
                                in self.model.state_dict().items()}
                            # break
                    else:
                        break
        else:
            self.patience -= 1
        return self.patience == 0
