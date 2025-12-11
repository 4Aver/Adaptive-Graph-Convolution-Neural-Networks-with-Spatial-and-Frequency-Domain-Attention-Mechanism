import torch

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode # 'min' or 'max'
        self.min_delta = min_delta #容许误差
        self.patience = patience #耐心次数
        self.best = None #最佳得分
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
                
if __name__ == '__main__':
    
    # es的用法如下：接受一个tensor序列，如果连续patience次都没有loss更小的，
    # 则es.step 设置为True，此时可以break               
    es = EarlyStopping(mode='min', min_delta=0.19, patience=4)
    x = torch.tensor([2,2,2,3,1.8,2.2,1.3,0.9])
    for i in x:
        print(i)
        if es.step(i):
            print('early stop')
            break
