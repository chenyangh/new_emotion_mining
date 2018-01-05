class EarlyStop:
    def __init__(self, tolerance):
        self.epoch = 0
        self.loss_list = []
        self.tolerance = tolerance

    def new_loss(self, loss):
        self.loss_list.append(loss)

    def if_stop(self):
        tmp_list = self.loss_list[-self.tolerance:]
        if not sorted(tmp_list) == tmp_list:
            return True
        else:
            return False
