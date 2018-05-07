
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

class Epsilon(object):

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay

    def __call__(self, t):
        return self.end + max(0, 1 - t / self.decay) * (self.start - self.end)

