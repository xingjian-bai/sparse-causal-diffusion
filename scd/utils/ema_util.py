import torch


class EMAModel:

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        self.original = None

    def step(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = self.decay * self.shadow[name].data + (1.0 - self.decay) * param.data

    def store(self, model):
        self.original = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

    def restore(self, model):
        if self.original is None:
            raise RuntimeError('Call store() before restore().')
        for name, param in model.named_parameters():
            if name in self.original:
                param.data.copy_(self.original[name].data)
        self.original = None

    @torch.no_grad()
    def copy_to(self, model):
        model_params = []
        shadow_params = []
        for name, param in model.named_parameters():
            model_params.append(param.data)
            shadow_params.append(self.shadow[name].data.to(param.device))

        torch._foreach_copy_(model_params, shadow_params)

    def state_dict(self):
        return {name: self.shadow[name].clone() for name in self.shadow}

    def load_state_dict(self, state_dict):
        if set(state_dict.keys()) != set(self.shadow.keys()):
            raise ValueError('state_dict keys do not match shadow keys.')

        for name, param in state_dict.items():
            self.shadow[name].data.copy_(param)
