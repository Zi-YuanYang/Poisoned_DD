import torch
import torch.nn as nn
import torch.nn.functional as F


class MySigmoid(nn.Module):
    def __init__(self, alpha = 1.0):
        super(MySigmoid, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        out = torch.sigmoid(self.alpha * x)
        # print("MySigmoid output range:", out.min().item(), out.max().item())
        return torch.clamp(out, min=0, max=1)

class FingerprintEncoder(nn.Module):
    def __init__(self, fingerprint_size, mid_size, img_size, channel):
        super(FingerprintEncoder, self).__init__()
        self.linear_layer_1 = nn.Linear(fingerprint_size, mid_size)
        self.linear_layer_2 = nn.Linear(mid_size, img_size * img_size)
        self.channel_layer_weight = nn.Linear(fingerprint_size, channel)
        self.channel_layer_bias = nn.Linear(fingerprint_size, channel)

        self.relu = nn.Tanh()  
        self.channel = channel
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) 
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input):
        weight = self.channel_layer_weight(input)
        weight = self.relu(weight)
        bias = self.channel_layer_bias(input)
        bias = self.relu(bias)

        out = self.linear_layer_1(input)
        out = self.relu(out)

        out = self.linear_layer_2(out)
        out = self.relu(out)

        out_repeated = out.unsqueeze(1).repeat(1, self.channel, 1)
        out_repeated = out_repeated * weight.unsqueeze(-1)
        out_repeated = out_repeated + bias.unsqueeze(-1)

        out_repeated = torch.clamp(out_repeated, min=-10, max=10)
        return out_repeated

class FingerprintDecoder(nn.Module):
    def __init__(self, input_size, mid_size, fingerprint_size, alpha):
        super(FingerprintDecoder, self).__init__()
        self.linear_layer_1 = nn.Linear(input_size, mid_size)
        self.linear_layer_2 = nn.Linear(mid_size, fingerprint_size)
        self.linear_layer_3 = nn.Linear(mid_size, input_size)

        self.relu = nn.ReLU()
        self.out_act = MySigmoid(alpha)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)  # Xavier 初始化
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input):
        out = self.linear_layer_1(input)
        out = self.relu(out)

        finger = self.linear_layer_2(out)
        finger = self.out_act(finger)

        clean_out = self.linear_layer_3(out)
        clean_out = self.relu(clean_out)

        clean_out = torch.clamp(clean_out, min=0, max=10)
        finger = torch.clamp(finger, min=0, max=1)

        return clean_out, finger