import torch
import torch.nn as nn


class EncoderMatchingSAE(nn.Module):
    def __init__(self, arguments):
        super(EncoderMatchingSAE, self).__init__()
        self.arguments = arguments
        self.fcA1 = nn.Linear(arguments.input_A_size, arguments.hidden_A_size)
        self.fcA1_ = nn.Linear(arguments.hidden_A_size, arguments.input_A_size)
        self.fcA2 = nn.Linear(arguments.hidden_A_size, arguments.encoding_size)
        self.fcA2_ = nn.Linear(arguments.encoding_size, arguments.hidden_A_size)
        self.fcB1 = nn.Linear(arguments.input_B_size, arguments.hidden_B_size)
        self.fcB1_ = nn.Linear(arguments.hidden_B_size, arguments.input_B_size)
        self.fcB2 = nn.Linear(arguments.hidden_B_size, arguments.encoding_size)
        self.fcB2_ = nn.Linear(arguments.encoding_size, arguments.hidden_B_size)

    def encode(self, x):
        a = x[:, self.arguments.party_A_idc]
        a = self.fcA1(a)
        a = torch.relu(a)
        a = self.fcA2(a)
        a = torch.relu(a)
        b = x[:, self.arguments.party_B_idc]
        b = self.fcB1(b)
        b = torch.relu(b)
        b = self.fcB2(b)
        b = torch.relu(b)
        return a, b

    def decode(self, a, b):
        a = self.fcA2_(a)
        a = torch.relu(a)
        a = self.fcA1_(a)
        a = torch.relu(a)
        b = self.fcB2_(b)
        b = torch.relu(b)
        b = self.fcB1_(b)
        b = torch.relu(b)
        x = torch.cat([a, b], -1)
        return x

    def forward(self, x):
        a, b = self.encode(x)
        x_rec = self.decode(a, b)
        return a, b, x_rec


class NNAutoEncoder(nn.Module):
    def __init__(self, arguments, input_size):
        super(NNAutoEncoder, self).__init__()
        self.arguments = arguments
        self.fc1 = nn.Linear(input_size, arguments.hidden_A_size)
        self.fc1_ = nn.Linear(arguments.hidden_A_size, input_size)
        self.fc2 = nn.Linear(arguments.hidden_A_size, arguments.encoding_size)
        self.fc2_ = nn.Linear(arguments.encoding_size, arguments.hidden_A_size)

    def encode(self, x, req_grad=False):
        if req_grad:
            h = self.fc1(x)
            h = torch.relu(h)
            z = self.fc2(h)
        else:
            with torch.no_grad():
                h = self.fc1(x)
                h = torch.relu(h)
                z = self.fc2(h)
        return z

    def decode(self, z):
        h = self.fc2_(z)
        h = torch.relu(h)
        x = self.fc1_(h)
        return x

    def forward(self, x):
        e = self.encode(x, True)
        x_rec = self.decode(e)
        return x_rec


class NNEncoder(nn.Module):
    def __init__(self, arguments, input_size, output_size, hidden_size=None):
        super(NNEncoder, self).__init__()
        self.arguments = arguments
        if hidden_size is None:
            self.fc1 = nn.Linear(input_size, arguments.hidden_A_size)
            self.fc2 = nn.Linear(arguments.hidden_A_size, output_size)
        else:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)

    def encode(self, x, req_grad=False):
        if req_grad:
            h = self.fc1(x)
            h = torch.relu(h)
            h = self.fc2(h)
            h = torch.relu(h)
            z = h
        else:
            with torch.no_grad():
                h = self.fc1(x)
                h = torch.relu(h)
                h = self.fc2(h)
                h = torch.relu(h)
                z = h
        return z

    def forward(self, x):
        z = self.encode(x, True)
        return z


class NNDisturber(nn.Module):
    def __init__(self, arguments, input_size, output_size):
        super(NNDisturber, self).__init__()
        self.arguments = arguments
        self.fc1 = nn.Linear(input_size, arguments.hidden_A_size)
        self.fc2 = nn.Linear(arguments.hidden_A_size, output_size)

    def __encode(self, x, req_grad=False):
        if req_grad:
            h = self.fc1(x)
            h = torch.relu(h)
            z = self.fc2(h)
        else:
            with torch.no_grad():
                h = self.fc1(x)
                h = torch.relu(h)
                z = self.fc2(h)
        return z

    def forward(self, x):
        z = self.__encode(x, True)
        s = 2 * torch.sigmoid(z) - 1
        return s

    def generate(self, x):
        z = self.__encode(x)
        s = 2 * torch.sigmoid(z) - 1
        return torch.round(s)


class NNClassifier(nn.Module):
    def __init__(self, arguments, input_size):
        super(NNClassifier, self).__init__()
        hidden_size = arguments.classifier_hidden_size
        output_size = arguments.num_classes
        depth = arguments.classifier_depth
        model = nn.Sequential()
        if depth < 1:
            model.add_module('input_to_linear0', nn.Linear(input_size, output_size))
            model.add_module('Sigmoid{}'.format(0), nn.Sigmoid())
        else:
            model.add_module('input_to_linear0', nn.Linear(input_size, hidden_size))
            model.add_module('ReLU{}'.format(0), nn.ReLU())
            for d in range(depth-1):
                model.add_module('linear{}_to_linear{}'.format(d, d + 1), nn.Linear(hidden_size, hidden_size))
                model.add_module('ReLU{}'.format(d+1), nn.ReLU())
            model.add_module('linear{}_to_output'.format(depth), nn.Linear(hidden_size, output_size))

        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out

    def predict(self, x):
        with torch.no_grad():
            c = self.forward(x)
            _, predicted = torch.max(c.data, 1)
        return predicted


class EncoderMatchingNNBaseline(nn.Module):

    def __init__(self, arguments):
        super(EncoderMatchingNNBaseline, self).__init__()
        self.arguments = arguments
        self.fc1 = nn.Linear(arguments.input_A_size, arguments.hidden_size)
        self.fc2 = nn.Linear(arguments.hidden_size, arguments.num_classes)

    def forward(self, x):
        out = self.fc1(x[:, self.arguments.party_A_idc])
        out = torch.relu(out)
        out = self.fc2(out)
        return out

    def predict(self, x):
        with torch.no_grad():
            c = self.forward(x)
            _, predicted = torch.max(c.data, 1)
        return predicted


class EncoderMatchingNNBaselineHigh(nn.Module):

    def __init__(self, arguments):
        super(EncoderMatchingNNBaselineHigh, self).__init__()
        self.arguments = arguments
        self.fc1 = nn.Linear(arguments.num_features, arguments.hidden_size)
        self.fc2 = nn.Linear(arguments.hidden_size, arguments.num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.relu(out)
        out = self.fc2(out)
        return out

    def predict(self, x):
        with torch.no_grad():
            c = self.forward(x)
            _, predicted = torch.max(c.data, 1)
        return predicted
