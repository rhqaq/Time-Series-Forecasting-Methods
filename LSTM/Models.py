import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # (N, L, D)
        a = self.attn(x)  # (N, L, 1)
        x = (x * a).sum(dim=1)  # (N, D)
        return x


class LSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=5,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.attn = SelfAttentionEncoder(5)
        self.fc = nn.Linear(5, 1)

    def forward(self, inputs):
        # (N,L,D) batch,时序长度,特征数量
        tensor = self.LSTM(inputs)[0]  # (N,L,D)
        tensor = self.attn(tensor)  # (N,D)
        tensor = self.fc(tensor)
        return tensor.reshape(-1)


class Simple_LSTM(nn.Module):
    def __init__(self, input_size,hidden_size):
        super().__init__()
        self.LSTM = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, inputs):
        # (N,L,D) batch,时序长度,特征数量
        output, (hn, cn) = self.LSTM(inputs)  # (N,L,D)
        # print(hn.shape)
        # tensor = self.fc(torch.sigmoid(output[:, -1, :])) # bi-lstm
        tensor = self.fc(torch.sigmoid(hn[-1]))
        # return tensor.squeeze()
        return tensor.reshape(-1)


class TimeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, cuda_flag=False):
        # assumes that batch_first is always true
        super(TimeLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cuda_flag = cuda_flag
        self.W_all = nn.Linear(hidden_size, hidden_size * 4)
        self.U_all = nn.Linear(input_size, hidden_size * 4)
        self.W_d = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)  # 和LSTM一样最后加了个全连接层
    #         self.attention = Attention(hidden_size, hidden_size)

    def forward(self, inputs, time_interval):
        # inputs: [batch, timesteps, input_size]
        # time_interval: [batch, timesteps]
        # h: [batch, hidden_size]
        # c: [batch, hidden_size]
        b, seq, embed = inputs.size()
        h = torch.zeros(b, self.hidden_size, requires_grad=False)
        c = torch.zeros(b, self.hidden_size, requires_grad=False)
        if self.cuda_flag:
            h = h.cuda()
            c = c.cuda()
        outputs = []
        for s in range(seq):
            c_s1 = torch.tanh(self.W_d(c))
            c_s2 = c_s1 * time_interval[:, s:s + 1].expand_as(c_s1)
            c_l = c - c_s1
            c_adj = c_l + c_s2
            outs = self.W_all(h) + self.U_all(inputs[:, s])
            f, i, o, c_tmp = torch.chunk(outs, 4, 1)
            f = torch.sigmoid(f)
            i = torch.sigmoid(i)
            o = torch.sigmoid(o)
            c_tmp = torch.sigmoid(c_tmp)
            c = f * c_adj + i * c_tmp
            h = o * torch.tanh(c)
            outputs.append(self.out(h))

        #         output = outputs[-1]  #   (batch, hidden_size)
        output = torch.stack(outputs, -1).sum(-1).reshape(-1)

        return output

class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = self.fc2(tensor)
        return tensor


class SVM(nn.Module):
    """
    Using fully connected neural network to implement linear SVM and Logistic regression with hinge loss and
    cross-entropy loss which computes softmax internally, respectively.
    """

    def __init__(self, input_size, num_classes):
        super(SVM, self).__init__()  # Call the init function of nn.Module
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

