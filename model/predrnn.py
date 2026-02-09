import torch
import torch.nn as nn

class PredRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim

        self.conv_xh = nn.Conv2d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)
        self.conv_m = nn.Conv2d(hidden_dim, 3 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x, h_prev, c_prev, m_prev):
        combined = torch.cat([x, h_prev], dim=1)
        i, f, g, o = torch.chunk(self.conv_xh(combined), 4, dim=1)

        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g

        mi, mf, mg = torch.chunk(self.conv_m(m_prev), 3, dim=1)
        mi, mf = torch.sigmoid(mi), torch.sigmoid(mf)
        mg = torch.tanh(mg)

        m = mf * m_prev + mi * mg
        h = o * torch.tanh(c + m)
        return h, c, m


class PredRNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, kernel_size, num_layers, out_steps):
        super().__init__()
        self.out_steps = out_steps

        self.cells = nn.ModuleList([
            PredRNNCell(input_dim if i == 0 else hidden_dims[i-1],
                        hidden_dims[i], kernel_size)
            for i in range(num_layers)
        ])

        self.conv_out = nn.Conv2d(hidden_dims[-1], input_dim, 1)

    def forward(self, x):
        B, Tin, C, H, W = x.shape
        device = x.device

        h = [torch.zeros(B, cell.hidden_dim, H, W, device=device) for cell in self.cells]
        c = [torch.zeros_like(h_i) for h_i in h]
        m = [torch.zeros_like(h_i) for h_i in h]

        prev = x[:, -1]
        outputs = []

        for _ in range(self.out_steps):
            inp = prev
            for i, cell in enumerate(self.cells):
                h[i], c[i], m[i] = cell(inp, h[i], c[i], m[i])
                inp = h[i]
            frame = self.conv_out(inp)
            outputs.append(frame)
            prev = frame

        return torch.stack(outputs, dim=1)
