import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", **kwargs):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = None
        self.prelu_weight = None
        match activation:
            case "relu":
                self.activation = F.relu
            case "gelu":
                self.activation = F.gelu
            case "relu6":
                self.activation = F.relu6
            case "elu":
                self.activation = F.elu
            case "selu":
                self.activation = F.selu
            case "celu":
                self.activation = F.celu
            case "leaky_relu":
                self.activation = F.leaky_relu
            case "prelu":
                self.prelu_weight = nn.Parameter(torch.Tensor(d_ff))
                nn.init.constant_(self.prelu_weight, 0.25)
                # self.activation = F.prelu
            case "rrelu":
                self.activation = F.rrelu
            case "glu":
                self.activation = F.glu
            case _:
                raise NotImplementedError(
                    f"Activation function '{activation}' is not implemented"
                )

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta,
            **kwargs
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.conv1(y.transpose(-1, 1))

        # Apply activation function
        if self.prelu_weight is not None:
            y = F.prelu(y, self.prelu_weight)
        elif self.activation == F.glu:
            # Ensure the dimension size is even for GLU
            # if y.size(1) % 2 != 0:
            #     y = F.pad(y, (0, 0, 0, 1), "constant", 0)  # Pad to make the size even
            if y.size(2) % 2 != 0:
                y = F.pad(y, (0, 1), "constant", 0)  # Pad to make the size even
            y = F.glu(y)
        else:
            y = self.activation(y)

        y = self.dropout(y)
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # Ensure the dimensions match before addition
        if x.size(2) != y.size(2):
            y = F.interpolate(y, size=x.size(2), mode="nearest")

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, **kwargs)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
