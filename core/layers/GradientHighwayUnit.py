import torch
import torch.nn as nn

class GHU(nn.Module):
    def __init__(self, layer_name, filter_size, num_features, input_channels, tln=False, initializer=0.001):
        super(GHU, self).__init__()
        self.layer_name = layer_name
        self.filter_size = filter_size
        self.num_features = num_features
        self.input_channels = input_channels
        self.tln = tln  # Tensor Layer Norm flag

        # State-to-state convolution: z -> 2*num_features
        self.state_conv = nn.Conv2d(
            num_features, 2*num_features, filter_size,
            padding=filter_size//2, bias=True  # 'same' padding approximation
        )
        # Input-to-state convolution: x -> 2*num_features
        self.input_conv = nn.Conv2d(
            input_channels, 2*num_features, filter_size,
            padding=filter_size//2, bias=True
        )

        # Initialize weights and biases
        if initializer != -1:
            nn.init.uniform_(self.state_conv.weight, -initializer, initializer)
            nn.init.uniform_(self.input_conv.weight, -initializer, initializer)
            nn.init.zeros_(self.state_conv.bias)  # Match TensorFlow default
            nn.init.zeros_(self.input_conv.bias)

        # Layer Normalization (applied on channel dimension)
        if self.tln:
            self.state_ln = nn.LayerNorm(2*num_features)
            self.input_ln = nn.LayerNorm(2*num_features)

    def init_state(self, inputs):
        # Initialize state z as zeros with shape (N, C, H, W)
        return torch.zeros(
            inputs.size(0), self.num_features,
            inputs.size(2), inputs.size(3),
            device=inputs.device, dtype=inputs.dtype
        )

    def forward(self, x, z):
        # If z is None, initialize it
        if z is None:
            z = self.init_state(x)

        # State-to-state convolution with optional LayerNorm
        z_concat = self.state_conv(z)  # Eq: z_concat = Conv(z)
        if self.tln:
            z_concat = z_concat.permute(0, 2, 3, 1)  # Channels last for LayerNorm
            z_concat = self.state_ln(z_concat)
            z_concat = z_concat.permute(0, 3, 1, 2)  # Restore NCHW

        # Input-to-state convolution with optional LayerNorm
        x_concat = self.input_conv(x)  # Eq: x_concat = Conv(x)
        if self.tln:
            x_concat = x_concat.permute(0, 2, 3, 1)
            x_concat = self.input_ln(x_concat)
            x_concat = x_concat.permute(0, 3, 1, 2)

        gates = x_concat + z_concat  # Eq: g = z_concat + x_concat

        # Split gates into p (tanh) and u (sigmoid)
        p, u = torch.split(gates, self.num_features, dim=1)  # Split along channels
        p = torch.tanh(p)  # Eq: p = tanh(p)
        u = torch.sigmoid(u)  # Eq: u = σ(u)

        # Update state: z_new = u*p + (1-u)*z
        z_new = u * p + (1 - u) * z  # Eq: z^{(t)} = u⊙p + (1-u)⊙z^{(t-1)}

        return z_new