import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalmLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_dim, width, filter_size, stride, layer_norm):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.padding = filter_size // 2
        
        # Temporal pathway parameters
        self.W_k = nn.Conv2d(input_channels, hidden_dim, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        self.W_v = nn.Conv2d(input_channels, hidden_dim, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        self.W_i = nn.Conv2d(input_channels, 1, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        self.W_f = nn.Conv2d(input_channels, 1, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        
        # Spatial pathway parameters
        self.W_k_prime = nn.Conv2d(input_channels, hidden_dim, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        self.W_v_prime = nn.Conv2d(input_channels, hidden_dim, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        self.W_i_prime = nn.Conv2d(input_channels, 1, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        self.W_f_prime = nn.Conv2d(input_channels, 1, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        
        # Output pathway parameters
        self.W_q = nn.Conv2d(input_channels, hidden_dim, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        self.W_q_prime = nn.Conv2d(input_channels, hidden_dim, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        self.W_o = nn.Conv2d(input_channels, hidden_dim, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        self.W_o_prime = nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        self.W_H = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False)
        
        # Initialize biases
        # nn.init.zeros_(self.W_k.bias)
        # nn.init.zeros_(self.W_v.bias)
        # nn.init.zeros_(self.W_i.bias)
        # nn.init.zeros_(self.W_f.bias)
        # nn.init.zeros_(self.W_k_prime.bias)
        # nn.init.zeros_(self.W_v_prime.bias)
        # nn.init.zeros_(self.W_i_prime.bias)
        # nn.init.zeros_(self.W_f_prime.bias)
        # nn.init.zeros_(self.W_q.bias)
        # nn.init.zeros_(self.W_q_prime.bias)
        # nn.init.zeros_(self.W_o.bias)
        # nn.init.zeros_(self.W_o_prime.bias)
        # nn.init.zeros_(self.W_H.bias)

    def forward(self, x_t, prev_states):
        """
        Args:
            x_t: Input tensor of shape (batch_size, height, width, input_channels)
            prev_states: Tuple containing:
                - C_prev: Temporal cell state (B, H, W, hidden_dim)
                - M_prev: Spatial cell state (B, H, W, hidden_dim)
                - m_prev: Temporal stabilization (B, H, W, 1)
                - m_prime_prev: Spatial stabilization (B, H, W, 1)
        Returns:
            H_t: Output tensor (B, H, W, input_channels)
            new_states: Updated states (C_t, M_t, m_t, m_prime_t)
        """
        n_prev, C_prev, M_prev, m_prev, m_prime_prev = prev_states
        # print(x_t.shape)
        # Convert input to channels-first for Conv2d
        x = x_t.permute(0, 3, 1, 2)  # (B, C, H, W)
        # print(x.shape)

        # --- Temporal Pathway ---
        # Key/Value projections
        k_t = (1 / torch.sqrt(torch.tensor(self.hidden_dim))) * self.W_k(x).permute(0, 2, 3, 1)
        v_t = self.W_v(x).permute(0, 2, 3, 1)
        i_tilde = self.W_i(x).permute(0, 2, 3, 1)
        f_tilde = self.W_f(x).permute(0, 2, 3, 1)
        
        # Stabilization
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i_t = torch.exp(i_tilde - m_t)
        f_t = torch.exp(f_tilde + m_prev - m_t)
        
        # Update temporal cell state
        C_t = f_t * C_prev + i_t * (v_t * k_t)
        n_t = f_t * n_prev + i_t * k_t
        
        # --- Spatial Pathway ---
        # Key/Value projections
        k_prime_t = (1 / torch.sqrt(torch.tensor(self.hidden_dim))) * self.W_k_prime(x).permute(0, 2, 3, 1)
        v_prime_t = self.W_v_prime(x).permute(0, 2, 3, 1)
        i_tilde_prime = self.W_i_prime(x).permute(0, 2, 3, 1)
        f_tilde_prime = self.W_f_prime(x).permute(0, 2, 3, 1)
        
        # Stabilization
        m_prime_t = torch.max(f_tilde_prime + m_prime_prev, i_tilde_prime)
        i_prime_t = torch.exp(i_tilde_prime - m_prime_t)
        f_prime_t = torch.exp(f_tilde_prime + m_prime_prev - m_prime_t)
        
        # Update spatial cell state
        M_t = f_prime_t * M_prev + i_prime_t * (v_prime_t * k_prime_t)
        n_prime_t = f_prime_t * C_t + i_prime_t * k_prime_t
        
        # --- Output Pathway ---
        # Query projections
        q_t = self.W_q(x).permute(0, 2, 3, 1)
        q_prime_t = self.W_q_prime(x).permute(0, 2, 3, 1)
        
        # Attention and fusion
        tilde_h = (C_t * q_t) / torch.max(torch.abs(n_t * q_t), torch.tensor(1.0))
        tilde_h_prime = (M_t * q_prime_t) / torch.max(torch.abs(n_prime_t * q_prime_t), torch.tensor(1.0))
        tilde_H = tilde_h * tilde_h_prime
        
        # Output gate
        tilde_o = self.W_o(x).permute(0, 2, 3, 1)
        concat = torch.cat([tilde_o, tilde_H], dim=-1).permute(0, 3, 1, 2)  # (B, 2d, H, W)
        o_t = torch.tanh(self.W_o_prime(concat).permute(0, 2, 3, 1))  # (B, H, W, C)
        # print(o_t.shape)
        # print(tilde_H.shape)

        # Final output
        H_tilde = self.W_H(tilde_H.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # print(H_tilde.shape)

        H_t = o_t * torch.tanh(H_tilde)
        
        # Convert back to channels-last (already in this format)
        new_states = (n_t, C_t, M_t, m_t, m_prime_t)
        return H_t, new_states

    def init_states(self, x_shape):
        """Initialize states for a new sequence."""
        batch_size, height, width, _ = x_shape
        device = next(self.parameters()).device
        return (
            torch.zeros((batch_size, height, width, self.hidden_dim), device=device),  # n_0
            torch.zeros((batch_size, height, width, self.hidden_dim), device=device),  # C_0
            torch.zeros((batch_size, height, width, self.hidden_dim), device=device),  # M_0
            torch.zeros((batch_size, height, width, 1), device=device),                # m_0
            torch.zeros((batch_size, height, width, 1), device=device),                # m_prime_0
        )