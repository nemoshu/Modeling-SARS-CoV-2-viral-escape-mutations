import torch
import torch.nn.functional as F
from torch import nn


class RnnModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size, dropout_p, cell_type):
        """
        Recurrent neural network.

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_size (int): number of units in the hidden layer
            dropout_p (float): dropout probability
            cell_type (string): type of cell ('LSTM', 'GRU' or 'RNN')
        """

        super(RnnModel, self).__init__()

        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.cell_type = cell_type

        self.dropout = nn.Dropout(dropout_p)

        if cell_type == 'LSTM':
            self.encoder = nn.LSTM(input_dim, hidden_size)
        elif cell_type == 'GRU':
            self.encoder = nn.GRU(input_dim, hidden_size)
        elif cell_type == 'RNN':
            self.encoder = nn.RNN(input_dim, hidden_size)

        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, input_seq, hidden_state):
        """
        Performs a forward pass.

        Args:
            input_seq (torch.Tensor): input sequence, with dimensions (seq_len, batch_size, input_dim)
            hidden_state (torch.Tensor): initial hidden state, typically from output of init_hidden

        Returns:
            score_seq (torch.Tensor): output prediction for mutation, shape (batch_size, output_dim)
            dummy_attn_weights (torch.Tensor): placeholder zeros, shape (batch_size, seq_len)
        """
        input_seq = self.dropout(input_seq)
        encoder_outputs, _ = self.encoder(input_seq, hidden_state)
        score_seq = self.out(encoder_outputs[-1, :, :])

        dummy_attn_weights = torch.zeros(input_seq.shape[1], input_seq.shape[0])
        return score_seq, dummy_attn_weights  # Placeholder attention weights

    def init_hidden(self, batch_size):
        """
        Initializes the hidden layer to all zeros

        Parameters:
            batch_size: Number of sequences in the batch
        Returns:
            For 'LSTM' cells, two all-zero 3D Pytorch tensors with dimensions (1, batch_size, hidden_size)
            For other types, one all-zero 3D Pytorch tensors with dimensions (1, batch_size, hidden_size)
        """

        if self.cell_type == 'LSTM':
            h_init = torch.zeros(1, batch_size, self.hidden_size)
            c_init = torch.zeros(1, batch_size, self.hidden_size)
            return h_init, c_init
        elif self.cell_type == 'GRU':
            return torch.zeros(1, batch_size, self.hidden_size)
        elif self.cell_type == 'RNN':
            return torch.zeros(1, batch_size, self.hidden_size)


class AttentionModel(nn.Module):

    def __init__(self, seq_length, input_dim, output_dim, hidden_size, dropout_p):
        """
            A temporal attention model using an LSTM encoder.

            Args:
                seq_length (int): length of input sequence
                input_dim (int): input dimension
                output_dim (int): output dimension
                hidden_size (int): number of units in the hidden layer
                dropout_p (float): dropout probability
        """
        super(AttentionModel, self).__init__()

        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.output_dim = output_dim

        self.encoder = nn.LSTM(input_dim, hidden_size)
        self.attn = nn.Linear(hidden_size, seq_length)
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, output_dim)

    def forward(self, input_seq, hidden_state):
        """
        Performs a forward pass.

        Args:
            input_seq (torch.Tensor): input sequence, with dimensions (seq_len, batch_size, input_dim)
            hidden_state (torch.Tensor): initial hidden state, typically from output of init_hidden

        Returns:
            score_seq (torch.Tensor): output prediction for mutation, shape (batch_size, output_dim)
            attn_weights (torch.Tensor): attention weights, shape (batch_size, seq_length)
        """
        input_seq = self.dropout(input_seq)
        encoder_outputs, (h, _) = self.encoder(input_seq, hidden_state)
        attn_applied, attn_weights = self.attention(encoder_outputs, h)
        score_seq = self.out(attn_applied.reshape(-1, self.hidden_size))

        return score_seq, attn_weights

    def attention(self, encoder_outputs, hidden):
        """
        Computes attention weights by scoring hidden states.

        Args:
            encoder_outputs (torch.Tensor): output of encoder
            hidden (torch.Tensor): hidden state
        Returns:
            attn_applied (torch.Tensor): weighted sum of encoder outputs
            attn_weights (torch.Tensor): attention weights
        """
        attn_weights = F.softmax(torch.squeeze(self.attn(hidden)), dim=1)
        attn_weights = torch.unsqueeze(attn_weights, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        attn_applied = torch.bmm(attn_weights, encoder_outputs)

        return attn_applied, torch.squeeze(attn_weights)

    def init_hidden(self, batch_size):
        """
        Initialises the hidden layer. Uses the GPU to speed up computation where possible.

        Parameters:
            batch_size: Number of sequences in the batch

        Returns:
            h_init (Tensor), c_init (Tensor):
            Two pytorch tensors with dimensions 1 * batch_size * hidden_size, initialised to all zeros
        """
        h_init = torch.zeros(1, batch_size, self.hidden_size)
        c_init = torch.zeros(1, batch_size, self.hidden_size)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        h_init = h_init.to(device)
        c_init = c_init.to(device)
        return h_init, c_init


class DaRnnModel(nn.Module):

    def __init__(self, seq_length, input_dim, output_dim, hidden_size, dropout_p):
        """
        A Dual-Attention RNN model with attention over both the input at each timestep
        and all hidden states of the encoder to make the final prediction.

        Args:
            seq_length (int): length of input sequence
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_size (int): number of units in the hidden layer
            dropout_p (float): dropout probability
        """
        super(DaRnnModel, self).__init__()

        self.n = input_dim # input dimension
        self.m = hidden_size # hidden dimension
        self.T = seq_length # sequence length
        self.output_dim = output_dim # output dimension

        self.dropout = nn.Dropout(dropout_p) # drop out layer

        self.encoder = nn.LSTM(self.n, self.m) # encoder

        # input attention weight layers
        self.We = nn.Linear(2 * self.m, self.T)
        self.Ue = nn.Linear(self.T, self.T)
        self.ve = nn.Linear(self.T, 1)

        # temporal attention weights
        self.Ud = nn.Linear(self.m, self.m)
        self.vd = nn.Linear(self.m, 1)
        self.out = nn.Linear(self.m, output_dim) # final output layer

    def forward(self, x, hidden_state):
        """
        Performs a forward pass.

        Args:
            x (torch.Tensor): input sequence, with dimensions (seq_len, batch_size, input_dim)
            hidden_state (torch.Tensor): initial hidden state, typically from output of init_hidden

        Returns:
            logits (torch.Tensor): output prediction for mutation, shape (batch_size, output_dim)
            beta (torch.Tensor): attention weights, shape (batch_size, seq_length)
        """
        x = self.dropout(x)
        h_seq = []
        for t in range(self.T):
            x_tilde, _ = self.input_attention(x, hidden_state, t)
            ht, hidden_state = self.encoder(x_tilde, hidden_state)
            h_seq.append(ht)

        h = torch.cat(h_seq, dim=0)
        c, beta = self.temporal_attention(h)
        logits = self.out(c)

        return logits, torch.squeeze(beta)

    def input_attention(self, x, hidden_state, t):
        """
        Computes attention over input features.

        Args:
            x (torch.Tensor): input sequence, with dimensions (seq_len, batch_size, input_dim)
            hidden_state (torch.Tensor): initial hidden state, typically from output of init_hidden
            t (int): sequence position (timestep)

        Returns:
            x_tilde (Tensor): weighted input features at t, shaped (1, batch_size, input_dim)
            alpha (Tensor): input attention weights, shaped (batch_size, seq_length)
        """
        x = x.permute(1, 2, 0)
        h, c = hidden_state
        h = h.permute(1, 0, 2)
        c = c.permute(1, 0, 2)
        hc = torch.cat([h, c], dim=2)

        e = self.ve(torch.tanh(self.We(hc) + self.Ue(x)))
        e = torch.squeeze(e)
        alpha = F.softmax(e, dim=1)
        xt = x[:, :, t]

        x_tilde = alpha * xt
        x_tilde = torch.unsqueeze(x_tilde, 0)

        return x_tilde, alpha

    def temporal_attention(self, h):
        """
        Computes attention over hidden states.

        Args:
            h (torch.Tensor): hidden states for all time steps, shape (seq_length, batch_size, hidden_size)
        Returns:
            c (torch.Tensor): weighted sums of hidden states, shape (batch_size, hidden_size)
            beta (torch.Tensor): attention weights, shape (batch_size, seq_length)
        """
        h = h.permute(1, 0, 2)
        l = self.vd(torch.tanh((self.Ud(h))))
        l = torch.squeeze(l)
        beta = F.softmax(l, dim=1)
        beta = torch.unsqueeze(beta, 1)
        c = torch.bmm(beta, h)
        c = torch.squeeze(c)

        return c, beta

    def init_hidden(self, batch_size):
        """
        Initialises hidden layers to all zeros.

        Args:
            batch_size (int): batch size

        Returns:
            h_init (torch.Tensor), c_init (torch.Tensor):
            Two all-zero tensors shape (1, batch_size, hidden_size)
        """
        h_init = torch.zeros(1, batch_size, self.m)
        c_init = torch.zeros(1, batch_size, self.m)

        return h_init, c_init


class TransformerModel(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_p):
        """
        Transformer method with attention mechanism.

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            dropout_p (float): dropout probability
        """
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim  # 100
        self.output_dim = output_dim  # 2
        self.hidden_size = 512
        self.dropout = nn.Dropout(dropout_p)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=5)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fnn = nn.Linear(input_dim, output_dim) # maps transformer encoder output to output dimension

    def forward(self, input_seq, hidden_state):
        """
        Performs a forward pass.

        Args:
            input_seq (torch.Tensor): input sequence, with dimensions (seq_len, batch_size, input_dim)
            hidden_state (torch.Tensor): initial hidden state, typically from output of init_hidden (unused)
        """
        out = self.dropout(input_seq)
        out = self.transformer_encoder(out)

        # out = torch.sum(out, 0)
        out = out[-1, :, :]

        out = self.fnn(out)
        return out, out

    def init_hidden(self, batch_size):
        """
        Initializes the hidden layer to all zeros, and sends them to the GPU where available.
        
        Parameters:
            batch_size: Number of sequences in the batch

        Returns:
            h_init (torch.Tensor), c_init (torch.Tensor):
            Two pytorch tensors with dimensions 1 * batch_size * hidden_size
        """
        h_init = torch.zeros(1, batch_size, self.hidden_size)
        c_init = torch.zeros(1, batch_size, self.hidden_size)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # use GPU where available
        h_init = h_init.to(device)
        c_init = c_init.to(device)
        return h_init, c_init
