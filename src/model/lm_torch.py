import torch
from pathlib import Path
import numpy as np
import einops

class MarathiLM(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_heads: int, n_layers: int, d_ff: int,
                 device = 'mps'):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.device = device

        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.d_model, device=self.device)
        self.position_embedding = torch.nn.Embedding(self.context_length, self.d_model, device=self.device)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads,
                                                         dim_feedforward=self.d_ff, device=self.device)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers,
                                                               norm=torch.nn.RMSNorm(normalized_shape=self.d_model,
                                                                                     device=self.device)
                                                               )
        self.final_output_layer = torch.nn.Linear(in_features=self.d_model, out_features=self.vocab_size,
                                                  device=self.device)

    def forward(self, x):
        token_embedding = self.embedding_layer(x)
        position_ids = torch.arange(self.context_length, device=self.device).unsqueeze(0).expand(x.shape[0], -1)
        position_embedding = self.position_embedding(position_ids)
        overall_embedding = token_embedding + position_embedding
        output = self.transformer_encoder(overall_embedding)
        output = self.final_output_layer(output)

        return output


if __name__ == '__main__':
    vocab_size = 6000
    context_length = 64
    d_model = 128
    num_heads = 4
    n_layers  =4
    d_ff = 128 * 4
    ROOT = Path(__file__).resolve().parent.parent.parent
    encoded_data_path = ROOT / "data" / "tukaram_gatha_train_encoded.npy"
    encoded_data = np.load(encoded_data_path)
    trial_batch = torch.tensor(encoded_data[:context_length])
    trial_batch = einops.rearrange(trial_batch, "b -> 1 b")

    trial_batch = trial_batch.to(device='mps')

    model = MarathiLM(vocab_size, context_length, d_model, num_heads, n_layers, d_ff)

    trial_output = model.forward(trial_batch)



    print()
    pass


