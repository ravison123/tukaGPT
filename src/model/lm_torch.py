import torch

class MarathiLM(torch.nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_heads: int, n_layers: int, d_ff: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.d_ff = d_ff

        self.embedding_layer = torch.nn.Embedding(self.vocab_size, self.d_model)
        self.position_embedding = torch.nn.Embedding(self.context_length, self.d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.num_heads,
                                                         dim_feedforward=self.d_ff)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers,
                                                               norm=torch.nn.RMSNorm(normalized_shape=self.d_model))
        self.final_output_layer = torch.nn.Linear(in_features=self.d_model, out_features=self.vocab_size)

    def forward(self, x):
        token_embedding = self.embedding_layer(x)
        position_embedding = self.position_embedding(x)
        overall_embedding = token_embedding + position_embedding
        output = self.transformer_encoder(overall_embedding)
        output = self.final_output_layer(output)

        return output


if __name__ == '__main__':
    pass


