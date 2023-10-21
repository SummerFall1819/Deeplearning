import torch
import torch.nn as nn
from modules import Encoder, LayerNorm

class TModel(nn.Module):
    def __init__(self, args):
        super(TModel, self).__init__()
        self.args = args
        # todo

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ 
        Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def add_position_embedding(self, sequence):
        """
        sequence.shape = [batch_size, len]
        """
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        # todo

        return sequence_emb

    def forward(self, input_ids):

        sequence_emb = self.add_position_embedding(input_ids)
        # todo 

        return sequence_output


