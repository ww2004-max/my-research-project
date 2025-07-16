from typing import Optional

import torch
import torch.nn as nn
from fairseq.models.transformer import TransformerEncoderBase
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.distributed import fsdp_wrap
from ZeroTrans.modules import TransformerEncoderLayerZero


class TransformerEncoderZero(TransformerEncoderBase):

    def __init__(self, cfg, dictionary, embed_tokens):
        super().__init__(cfg, dictionary, embed_tokens)
        self.language_num = cfg.language_num
        self.lse = cfg.lse
        self.lse_dim = cfg.lse_dim
        self.lse_position = cfg.lse_position
        if self.lse:
            encoder_ls_embed = nn.Embedding(cfg.language_num, self.lse_dim)
            nn.init.normal_(encoder_ls_embed.weight, mean=0, std=self.lse_dim ** -0.5)
            self.ls_embed = encoder_ls_embed

    def build_encoder_layer(self, cfg):
        layer = TransformerEncoderLayerZero(cfg)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
            src_direction=None,
            tgt_direction=None,
    ):
        return self.forward_scriptable(
            src_tokens, src_lengths, return_all_hiddens, token_embeddings, src_direction, tgt_direction,
        )

    def forward_scriptable(
            self,
            src_tokens,
            src_lengths: Optional[torch.Tensor] = None,
            return_all_hiddens: bool = False,
            token_embeddings: Optional[torch.Tensor] = None,
            src_direction=None,
            tgt_direction=None,
    ):
        tgt_direction = tgt_direction - 1
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        token_tgt_direction = tgt_direction.unsqueeze(1).repeat(1, src_tokens.size()[1])
        if self.lse:
            ls_representation = self.ls_embed(token_tgt_direction)
            # ls_representation[:, 0,:] = ls_representation[:, 0,:] * False

        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))
            if self.lse:
                ls_representation = ls_representation * (
                        1 - encoder_padding_mask.unsqueeze(-1).type_as(ls_representation))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        if self.lse:
            ls_representation = ls_representation.transpose(0, 1)
        encoder_states = []
        if return_all_hiddens:
            encoder_states.append(x)
        # encoder layers
        count = 1
        for layer in self.layers:
            if self.lse and count == self.lse_position:
                x = layer(
                    x,
                    encoder_padding_mask=encoder_padding_mask if has_pads else None,
                    ls_representation=ls_representation
                )
            else:
                x = layer(
                    x,
                    encoder_padding_mask=encoder_padding_mask if has_pads else None,
                    ls_representation=None
                )
            count += 1
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        src_lengths = src_tokens.ne(self.padding_idx).sum(dim=1, dtype=torch.int32).reshape(-1, 1).contiguous()

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [src_lengths],
            "language_num": self.language_num,
        }
