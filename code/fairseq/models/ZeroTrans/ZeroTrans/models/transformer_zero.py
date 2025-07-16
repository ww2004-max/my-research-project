from typing import Optional

from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer.transformer_legacy import (
    TransformerModel
)
from fairseq.models.transformer.transformer_config import (
    TransformerConfig,
)
from ZeroTrans.models.transformer_encoder_zero import TransformerEncoderZero


@register_model("transformer_zero")
class TransformerModelZero(TransformerModel):

    def __init__(self, cfg, encoder, decoder):
        super().__init__(cfg, encoder, decoder)
        if cfg.freeze_position != 0:
            count = cfg.freeze_position - 1
            for param in self.encoder.embed_tokens.parameters():
                param.requires_grad = False
            for param in self.decoder.embed_tokens.parameters():
                param.requires_grad = False
        
            for i in range(count):
                layer = self.encoder.layers[i]
                for param in layer.parameters():
                    param.requires_grad = False

            for param in self.encoder.layers[count].self_attn.parameters():
                param.requires_grad = False
            for param in self.encoder.layers[count].self_attn_layer_norm.parameters():
                param.requires_grad = False


    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        """Add model-specific arguments to the parser."""
        parser.add_argument('--language-num', default=5, type=int)
        parser.add_argument('--lse', default=False, type=bool, help="switch of language specific embedding")
        parser.add_argument('--lse-position', default=5, type=int)
        parser.add_argument('--lse-dim', default=64, type=int)
        parser.add_argument('--freeze-position', default=0, type=int)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoderZero(
            TransformerConfig.from_namespace(args), src_dict, embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return super().build_decoder(
            TransformerConfig.from_namespace(args), tgt_dict, embed_tokens
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        **kwargs,
    ):
        # kwargs.get return tuple, but we need tensor, so add [0]
        src_direction = kwargs.get("src_direction", None),
        tgt_direction = kwargs.get("tgt_direction", None),
        src_direction = src_direction[0]
        tgt_direction = tgt_direction[0]
        if src_direction is None:
            raise ValueError(
                "building model has no src_direction!"
            )
        if tgt_direction is None:
            raise ValueError(
                "building model has no tgt_direction!"
            )
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            src_direction=src_direction,
            tgt_direction=tgt_direction,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out[1]["src_direction"] = src_direction
        decoder_out[1]["tgt_direction"] = tgt_direction
        decoder_out[1]["prev_output_tokens"] = prev_output_tokens
        decoder_out[1]["language_num"] = encoder_out["language_num"]

        return decoder_out


@register_model_architecture("transformer_zero", "transformer_zero")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.2)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.merge_src_tgt_embed = getattr(args, "merge_src_tgt_embed", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

@register_model_architecture("transformer_zero", "transformer_zero_opus")
def transformer_zero_opus(args):
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.dropout = getattr(args, "dropout", 0.1)
    base_architecture(args)


@register_model_architecture("transformer_zero", "transformer_vaswani_wmt_en_de_big_zero")
def transformer_vaswani_wmt_en_de_big_zero(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)