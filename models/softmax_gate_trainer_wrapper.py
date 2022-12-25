from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils import ModelOutput

from modules.moe import SoftmaxGate


@dataclass
class SoftmaxGateOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class SoftmaxGateTrainingConfig(PretrainedConfig):
    def __init__(
        self,
        embed_dim=768,
        num_experts=2,
        intermediate_dim=128,
        pad_token_id=0,
        use_cache=True,
        classifier_dropout=None,
        num_labels=2,
        initializer_range=0.02,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.embed_dim = embed_dim
        self.num_experts = num_experts
        self.intermediate_dim = intermediate_dim
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.num_labels = num_labels
        self.initializer_range = initializer_range


class SoftmaxGateTrainingPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SoftmaxGateTrainingConfig
    base_model_prefix = "softmax_gate_training"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class SoftmaxGateTraining(SoftmaxGateTrainingPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_experts = config.num_experts
        self.config = config

        self.num_labels = config.num_labels

        self.gate = SoftmaxGate(
            embed_dim=config.embed_dim,
            num_experts=config.num_experts,
            intermediate_dim=config.intermediate_dim,
        )

        self.classifier = torch.nn.Linear(config.embed_dim, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        gate_input_embed: Optional[torch.Tensor] = None,
        expert1_embed: Optional[torch.Tensor] = None,
        expert2_embed: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        #     ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        weights = self.gate(gate_input_embed).unsqueeze(1)

        embeds = torch.cat(
            (expert1_embed.unsqueeze(2), expert2_embed.unsqueeze(2)), dim=2
        )

        final_representation = torch.sum(embeds * weights, dim=2)

        logits = self.classifier(final_representation)

        loss = None

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return SoftmaxGateOutput(
            loss=loss,
            logits=logits,
            # hidden_states=final_representation
        )
