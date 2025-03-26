# -*- coding: utf-8 -*-
# Copyright (C) 2020 Unbabel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""
Universal Encoder
==============
    Pretrained encoder from Hugging Face.
"""
from typing import Dict, Optional

import torch

from comet.encoders.base import Encoder

import inspect
from transformers import AutoTokenizer, AutoModel, AutoConfig, MT5EncoderModel


class UniversalEncoder(Encoder):
    """Universal encoder using AutoModel and AutoTokenizer.

    Args:
        pretrained_model (str): Pretrained model from Hugging Face.
        load_pretrained_weights (bool): If set to True loads the pretrained weights
            from Hugging Face.
        local_files_only (bool): Whether or not to only look at local files.
    """

    def __init__(
        self,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> None:
        """Initializes a universal model compatible with all transformer encoders.

        Args:
            pretrained_model (str): Name or path of the pretrained model.
            load_pretrained_weights (bool): If True, loads pretrained weights.
            local_files_only (bool): If True, loads only from local files.
        """
        super().__init__()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model, use_fast=False, local_files_only=local_files_only
        )

        if 'mt5' in pretrained_model:
            self.model = MT5EncoderModel.from_pretrained(pretrained_model)
        else:
            # Load model with or without pretrained weights
            if load_pretrained_weights:
                self.model = AutoModel.from_pretrained(pretrained_model)
            else:
                config = AutoConfig.from_pretrained(pretrained_model, local_files_only=local_files_only)
                self.model = AutoModel.from_config(config)
            
            if hasattr(self.model, 'pooler'):
                if load_pretrained_weights:
                    self.model = AutoModel.from_pretrained(pretrained_model, add_pooling_layer=False)
                else:
                    config = AutoConfig.from_pretrained(pretrained_model, local_files_only=local_files_only)
                    self.model = AutoModel.from_config(config, add_pooling_layer=False)
            
        self.model.encoder.output_hidden_states = True
        
        if hasattr(self.model, 'decoder'):
            self.model.decoder=None
        
    
    @property
    def output_units(self) -> int:
        """Max number of tokens the encoder handles."""
        return self.model.config.hidden_size

    @property
    def max_positions(self) -> int:
        """Max number of tokens the encoder handles."""
        if hasattr(self.model.config, 'max_position_embeddings'):
            return self.model.config.max_position_embeddings - 2
        else:
            # Fallback value, assuming a common max length of 512 (adjust as necessary)
            return 512

    @property
    def num_layers(self) -> int:
        """Number of model layers available."""
        return self.model.config.num_hidden_layers + 1

    @property
    def size_separator(self) -> int:
        """Automatically determine the size of the separator based on the tokenizer.

        Returns:
            int: Number of tokens used between two segments.
        """
        # Sample input with two segments
        sentence = "This is a sentence."

        # Tokenize the sentence with two segments separated by the separator token
        tokenized = self.tokenizer.encode_plus(
            sentence, sentence, add_special_tokens=True, return_tensors="pt"
        )
        
        if hasattr(self.tokenizer, 'sep_token') and self.tokenizer.sep_token:
            # For BERT-based models, use the separator token
            separator_token = self.tokenizer.sep_token
        else:
            separator_token = self.tokenizer.eos_token 

        tokens=self.tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0])

        # Count the number of separator tokens in the encoded sequence
        return tokens[1:-1].count(separator_token)

    @property
    def uses_token_type_ids(self) -> bool:
        """Whether or not the model uses token type ids to differentiate sentences."""
        if hasattr(self.model.config, 'token_type_vocab_size'):
            return self.model.config.token_type_vocab_size > 1
        elif hasattr(self.model.config, 'type_vocab_size'):  # Fix: Check for type_vocab_size
            return self.model.config.type_vocab_size > 1
        # For T5 and mT5 models (they don't use token_type_ids)
        elif hasattr(self.model.config, 'vocab_size') and not hasattr(self.model.config, 'token_type_vocab_size'):
            return False
        else:
            # Default behavior: Return False if no information is available
            return False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model: str,
        load_pretrained_weights: bool = True,
        local_files_only: bool = False,
    ) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.

        Args:
            pretrained_model (str):Name of the pretrain model to be loaded.
            load_pretrained_weights (bool): If set to True loads the pretrained weights
                from Hugging Face
            local_files_only (bool): Whether or not to only look at local files.

        Returns:
            Encoder: UniversalEncoder object.
        """
        return UniversalEncoder(pretrained_model, load_pretrained_weights, local_files_only)
    
    def freeze_embeddings(self) -> None:
        """Freezes the embedding layer for different types of encoder models."""
        if hasattr(self.model, "embeddings"):  # For BERT-based models
            embedding_layer = self.model.embeddings
        elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "embed_tokens"):  # For mT5 & T5
            embedding_layer = self.model.encoder.embed_tokens
        elif hasattr(self.model, "get_input_embeddings"):  # General fallback
            embedding_layer = self.model.get_input_embeddings()
        else:
            raise ValueError("Could not locate embeddings layer in the model.")

        for param in embedding_layer.parameters():
            param.requires_grad = False
    
    def layerwise_lr(self, lr: float, decay: float):
        """Calculates the learning rate for each layer by applying a small decay.

        Args:
            lr (float): Learning rate for the highest encoder layer.
            decay (float): decay percentage for the lower layers.

        Returns:
            list: List of model parameters for all layers and the corresponding lr.
        """

        opt_parameters = []
    
        # Identify the encoder layers
        if hasattr(self.model, "encoder"):
            if hasattr(self.model.encoder, "layer"):  # BERT-based models
                layers = self.model.encoder.layer
            elif hasattr(self.model.encoder, "block"):  # T5/mT5 models
                layers = self.model.encoder.block
            elif hasattr(self.model.encoder, "layers"):  # NLLB models
                layers = self.model.encoder.layers
            else:
                raise ValueError("Unknown encoder layer structure.")
        else:
            raise ValueError("Model does not have an encoder.")

        num_layers = len(layers)
        # Last layer keeps LR
        opt_parameters.append(
            {"params": layers[-1].parameters(), "lr": lr}
        )


        # Decay at each layer.
        for i in range(2, num_layers + 1):
            opt_parameters.append(
                {
                    "params": layers[-i].parameters(),
                    "lr": lr * decay ** (i - 1),
                }
            )

        # Handle embedding layer
        if hasattr(self.model, "embeddings"):  # BERT-based models
            embedding_layer = self.model.embeddings
        elif hasattr(self.model.encoder, "embed_tokens"):  # T5/mT5 models
            embedding_layer = self.model.encoder.embed_tokens
        elif hasattr(self.model, "get_input_embeddings"):  # General fallback
            embedding_layer = self.model.get_input_embeddings()
        else:
            raise ValueError("Could not locate embeddings layer.")
        
        # Embedding Layer
        opt_parameters.append(
            {
                "params": embedding_layer.parameters(),
                "lr": lr * decay ** num_layers,
            }
        )
        
        return opt_parameters

    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Model forward pass for all encoder types.

        Args:
            input_ids (torch.Tensor): Tokenized input.
            attention_mask (torch.Tensor): Attention mask for padding tokens.
            token_type_ids (Optional[torch.Tensor]): For models that use token type IDs.
            **kwargs: Additional arguments for specific encoder models.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'sentemb', 'wordemb', 'all_layers', 'attention_mask'.
        """
        
        # Handle whether the model requires token_type_ids (e.g., BERT-based models)
        if self.uses_token_type_ids:
            model_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                return_dict=True,  # Return a dictionary (standard for newer models like BERT)
                **kwargs
            )
        else:
            if hasattr(self.model, 'decoder'):
                model_outputs = self.model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,  # Ensure dictionary format
                **kwargs
            )
            else:
                # For models like T5 or mT5 that do not require token_type_ids
                model_outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,  # Ensure dictionary format
                    **kwargs
                )

        # Depending on the model architecture, extract the sentence and word embeddings.
        # `last_hidden_state` is usually returned for sentence/word embeddings.
        last_hidden_states = model_outputs.last_hidden_state

        if hasattr(model_outputs, 'pooler_output') and model_outputs.pooler_output is not None:
            sentence_embedding = model_outputs.pooler_output
        else:
            # For models like XLM-R, we use the first token (CLS token) for the sentence embedding
            sentence_embedding = last_hidden_states[:, 0, :]

        # Extracting all hidden layers if available
        all_layers = model_outputs.hidden_states if 'hidden_states' in model_outputs.keys() else None

        return {
            "sentemb": sentence_embedding, 
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }
