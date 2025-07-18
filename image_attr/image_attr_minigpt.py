import numpy as np
import pandas as pd
import torch as ch
from numpy.typing import NDArray
from typing import Dict, Any, Optional, List
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from .context_partitioner import BaseContextPartitioner, SimpleContextPartitioner, SimpleImagePartitioner
from .solver import BaseSolver, LassoRegression
from .utils import (
    get_masks_and_logit_probs,
    get_minigpt_masks_and_logit_probs,
    get_vlm_masks_and_logit_probs,
    aggregate_logit_probs,
    split_text,
    highlight_word_indices,
    get_attributions_df,
    get_vlm_attributions_df,
    char_to_token,
)

from PIL import Image

import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.utils import save_image
from minigpt_utils import visual_attacker, prompt_wrapper, generator

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

DEFAULT_GENERATE_KWARGS = {"num_beams": 5, "max_new_tokens" :256, 
                        "min_length": 1,  "repetition_penalty": 1.5, 
                        "length_penalty": 1.0, "temperature": 1, "do_sample":True}


class ImageAttr_MINIGPT:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        pretrained_model_name_or_path: str,
        processor: None
    ) -> None:
       
        self.model_name = pretrained_model_name_or_path
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.generator = generator.Generator(model=model)

        self.logger = logging.getLogger("ImageAttr")
        self.logger.setLevel(logging.DEBUG)  # TODO: change to INFO later
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str=" ",
        tokenizer_name: str = " ",
        device: str = "cuda",
        model_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {},
        **kwargs: Dict[str, Any],
    ) -> "ImageAttr_MINIGPT":
    
        def parse_args():

            parser = argparse.ArgumentParser(description="Demo")
            parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
            parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
            parser.add_argument(
                "--options",
                nargs="+",
                help="override some settings in the used config, the key-value pair "
                    "in xxx=yyy format will be merged into config file (deprecate), "
                    "change to --cfg-options instead.",
            )
            args = parser.parse_args()
            return args
        
        args = parse_args()
        cfg = Config(args)

        model_config = cfg.model_cfg
        model_config.device_8bit = args.gpu_id
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
        model.eval()

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        print('[Initialization Finished]\n')
        # tokenizer = model.llama_tokenizer
        tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-13b-v1.1",  **tokenizer_kwargs)
        return cls(model, tokenizer, pretrained_model_name_or_path, processor, **kwargs)

    def init_image_query(
        self,
        image: Image,
        query: str,
        source_type: str = "patch",
        generate_kwargs: Optional[Dict[str, Any]] = None,
        num_ablations: int = 64,
        ablation_keep_prob: float = 0.5,
        batch_size: int = 1,
        solver: Optional[BaseSolver] = None,
    ):
        assert source_type=='patch'
        self.partitioner = SimpleImagePartitioner(image, source_type=source_type, image_size=224)
        prefix = prompt_wrapper.minigpt4_chatbot_prompt
        
        prefix = prefix % ('Please extend the following sentence: %s')
        self.query = prefix % query
        
        self.generate_kwargs = generate_kwargs or DEFAULT_GENERATE_KWARGS
        self.num_ablations = num_ablations
        self.ablation_keep_prob = ablation_keep_prob
        self.batch_size = batch_size
        self.solver = solver or LassoRegression()
    
        self._cache = {}

    def _get_query_ids(
        self,
        return_prompt: bool = False,
    ):
        chat_prompt = self.query
        chat_prompt_ids = self.tokenizer.encode(chat_prompt, add_special_tokens=False)

        if return_prompt:
            return chat_prompt_ids, chat_prompt
        else:
            return chat_prompt_ids

  
    @property
    def _response_start(self):
        prompt_ids = self._get_query_ids()
        return len(prompt_ids)
    
    @property
    def _output(self):
        if self._cache.get("output") is None:
            prompt_ids, prompt = self._get_query_ids(return_prompt=True)
            img_prompt = [self.processor(self.partitioner.get_image()).unsqueeze(0).to(self.model.device)]
            prompt_wrap = prompt_wrapper.Prompt(model=self.model, text_prompts=[prompt], img_prompts=[img_prompt])
            response, _ = self.generator.generate(prompt_wrap)
            self._cache["output"] = prompt + response            
        return self._cache["output"]
    
    def specify_output(self, jb_output):
        if self._cache.get("output") is None:
            prompt_ids, prompt = self._get_query_ids(return_prompt=True)
            self._cache["output"] = prompt + jb_output
        return self._cache["output"]
    
    @property
    def _output_tokens(self):
        return self.tokenizer(self._output, add_special_tokens=False)

    @property
    def _response_ids(self):
        return self._output_tokens["input_ids"][self._response_start :]

    @property
    def response(self): 
        output_tokens = self._output_tokens
        char_response_start = output_tokens.token_to_chars(self._response_start).start
        return self._output[char_response_start:]

    @property
    def num_sources(self) -> int:
        """
        The number of sources within the context. I.e., the number of sources
        that the context is partitioned into.

        Returns:
            int:
                The number of sources in the context.
        """
        return self.partitioner.num_sources

    @property
    def sources(self) -> List[np.array]:
        """
        The sources within the context. I.e., the context as a list
        where each element is a source.

        Returns:
            List[str]:
                The sources within the context.
        """
        return self.partitioner.sources

    def _char_range_to_token_range(self, start_index, end_index):
        output_tokens = self._output_tokens
        response_start = self._response_start
        offset = output_tokens.token_to_chars(response_start).start
        ids_start_index = char_to_token(output_tokens, start_index + offset)
        ids_end_index = char_to_token(output_tokens, end_index + offset - 1) + 1
        return ids_start_index - response_start, ids_end_index - response_start

    def _indices_to_token_indices(self, start_index=None, end_index=None):
        if start_index is None or end_index is None:
            start_index = 0
            end_index = len(self.response)
        if not (0 <= start_index < end_index <= len(self.response)):
            raise ValueError(
                f"Invalid selection range ({start_index}, {end_index}). "
                f"Please select any range within (0, {len(self.response)})."
            )
        return self._char_range_to_token_range(start_index, end_index)

    def _compute_vlm_masks_and_logit_probs(self) -> None:
        self._cache["reg_masks"], self._cache["reg_logit_probs"] = (
            get_minigpt_masks_and_logit_probs(
                self.model,
                self.processor,
                self.tokenizer,
                self.num_ablations,
                self.num_sources,
                self.partitioner,
                self._get_query_ids,
                self.response,
                self._response_ids,
                self.ablation_keep_prob,
                self.batch_size,
            )
        )

    @property
    def _masks(self):
        if self._cache.get("reg_masks") is None:
            self._compute_vlm_masks_and_logit_probs()
        return self._cache["reg_masks"]

    @property
    def _logit_probs(self):
        if self._cache.get("reg_logit_probs") is None:
            self._compute_vlm_masks_and_logit_probs()
        return self._cache["reg_logit_probs"]

    def _get_attributions_for_ids_range(self, ids_start_idx, ids_end_idx) -> tuple:
        outputs = aggregate_logit_probs(self._logit_probs[:, ids_start_idx:ids_end_idx])
        num_output_tokens = ids_end_idx - ids_start_idx
        
        weight, bias = self.solver.fit(self._masks, outputs, num_output_tokens)
        return weight, bias

    def get_attributions(
        self,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None,
        as_dataframe: bool = False,
        top_k: Optional[int] = None,
        verbose: bool = True,
    ) -> NDArray | Any:
        """
        Get the attributions for (part of) the response.

        Arguments:
            start_idx (int, optional):
                Start index of the part to attribute to. If None, defaults to
                the start of the response.
            end_idx (int, optional):
                End index of the part to attribute to. If None, defaults to the
                end of the response.
            as_dataframe (bool, optional):
                If True, return the attributions as a stylized dataframe in
                sorted order. Otherwise, return them as a numpy array where
                the ith element corresponds to the score of the ith source
                within the context. Defaults to False.
            top_k (int, optional):
                Only used if as_dataframe is True. Number of top attributions to
                return. If None, all attributions are returned. Defaults to None.
            verbose (bool, optional):
                If True, print the selected part of the response. Defaults to
                True.

        Returns:
            NDArray | Any:
                If as_dataframe is False, return a numpy array where the ith element
                corresponds to the score of the ith source within the context.
                Otherwise, return a stylized dataframe in sorted order.
        """
        if self.num_sources == 0:
            self.logger.warning("No sources to attribute to!")
            return np.array([])

        if not as_dataframe and top_k is not None:
            self.logger.warning("top_k is ignored when not using dataframes.")

        ids_start_idx, ids_end_idx = self._indices_to_token_indices(start_idx, end_idx)
        selected_text = self.response[start_idx:end_idx]
        selected_tokens = self._response_ids[ids_start_idx:ids_end_idx]
        decoded_text = self.tokenizer.decode(selected_tokens)
        if selected_text.strip() not in decoded_text.strip() and verbose:
            self.logger.warning(
                "Decoded selected tokens do not match selected text.\n"
                "If the following look close enough, feel free to ignore:\n"
                "What you selected: %s\nWhat is being attributed: %s",
                selected_text.strip(),
                decoded_text.strip(),
            )

        if verbose:
            print(f"Attributed: {decoded_text.strip()}")
        

        # _bias is the bias term in the l1 regression
        attributions, _bias = self._get_attributions_for_ids_range(
            ids_start_idx,
            ids_end_idx,
        )

        if as_dataframe:
            return get_vlm_attributions_df(attributions, self.partitioner, top_k=top_k)
        else:
            return attributions
        
        
