# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team.
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
import os

import torch
from torch import nn 
# from torch import Tensor, device

from jjuke.net_utils.options import instantiate_from_config
# import accelerate
# from accelerate.utils import set_module_tensor_to_device
# from accelerate.utils.versions import is_torch_version
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HOME
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError
from requests import HTTPError

__version__ = "0.8.0"

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "diffusion_pytorch_model.bin"
FLAX_WEIGHTS_NAME = "diffusion_flax_model.msgpack"
ONNX_WEIGHTS_NAME = "model.onnx"
ONNX_EXTERNAL_WEIGHTS_NAME = "weights.pb"
SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors" # from version 0.26.0
SAFETENSORS_FILE_EXTENSION = "safetensors" # from version 0.26.0
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co") # from version 0.26.0
DIFFUSERS_CACHE = os.path.join(HF_HOME, "diffusers")
DIFFUSERS_DYNAMIC_MODULE_NAME = "diffusers_modules"
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(HF_HOME, "modules")) # from version 0.26.0


def _load_state_dict_into_model(model_to_load, state_dict):
    # Convert old format to new format if needed from a PyTorch state_dict
    # copy state_dict so _load_from_state_dict can modify it
    state_dict = state_dict.copy()
    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: torch.nn.Module, prefix=""):
        args = (state_dict, prefix, {}, True, [], [], error_msgs)
        module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model_to_load)
    return error_msgs


def load_model_from_pretrained(
    model: nn.Module,
    load_config: dict,
    cache_dir=DIFFUSERS_CACHE,
    ignore_mismatched_sizes=False,
    force_download=False,
    resume_download=False,
    proxies=None,
    output_loading_info=False,
    local_files_only=False,
    use_auth_token=None,
    revision=None,
    torch_dtype=None
):
    """ My version of from_pretrained class function. It is originally a class function in the `ModelMixin`. """
    
    user_agent = {"diffusers": __version__, "file_type": "model", "framework": "pytorch"}
    
    # Load config if we don't provide a configuration
    config_path = load_config.pretrained_model_name_or_path
    subfolder = load_config.subfolder
    
    # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the Load model.
    pretrained_model_name_or_path = str(load_config.pretrained_model_name_or_path)
    if os.path.isdir(pretrained_model_name_or_path):
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)):
            # Load from a PyTorch checkpoint
            model_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        elif subfolder is not None and os.path.isfile(
            os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)
        ):
            model_file = os.path.join(pretrained_model_name_or_path, subfolder, WEIGHTS_NAME)
        else:
            raise EnvironmentError(
                f"Error no file named {WEIGHTS_NAME} found in directory {pretrained_model_name_or_path}."
            )
    else:
        try:
            # Load from URL or cache if already cached
            model_file = hf_hub_download(
                pretrained_model_name_or_path,
                filename=WEIGHTS_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                subfolder=subfolder,
                revision=revision,
            )

        except RepositoryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier "
                "listed on 'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a "
                "token having permission to this repo with `use_auth_token` or log in with `huggingface-cli "
                "login`."
            )
        except RevisionNotFoundError:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for "
                "this model name. Check the model page at "
                f"'https://huggingface.co/{pretrained_model_name_or_path}' for available revisions."
            )
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {WEIGHTS_NAME}."
            )
        except HTTPError as err:
            raise EnvironmentError(
                "There was a specific connection error when trying to load"
                f" {pretrained_model_name_or_path}:\n{err}"
            )
        except ValueError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load this model, couldn't find it"
                f" in the cached files and it looks like {pretrained_model_name_or_path} is not the path to a"
                f" directory containing a file named {WEIGHTS_NAME} or"
                " \nCheckout your internet connection or see how to run the library in"
                " offline mode at 'https://huggingface.co/docs/diffusers/installation#offline-mode'."
            )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it from "
                "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a file named {WEIGHTS_NAME}"
            )
    
    # load the state_dict of the model
    state_dict = torch.load(model_file, map_location="cpu")
    
    # NOTE: this part is only for the case of using `stable-diffusion-2-base`.
    if config_path == "stabilityai/stable-diffusion-2-base":
        for name, param in state_dict.items():
            if "proj_in.weight" in name or "proj_out.weight" in name:
                reshaped_param = param[:, :, None, None]
                state_dict[name] = reshaped_param
    model_state_dict = model.state_dict()
    
    loaded_keys = [k for k in state_dict.keys()]
    expected_keys = list(model_state_dict.keys())
    original_loaded_keys = loaded_keys
    missing_keys = list(set(expected_keys) - set(loaded_keys))
    unexpected_keys = list(set(loaded_keys) - set(expected_keys))
    
    def _find_mismatched_keys(state_dict, model_state_dict, loaded_keys, ignore_mismatched_sizes):
        mismatched_keys = []
        if ignore_mismatched_sizes:
            for checkpoint_key in loaded_keys:
                model_key = checkpoint_key
                if model_key in model_state_dict and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape:
                    mismatched_keys.append((checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape))
                    del state_dict[checkpoint_key]
        return mismatched_keys
    
    # whole checkpoint
    model_to_load = model
    mismatched_keys = _find_mismatched_keys(state_dict, model_state_dict, original_loaded_keys, ignore_mismatched_sizes)
    error_msgs = _load_state_dict_into_model(model_to_load, state_dict)
    
    if len(error_msgs) > 0:
        error_msg = "\n\t".join(error_msgs)
        if "size mismatch" in error_msg:
            error_msg += "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `load_model_from_pretrained` function."
        raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
    if len(unexpected_keys) > 0:
        print(
            f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
            f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
            f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task"
            " or with another architecture (e.g. initializing a BertForSequenceClassification model from a"
            " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
            f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly"
            " identical (initializing a BertForSequenceClassification model from a"
            " BertForSequenceClassification model)."
        )
    else:
        print(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        print(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
            " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
        )
    elif len(mismatched_keys) == 0:
        print(
            f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the"
            f" checkpoint was trained on, you can already use {model.__class__.__name__} for predictions"
            " without further training."
        )
    if len(mismatched_keys) > 0:
        mismatched_warning = "\n".join(
            [
                f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                for key, shape1, shape2 in mismatched_keys
            ]
        )
        print(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
            f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
            f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be"
            " able to use it for predictions and inference."
        )
    
    loading_info = {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "mismatched_keys": mismatched_keys,
        "error_msgs": error_msgs
    }
    
    if torch_dtype is not None and not isinstance(torch_dtype, torch.dtype):
        raise ValueError(
            f"{torch_dtype} needs to be of type `torch.dtype`, e.g. `torch.float16`, but is {type(torch_dtype)}."
        )
    elif torch_dtype is not None:
        model = model.to(torch_dtype)
    
    if output_loading_info:
        return model, loading_info
    return model