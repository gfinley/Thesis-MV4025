import logging
import numpy as np
import gym
import torch.nn.functional as F
import math

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC, AppendBiasLayer, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict

torch, nn = try_import_torch()


import numpy as np
from typing import Dict, List
import gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType

torch, nn = try_import_torch()


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
import torch as th
import hexagdly

#baysian stuff
import bayesian_torch
from bayesian_torch import layers as bnn_layers
from bayesian_torch.layers.variational_layers.linear_variational import LinearVariational



logger = logging.getLogger(__name__)
#all of the aboce is imported from torch FC base model in an attempt to convert to a CNN hexagly enabled network

class HexBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.hexConv2d = hexagdly.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.relu = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        residual = x
        x = self.hexConv2d(x)
        if self.residual:
            x += residual
        x = self.relu(x)
        return x



class MyCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512, n_residual_layers = 7, use_residual = False):
        super(MyCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)

        n_input_channels = observation_space.shape[0]
        convs_per_layer = 64
        self.layers = OrderedDict()
        self.layers.update( {'conv': HexBlock(n_input_channels, convs_per_layer, residual=False)} )
        for i in range(n_residual_layers):
            layer_name = "resid"+str(i+1)
            self.layers.update( {layer_name: HexBlock(convs_per_layer, convs_per_layer, residual=use_residual)} ) 
        self.layers.update( {'flatten': nn.Flatten()})
        self.cnn = nn.Sequential(self.layers)
        model = self.cnn
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameter count {pytorch_total_params}")
        self.print_toggle = False

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if self.print_toggle:
            #print(observations)
            
            self.print_toggle = False
        #print("--------------Here are the observations--------------------")
        #print(observations[4])
        return self.linear(self.cnn(observations))


class Navy_CNN(TorchModelV2, nn.Module):
    """Generic fully connected network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        hiddens = list(model_config.get("fcnet_hiddens", [])) + list(
            model_config.get("post_fcnet_hiddens", [])
        )
        activation = model_config.get("fcnet_activation")
        if not model_config.get("fcnet_hiddens", []):
            activation = model_config.get("post_fcnet_activation")
        no_final_linear = model_config.get("no_final_linear")
        self.vf_share_layers = model_config.get("vf_share_layers")
        self.free_log_std = model_config.get("free_log_std")
        # Generate free-floating bias variables for the second half of
        # the outputs.
        if self.free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two",
                num_outputs,
            )
            num_outputs = num_outputs // 2

        layers = []
        prev_layer_size = int(np.product(obs_space.shape))
        self._logits = None

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = size

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation,
                )
            )
            prev_layer_size = num_outputs
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                layers.append(
                    SlimFC(
                        in_size=prev_layer_size,
                        out_size=hiddens[-1],
                        initializer=normc_initializer(1.0),
                        activation_fn=activation,
                    )
                )
                prev_layer_size = hiddens[-1]
            if num_outputs:
                self._logits = SlimFC(
                    in_size=prev_layer_size,
                    out_size=num_outputs,
                    initializer=normc_initializer(0.01),
                    activation_fn=None,
                )
            else:
                self.num_outputs = ([int(np.product(obs_space.shape))] + hiddens[-1:])[
                    -1
                ]

        # Layer to add the log std vars to the state-dependent means.
        if self.free_log_std and self._logits:
            self._append_free_log_std = AppendBiasLayer(num_outputs)

        self._hidden_layers = nn.Sequential(*layers)

        self._value_branch_separate = None
        if not self.vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            prev_vf_layer_size = int(np.product(obs_space.shape))
            vf_layers = []
            for size in hiddens:
                vf_layers.append(
                    SlimFC(
                        in_size=prev_vf_layer_size,
                        out_size=size,
                        activation_fn=activation,
                        initializer=normc_initializer(1.0),
                    )
                )
                prev_vf_layer_size = size
            self._value_branch_separate = nn.Sequential(*vf_layers)

        self._value_branch = SlimFC(
            in_size=prev_layer_size,
            out_size=1,
            initializer=normc_initializer(0.01),
            activation_fn=None,
        )
        # Holds the current "base" output (before logits layer).
        self._features = None
        # Holds the last input, in case value branch is separate.
        self._last_flat_in = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        obs = input_dict["obs_flat"].float()
        self._last_flat_in = obs.reshape(obs.shape[0], -1)
        self._features = self._hidden_layers(self._last_flat_in)
        logits = self._logits(self._features) if self._logits else self._features
        if self.free_log_std:
            logits = self._append_free_log_std(logits)
        return logits, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            return self._value_branch(
                self._value_branch_separate(self._last_flat_in)
            ).squeeze(1)
        else:
            return self._value_branch(self._features).squeeze(1)



#ORIGION
#https://github.com/ray-project/ray/blob/master/rllib/models/torch/visionnet.py


class VisionNetwork_CNN(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):

        if not model_config["custom_model_config"].get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config["custom_model_config"].get("conv_activation")
        filters = self.model_config["custom_model_config"]["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config["custom_model_config"].get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config["custom_model_config"].get("post_fcnet_activation"), framework="torch"
        )

        no_final_linear = self.model_config["custom_model_config"].get("no_final_linear")
        vf_share_layers = self.model_config["custom_model_config"].get("vf_share_layers")

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = True
        self._logits = None

        layers = []
        ( in_channels, w, h) = obs_space.shape
        
        

            #origional layer for model
            #layers.append(
            #    SlimConv2d(
            ##        in_channels,
            #        out_channels,
            #        kernel,
            #        stride,
            #        padding,
            #        activation_fn=activation,
        layers.append(HexBlock(14,64))
        for _ in range(0,6):
            layers.append(
                HexBlock(64,64,residual = False)
            )
        layers.append(nn.Flatten())
        
        #calculate flatten size
        size_layer = 1600
        #add linear layers of layer size,512,512,12
        layers.append(nn.Linear(size_layer,512))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(512,512))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(512,7))
        layers.append(nn.ReLU())
        
        self._logits = layers.pop() 
        self._convs = nn.Sequential(*layers)  
        self._value_branch_separate = nn.Sequential(*layers)   
        self._features = None  
        

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
        ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        #removed permutation becuase this is what I want>
        #self._features = self._features.permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        # Store features to save forward pass when getting value_function out.
        if not self._value_branch_separate:
            self._features = conv_out

        if not self.last_layer_is_flattened:
            if self._logits:
                conv_out = self._logits(conv_out)
            if len(conv_out.shape) == 4:
                if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
                    raise ValueError(
                        "Given `conv_filters` ({}) do not result in a [B, {} "
                        "(`num_outputs`), 1, 1] shape (but in {})! Please "
                        "adjust your Conv2D stack such that the last 2 dims "
                        "are both 1.".format(
                            self.model_config["conv_filters"],
                            self.num_outputs,
                            list(conv_out.shape),
                        )
                    )
                logits = conv_out.squeeze(3)
                logits = logits.squeeze(2)
            else:
                logits = conv_out
            return logits, state
        else:
            return conv_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            value = self._value_branch_separate(self._features)
            #trying to clear unknown dimention value
            #return value.squeeze()
            #value = value.squeeze(3)
            #value = value.squeeze(2)
            return value.squeeze(1)
        else:
            if not self.last_layer_is_flattened:
                features = self._features.squeeze(3)
                features = features.squeeze(2)
            else:
                features = self._features
            return self._value_branch(features).squeeze(1)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res

class VisionNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):

        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch"
        )

        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        
        #Changed
        #MAGIC NUMBER corrects a value function error for dimentionality
        self.last_layer_is_flattened = True
        self._logits = None

        layers = []
        print(obs_space.shape)
        #changed for atlatl
        (in_channels,w, h ) = obs_space.shape

        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]

        # No final linear: Last layer has activation function and exits with
        # num_outputs nodes (this could be a 1x1 conv or a FC layer, depending
        # on `post_fcnet_...` settings).
        if no_final_linear and num_outputs:
            out_channels = out_channels if post_fcnet_hiddens else num_outputs
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,  # padding=valid
                    activation_fn=activation,
                )
            )

            # Add (optional) post-fc-stack after last Conv2D layer.
            layer_sizes = post_fcnet_hiddens[:-1] + (
                [num_outputs] if post_fcnet_hiddens else []
            )
            for i, out_size in enumerate(layer_sizes):
                layers.append(
                    SlimFC(
                        in_size=out_channels,
                        out_size=out_size,
                        activation_fn=post_fcnet_activation,
                        initializer=normc_initializer(1.0),
                    )
                )
                out_channels = out_size

        # Finish network normally (w/o overriding last layer size with
        # `num_outputs`), then add another linear one of size `num_outputs`.
        else:
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,  # padding=valid
                    activation_fn=activation,
                )
            )

            # num_outputs defined. Use that to create an exact
            # `num_output`-sized (1,1)-Conv2D.
            if num_outputs:
                in_size = [
                    np.ceil((in_size[0] - kernel[0]) / stride),
                    np.ceil((in_size[1] - kernel[1]) / stride),
                ]
                padding, _ = same_padding(in_size, [1, 1], [1, 1])
                if post_fcnet_hiddens:
                    layers.append(nn.Flatten())
                    #add the flatten size here to allow the flatten to flow into the FC layers
                    # MAGIC NUMBER
                    in_size = 1600
                    # Add (optional) post-fc-stack after last Conv2D layer.
                    for i, out_size in enumerate(post_fcnet_hiddens + [num_outputs]):
                        layers.append(
                            SlimFC(
                                in_size=in_size,
                                out_size=out_size,
                                activation_fn=post_fcnet_activation
                                if i < len(post_fcnet_hiddens) - 1
                                else None,
                                initializer=normc_initializer(1.0),
                            )
                        )
                        in_size = out_size
                    # Last layer is logits layer.
                    self._logits = layers.pop()

                else:
                    self._logits = SlimConv2d(
                        out_channels,
                        num_outputs,
                        [1, 1],
                        1,
                        padding,
                        activation_fn=None,
                    )

            # num_outputs not known -> Flatten, then set self.num_outputs
            # to the resulting number of nodes.
            else:
                self.last_layer_is_flattened = True
                layers.append(nn.Flatten())

        self._convs = nn.Sequential(*layers)

        # If our num_outputs still unknown, we need to do a test pass to
        # figure out the output dimensions. This could be the case, if we have
        # the Flatten layer at the end.
        if self.num_outputs is None:
            # Create a B=1 dummy sample and push it through out conv-net.
            dummy_in = (
                torch.from_numpy(self.obs_space.sample())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )
            dummy_out = self._convs(dummy_in)
            self.num_outputs = dummy_out.shape[1]

        # Build the value layers
        self._value_branch_separate = self._value_branch = None
        if vf_share_layers:
            print("Output channel for Value branch is ", out_channels)
            ## MAGIC NUMBER
            out_channels = 7
            self._value_branch = SlimFC(
                out_channels, 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        else:
            vf_layers = []
            (in_channels, w, h) = obs_space.shape
            in_size = [w, h]
            for out_channels, kernel, stride in filters[:-1]:
                padding, out_size = same_padding(in_size, kernel, stride)
                vf_layers.append(
                    SlimConv2d(
                        in_channels,
                        out_channels,
                        kernel,
                        stride,
                        padding,
                        activation_fn=activation,
                    )
                )
                in_channels = out_channels
                in_size = out_size

            out_channels, kernel, stride = filters[-1]
            vf_layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    None,
                    activation_fn=activation,
                )
            )

            vf_layers.append(
                SlimConv2d(
                    in_channels=out_channels,
                    out_channels=1,
                    kernel=1,
                    stride=1,
                    padding=None,
                    activation_fn=None,
                )
            )
            self._value_branch_separate = nn.Sequential(*vf_layers)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 1, 2, 3)
        conv_out = self._convs(self._features)
        # Store features to save forward pass when getting value_function out.
        if not self._value_branch_separate:
            self._features = conv_out

        if not self.last_layer_is_flattened:
            if self._logits:
                conv_out = self._logits(conv_out)
            if len(conv_out.shape) == 4:
                if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
                    raise ValueError(
                        "Given `conv_filters` ({}) do not result in a [B, {} "
                        "(`num_outputs`), 1, 1] shape (but in {})! Please "
                        "adjust your Conv2D stack such that the last 2 dims "
                        "are both 1.".format(
                            self.model_config["conv_filters"],
                            self.num_outputs,
                            list(conv_out.shape),
                        )
                    )
                logits = conv_out.squeeze(3)
                logits = logits.squeeze(2)
            else:
                logits = conv_out
            return logits, state
        else:
            return conv_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            value = self._value_branch_separate(self._features)
            value = value.squeeze(3)
            value = value.squeeze(2)
            return value.squeeze(1)
        else:
            if not self.last_layer_is_flattened:
                features = self._features.squeeze(3)
                features = features.squeeze(2)
            else:
                features = self._features
            #return self._value_branch(features).squeeze(1)
            return self._value_branch(features)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs.permute(0, 1, 2, 3))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res

class HEX_VisionNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):

        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch"
        )

        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        
        #Changed
        #MAGIC NUMBER corrects a value function error for dimentionality
        self.last_layer_is_flattened = True
        self._logits = None

        layers = []
        print(obs_space.shape)
        #changed for atlatl
        (in_channels,w, h ) = obs_space.shape

        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                HexBlock(
                    in_channels,
                    out_channels,
                )
            )
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]

        # No final linear: Last layer has activation function and exits with
        # num_outputs nodes (this could be a 1x1 conv or a FC layer, depending
        # on `post_fcnet_...` settings).
        if no_final_linear and num_outputs:
            out_channels = out_channels if post_fcnet_hiddens else num_outputs
            layers.append(
                HexBlock(
                    in_channels,
                    out_channels,
                )
            )

            # Add (optional) post-fc-stack after last Conv2D layer.
            layer_sizes = post_fcnet_hiddens[:-1] + (
                [num_outputs] if post_fcnet_hiddens else []
            )
            for i, out_size in enumerate(layer_sizes):
                layers.append(
                    SlimFC(
                        in_size=out_channels,
                        out_size=out_size,
                        activation_fn=post_fcnet_activation,
                        initializer=normc_initializer(1.0),
                    )
                )
                out_channels = out_size

        # Finish network normally (w/o overriding last layer size with
        # `num_outputs`), then add another linear one of size `num_outputs`.
        else:
            layers.append(
                HexBlock(
                    in_channels,
                    out_channels,
                )
            )

            # num_outputs defined. Use that to create an exact
            # `num_output`-sized (1,1)-Conv2D.
            if num_outputs:
                in_size = [
                    np.ceil((in_size[0] - kernel[0]) / stride),
                    np.ceil((in_size[1] - kernel[1]) / stride),
                ]
                padding, _ = same_padding(in_size, [1, 1], [1, 1])
                if post_fcnet_hiddens:
                    layers.append(nn.Flatten())
                    #add the flatten size here to allow the flatten to flow into the FC layers
                    # MAGIC NUMBER
                    in_size = 64*w*h
                    # Add (optional) post-fc-stack after last Conv2D layer.
                    for i, out_size in enumerate(post_fcnet_hiddens + [num_outputs]):
                        layers.append(
                            SlimFC(
                                in_size=in_size,
                                out_size=out_size,
                                activation_fn=post_fcnet_activation
                                if i < len(post_fcnet_hiddens) - 1
                                else None,
                                initializer=normc_initializer(1.0),
                            )
                        )
                        in_size = out_size
                    # Last layer is logits layer.
                    self._logits = layers.pop()

                else:
                    self._logits = SlimConv2d(
                        out_channels,
                        num_outputs,
                        [1, 1],
                        1,
                        padding,
                        activation_fn=None,
                    )

            # num_outputs not known -> Flatten, then set self.num_outputs
            # to the resulting number of nodes.
            else:
                self.last_layer_is_flattened = True
                layers.append(nn.Flatten())

        self._convs = nn.Sequential(*layers)

        # If our num_outputs still unknown, we need to do a test pass to
        # figure out the output dimensions. This could be the case, if we have
        # the Flatten layer at the end.
        if self.num_outputs is None:
            # Create a B=1 dummy sample and push it through out conv-net.
            dummy_in = (
                torch.from_numpy(self.obs_space.sample())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )
            dummy_out = self._convs(dummy_in)
            self.num_outputs = dummy_out.shape[1]

        # Build the value layers
        self._value_branch_separate = self._value_branch = None
        if vf_share_layers:
            print("Output channel for Value branch is ", out_channels)
            ## MAGIC NUMBER
            out_channels = 7
            self._value_branch = SlimFC(
                out_channels, 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        else:
            vf_layers = []
            (in_channels, w, h) = obs_space.shape
            in_size = [w, h]
            for out_channels, kernel, stride in filters[:-1]:
                padding, out_size = same_padding(in_size, kernel, stride)
                vf_layers.append(
                    HexBlock(
                        in_channels,
                        out_channels,
                    )
                )
                in_channels = out_channels
                in_size = out_size

            out_channels, kernel, stride = filters[-1]
            vf_layers.append(
                HexBlock(
                    in_channels,
                    out_channels,
                )
            )

            vf_layers.append(
                HexBlock(
                    in_channels=out_channels,
                    out_channels=1,

                )
            )
            self._value_branch_separate = nn.Sequential(*vf_layers)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 1, 2, 3)
        conv_out = self._convs(self._features)
        # Store features to save forward pass when getting value_function out.
        if not self._value_branch_separate:
            self._features = conv_out

        if not self.last_layer_is_flattened:
            if self._logits:
                conv_out = self._logits(conv_out)
            if len(conv_out.shape) == 4:
                if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
                    raise ValueError(
                        "Given `conv_filters` ({}) do not result in a [B, {} "
                        "(`num_outputs`), 1, 1] shape (but in {})! Please "
                        "adjust your Conv2D stack such that the last 2 dims "
                        "are both 1.".format(
                            self.model_config["conv_filters"],
                            self.num_outputs,
                            list(conv_out.shape),
                        )
                    )
                logits = conv_out.squeeze(3)
                logits = logits.squeeze(2)
            else:
                logits = conv_out
            return logits, state
        else:
            return conv_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            value = self._value_branch_separate(self._features)
            value = value.squeeze(3)
            value = value.squeeze(2)
            return value.squeeze(1)
        else:
            if not self.last_layer_is_flattened:
                features = self._features.squeeze(3)
                features = features.squeeze(2)
            else:
                features = self._features
            #return self._value_branch(features).squeeze(1)
            return self._value_branch(features)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs.permute(0, 1, 2, 3))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res


class Baysian_network(TorchModelV2, nn.Module):
    """modification of HEX_vision network to work with hexagonal grids
    adds the bayisan fully connected laters and attempts to train""" 

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):

        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch"
        )

        no_final_linear = self.model_config.get("no_final_linear")
        vf_share_layers = self.model_config.get("vf_share_layers")

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        
        #Changed
        #MAGIC NUMBER corrects a value function error for dimentionality
        self.last_layer_is_flattened = True
        self._logits = None

        layers = []
        print(obs_space.shape)
        #changed for atlatl
        (in_channels,w, h ) = obs_space.shape

        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                HexBlock(
                    in_channels,
                    out_channels,
                )
            )
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]

        # No final linear: Last layer has activation function and exits with
        # num_outputs nodes (this could be a 1x1 conv or a FC layer, depending
        # on `post_fcnet_...` settings).
        if no_final_linear and num_outputs:
            out_channels = out_channels if post_fcnet_hiddens else num_outputs
            layers.append(
                HexBlock(
                    in_channels,
                    out_channels,
                )
            )

            # Add (optional) post-fc-stack after last Conv2D layer.
            layer_sizes = post_fcnet_hiddens[:-1] + (
                [num_outputs] if post_fcnet_hiddens else []
            )
            for i, out_size in enumerate(layer_sizes):
                layers.append(
                    LinearVariational(
                        in_features=out_channels,
                        out_features=out_size,
                    )
                )
                out_channels = out_size

        # Finish network normally (w/o overriding last layer size with
        # `num_outputs`), then add another linear one of size `num_outputs`.
        else:
            layers.append(
                HexBlock(
                    in_channels,
                    out_channels,
                )
            )

            # num_outputs defined. Use that to create an exact
            # `num_output`-sized (1,1)-Conv2D.
            if num_outputs:
                in_size = [
                    np.ceil((in_size[0] - kernel[0]) / stride),
                    np.ceil((in_size[1] - kernel[1]) / stride),
                ]
                padding, _ = same_padding(in_size, [1, 1], [1, 1])
                if post_fcnet_hiddens:
                    layers.append(nn.Flatten())
                    #add the flatten size here to allow the flatten to flow into the FC layers
                    # MAGIC NUMBER
                    in_size = 64*w*h
                    # Add (optional) post-fc-stack after last Conv2D layer.
                    for i, out_size in enumerate(post_fcnet_hiddens + [num_outputs]):
                        layers.append(
                            LinearVariational(
                        in_features=in_size,
                        out_features=out_size,
                    )
                        )
                        in_size = out_size
                    # Last layer is logits layer.
                    self._logits = layers.pop()

                else:
                    self._logits = SlimConv2d(
                        out_channels,
                        num_outputs,
                        [1, 1],
                        1,
                        padding,
                        activation_fn=None,
                    )

            # num_outputs not known -> Flatten, then set self.num_outputs
            # to the resulting number of nodes.
            else:
                self.last_layer_is_flattened = True
                layers.append(nn.Flatten())

        self._convs = nn.Sequential(*layers)

        # If our num_outputs still unknown, we need to do a test pass to
        # figure out the output dimensions. This could be the case, if we have
        # the Flatten layer at the end.
        if self.num_outputs is None:
            # Create a B=1 dummy sample and push it through out conv-net.
            dummy_in = (
                torch.from_numpy(self.obs_space.sample())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
            )
            dummy_out = self._convs(dummy_in)
            self.num_outputs = dummy_out.shape[1]

        # Build the value layers
        self._value_branch_separate = self._value_branch = None
        if vf_share_layers:
            print("Output channel for Value branch is ", out_channels)
            ## MAGIC NUMBER
            out_channels = 7
            self._value_branch = SlimFC(
                out_channels, 1, initializer=normc_initializer(0.01), activation_fn=None
            )
        else:
            vf_layers = []
            (in_channels, w, h) = obs_space.shape
            in_size = [w, h]
            for out_channels, kernel, stride in filters[:-1]:
                padding, out_size = same_padding(in_size, kernel, stride)
                vf_layers.append(
                    HexBlock(
                        in_channels,
                        out_channels,
                    )
                )
                in_channels = out_channels
                in_size = out_size

            out_channels, kernel, stride = filters[-1]
            vf_layers.append(
                HexBlock(
                    in_channels,
                    out_channels,
                )
            )

            vf_layers.append(
                HexBlock(
                    in_channels=out_channels,
                    out_channels=1,

                )
            )
            self._value_branch_separate = nn.Sequential(*vf_layers)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 1, 2, 3)
        conv_out = self._convs(self._features)
        # Store features to save forward pass when getting value_function out.
        if not self._value_branch_separate:
            self._features = conv_out

        if not self.last_layer_is_flattened:
            if self._logits:
                conv_out = self._logits(conv_out)
            if len(conv_out.shape) == 4:
                if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
                    raise ValueError(
                        "Given `conv_filters` ({}) do not result in a [B, {} "
                        "(`num_outputs`), 1, 1] shape (but in {})! Please "
                        "adjust your Conv2D stack such that the last 2 dims "
                        "are both 1.".format(
                            self.model_config["conv_filters"],
                            self.num_outputs,
                            list(conv_out.shape),
                        )
                    )
                logits = conv_out.squeeze(3)
                logits = logits.squeeze(2)
            else:
                logits = conv_out
            return logits, state
        else:
            return conv_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            value = self._value_branch_separate(self._features)
            value = value.squeeze(3)
            value = value.squeeze(2)
            return value.squeeze(1)
        else:
            if not self.last_layer_is_flattened:
                features = self._features.squeeze(3)
                features = features.squeeze(2)
            else:
                features = self._features
            #return self._value_branch(features).squeeze(1)
            return self._value_branch(features)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs.permute(0, 1, 2, 3))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res


#bayesian code taken from https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb
#for integration in to Baysian_network
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()
class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
    
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        PI = 0.5
        SIGMA_1 = torch.FloatTensor([math.exp(-0)])
        SIGMA_2 = torch.FloatTensor([math.exp(-6)])
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(PI, SIGMA_1, SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample:
            weight = self.weight.sample() 
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)

class Navy_VisionNetwork(TorchModelV2, nn.Module):
    """Generic vision network."""

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):

        if not model_config["custom_model_config"].get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config["custom_model_config"].get("conv_activation")
        filters = self.model_config["custom_model_config"]["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        # Post FC net config.
        post_fcnet_hiddens = model_config["custom_model_config"].get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config["custom_model_config"].get("post_fcnet_activation"), framework="torch"
        )

        no_final_linear = self.model_config["custom_model_config"].get("no_final_linear")
        vf_share_layers = self.model_config["custom_model_config"].get("vf_share_layers")

        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        self.last_layer_is_flattened = True
        self._logits = None

        layers = []
        ( in_channels, w, h) = obs_space.shape
        
        

            #origional layer for model
            #layers.append(
            #    SlimConv2d(
            ##        in_channels,
            #        out_channels,
            #        kernel,
            #        stride,
            #        padding,
            #        activation_fn=activation,
        layers.append(HexBlock(12,64))
        for _ in range(0,6):
            layers.append(
                HexBlock(64,64,residual = False)
            )
        layers.append(nn.Flatten())
        
        #calculate flatten size
        size_layer = 6*6*64
        #add linear layers of layer size,512,512,12
        layers.append(nn.Linear(size_layer,512))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(512,512))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(512,19))
        layers.append(nn.ReLU())
        
        self._logits = layers.pop() 
        self._convs = nn.Sequential(*layers)  
        self._value_branch_separate = nn.Sequential(*layers)   
        self._features = None  
        

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType,
    ) -> (TensorType, List[TensorType]):
        self._features = input_dict["obs"].float()
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        #removed permutation becuase this is what I want>
        #self._features = self._features.permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        # Store features to save forward pass when getting value_function out.
        if not self._value_branch_separate:
            self._features = conv_out

        if not self.last_layer_is_flattened:
            if self._logits:
                conv_out = self._logits(conv_out)
            if len(conv_out.shape) == 4:
                if conv_out.shape[2] != 1 or conv_out.shape[3] != 1:
                    raise ValueError(
                        "Given `conv_filters` ({}) do not result in a [B, {} "
                        "(`num_outputs`), 1, 1] shape (but in {})! Please "
                        "adjust your Conv2D stack such that the last 2 dims "
                        "are both 1.".format(
                            self.model_config["conv_filters"],
                            self.num_outputs,
                            list(conv_out.shape),
                        )
                    )
                logits = conv_out.squeeze(3)
                logits = logits.squeeze(2)
            else:
                logits = conv_out
            return logits, state
        else:
            return conv_out, state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        if self._value_branch_separate:
            value = self._value_branch_separate(self._features)
            #trying to clear unknown dimention value
            #return value.squeeze()
            #value = value.squeeze(3)
            #value = value.squeeze(2)
            return value.squeeze(1)
        else:
            if not self.last_layer_is_flattened:
                features = self._features.squeeze(3)
                features = features.squeeze(2)
            else:
                features = self._features
            return self._value_branch(features).squeeze(1)

    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res