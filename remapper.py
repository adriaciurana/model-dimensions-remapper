import functools
import inspect
import sys
from abc import ABC, abstractmethod
from typing import Any, Optional, Type

import torch
from einops import rearrange
from torch import nn


class InputAdapterSingleValue(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kargs) -> tuple[list[Any], dict[str, Any]]:
        output = self.module(*args, **kargs)
        return [output], {}


class Rearrange(nn.Module):
    def __init__(self, expr: str):
        super().__init__()
        self.expr = expr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, self.expr)


class AdaptedModel(nn.Module):
    def __init__(
        self, input_adapter: nn.Module, output_adapter: nn.Module, model: nn.Module
    ):
        super().__init__()
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.model = model

    def forward(self, *args, **kargs) -> Any:
        args, kargs = self.input_adapter(*args, **kargs)
        output = self.model(*args, **kargs)
        return self.output_adapter(output)


class BaseTranslator(ABC):
    def __init__(
        self,
        input_adapter: Optional[nn.Module] = InputAdapterSingleValue(nn.Identity()),
        output_adapter: Optional[nn.Module] = nn.Identity(),
    ):
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter

    @abstractmethod
    def match_layer(self, layer_input_class: Type) -> bool:
        ...

    @abstractmethod
    def translate_layer(self, layer_input_class: Type) -> nn.Module:
        ...

    @abstractmethod
    def __call__(
        self,
        layer_input_class: nn.Module,
        layer_output_class: nn.Module,
        layer_output_param_name: str,
        layer_input_param_value: Any,
    ) -> Any:
        ...


class EasyGenericTranslatorMdtoNd(BaseTranslator):
    """
    This translator only change the padding, kernel_size,
    stride and dilation to allow one-row image.
    """

    def __init__(
        self, input_adapter: nn.Module, output_adapter: nn.Module, m: str, n: str
    ):
        super().__init__(
            input_adapter=input_adapter,
            output_adapter=output_adapter,
        )
        self.input_dim = m
        self.output_dim = n

    def match_layer(self, layer_input_class: Type) -> bool:
        return layer_input_class.__name__.endswith(self.input_dim)

    def translate_layer(self, layer_input_class: Type) -> nn.Module:
        # Get the python module from the layer class
        module_layer_input_name = layer_input_class.__module__

        # Obtain the layer that replaces the old layer
        module_output_dim_class_name = layer_input_class.__name__.replace(
            self.input_dim, self.output_dim
        )

        # Get the original module from the module layer name
        module_layer_input = sys.modules[module_layer_input_name]

        # Find the output_dim analogy
        layer_output_class = getattr(module_layer_input, module_output_dim_class_name)

        return layer_output_class

    def __call__(
        self,
        layer_input_class: nn.Module,
        layer_output_class: nn.Module,
        layer_output_param_name: str,
        layer_input_param_value: Any,
    ) -> nn.Module:
        if layer_output_param_name in ["padding"]:
            if isinstance(layer_input_param_value, tuple):
                layer_input_param_value = layer_input_param_value[0], 0

        elif layer_output_param_name in ["kernel_size", "stride", "dilation"]:
            if isinstance(layer_input_param_value, tuple):
                layer_input_param_value = layer_input_param_value[0], 1

        return layer_input_param_value


class Translator2dto1d(EasyGenericTranslatorMdtoNd):
    """
    This translator only change the padding, kernel_size,
    stride and dilation to allow one-row image.
    """

    def __init__(
        self,
        input_adapter: nn.Module = InputAdapterSingleValue(
            Rearrange("b c h -> b c h 1")
        ),
        output_adapter: nn.Module = nn.Identity(),
    ):
        super().__init__(
            input_adapter=input_adapter, output_adapter=output_adapter, m="2d", n="2d"
        )

    def match_layer(self, layer_input_class: Type) -> bool:
        return layer_input_class.__name__.endswith(self.input_dim)

    def translate_layer(self, layer_input_class: Type) -> nn.Module:
        # Get the python module from the layer class
        module_layer_input_name = layer_input_class.__module__

        # Obtain the layer that replaces the old layer
        module_output_dim_class_name = layer_input_class.__name__.replace(
            self.input_dim, self.output_dim
        )

        # Get the original module from the module layer name
        module_layer_input = sys.modules[module_layer_input_name]

        # Find the output_dim analogy
        layer_output_class = getattr(module_layer_input, module_output_dim_class_name)

        return layer_output_class

    def __call__(
        self,
        layer_input_class: nn.Module,
        layer_output_class: nn.Module,
        layer_output_param_name: str,
        layer_input_param_value: Any,
    ) -> nn.Module:
        if layer_output_param_name in ["padding"]:
            if isinstance(layer_input_param_value, tuple):
                layer_input_param_value = layer_input_param_value[0], 0

        elif layer_output_param_name in ["kernel_size", "stride", "dilation"]:
            if isinstance(layer_input_param_value, tuple):
                layer_input_param_value = layer_input_param_value[0], 1

        return layer_input_param_value


class TranslatorOnlyResNet2dto1d(EasyGenericTranslatorMdtoNd):
    """
    Replace 2d layers by 1d layers
    """

    def __init__(
        self,
        input_adapter: nn.Module = InputAdapterSingleValue(nn.Identity()),
        output_adapter: nn.Module = nn.Identity(),
    ):
        super().__init__(
            input_adapter=input_adapter, output_adapter=output_adapter, m="2d", n="1d"
        )

    def __call__(
        self,
        layer_input_class: nn.Module,
        layer_output_class: nn.Module,
        layer_output_param_name: str,
        layer_input_param_value: Any,
    ):
        if isinstance(layer_input_param_value, tuple):
            layer_input_param_value = layer_input_param_value[0]

        return layer_input_param_value


class TranslatorOnlyResNet2dto3d(EasyGenericTranslatorMdtoNd):
    """
    Replace 2d layers by 1d layers
    """

    def __init__(
        self,
        input_adapter: nn.Module = nn.Identity(),
        output_adapter: nn.Module = nn.Identity(),
    ):
        super().__init__(
            input_adapter=input_adapter, output_adapter=output_adapter, m="2d", n="3d"
        )

    def __call__(
        self,
        layer_input_class: nn.Module,
        layer_output_class: nn.Module,
        layer_output_param_name: str,
        layer_input_param_value: Any,
    ):
        if isinstance(layer_input_param_value, tuple):
            layer_input_param_value = (
                layer_input_param_value[0],
                layer_input_param_value[0],
                layer_input_param_value[0],
            )

        return layer_input_param_value


def convert(
    model: nn.Module,
    translator: BaseTranslator = Translator2dto1d(),
    adapter: bool = True,
) -> nn.Module:
    layer_input_name: str
    layer_input: nn.Module
    for layer_input_name, layer_input in model.named_modules():
        # If the layers comes from pytorch
        layer_input_class = layer_input.__class__
        module_layer_input_name = layer_input_class.__module__
        if module_layer_input_name.startswith("torch.nn.modules"):
            # If can be translated to output_dim
            if translator.match_layer(layer_input_class):
                # Obtain the same layer analogy from output_dim
                layer_output_class = translator.translate_layer(layer_input_class)

                # Obtain the visible parameters of the nd layer
                layer_input_params = layer_input.__dict__
                layer_output_inspect_constructor = inspect.signature(
                    layer_output_class.__init__
                )

                # Create the arguments for the output_dim constructor
                layer_output_constructor_params = {}

                # Check all the arguments needed for the M class
                for (
                    layer_output_param_name
                ) in layer_output_inspect_constructor.parameters.keys():
                    # Check if we have this parameters on the N object
                    if layer_output_param_name in layer_input_params.keys():
                        # Translate the parameter from M to N
                        layer_input_param_value = layer_input_params[
                            layer_output_param_name
                        ]
                        layer_output_param_value = translator(
                            layer_input_class,
                            layer_output_class,
                            layer_output_param_name,
                            layer_input_param_value,
                        )
                        layer_output_constructor_params[
                            layer_output_param_name
                        ] = layer_output_param_value

                # Replace this the M layer by the N layer
                layer_input_name_list = layer_input_name.split(".")
                model_root_obj = functools.reduce(
                    lambda a, b: getattr(a, b), [model] + layer_input_name_list[:-1]
                )
                setattr(
                    model_root_obj,
                    layer_input_name_list[-1],
                    layer_output_class(**layer_output_constructor_params),
                )

    # Add wrapper
    if adapter:
        assert (
            translator.input_adapter is not None
        ), "translator.input_adapter cannot be None."
        assert (
            translator.output_adapter is not None
        ), "translator.output_adapter cannot be None."
        return AdaptedModel(
            input_adapter=translator.input_adapter,
            output_adapter=translator.output_adapter,
            model=model,
        )

    return model
