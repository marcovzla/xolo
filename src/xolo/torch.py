import dataclasses
import math
import os
import random
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar

import numpy as np
import torch

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

from xolo.hooks import Hook, HookAlreadyRegisteredError, HookNotRegisteredError
from xolo.typing import Args, KwArgs
from xolo.utils import is_dataclass_instance, is_namedtuple_instance

# Type Aliases
T = TypeVar('T')
Device = str | int | torch.device



def set_seed(seed: int) -> torch.Generator:
    """
    Sets the random number generator seeds for Python, NumPy, and PyTorch.

    This function takes an integer seed value and sets the random number generator seeds
    for Python's built-in `random` module, NumPy's random module, and PyTorch's random module.
    The provided seed value ensures reproducibility of random number generation across
    different libraries and functions.

    Args:
        seed (int): The seed value to initialize the random number generators.

    Returns:
        torch.Generator: A PyTorch random number generator with the specified seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    return torch.manual_seed(seed)



def seed_worker(worker_id: int):
    """
    Function that can be used as DataLoader's worker_init_fn to preserve reproducibility.
    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    set_seed(worker_seed)



def enable_full_determinism(seed: int, *, warn_only: bool = False):
    """
    Enables full determinism in PyTorch operations for reproducible results.

    This function configures various settings within the PyTorch environment to ensure
    full determinism in computations. By setting a common seed and modifying relevant
    environment variables, it aims to make PyTorch operations consistent and reproducible.
    This is especially useful for debugging and achieving consistent results across runs.

    Args:
        seed (int): The seed value to initialize the random number generators.
        warn_only (bool, optional): If True, warnings about non-deterministic operations
            will be displayed, but the operations will not be disabled. Defaults to False.

    Note:
        - Enabling full determinism might impact performance due to certain optimizations
          being disabled.
        - CUDA-based operations and libraries are also configured for determinism.
    """
    set_seed(seed)
    # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#concurrent-execution-between-host-and-device
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # https://pytorch.org/docs/stable/notes/randomness.html#avoiding-nondeterministic-algorithms
    torch.use_deterministic_algorithms(mode=True, warn_only=warn_only)
    # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-determinism
    torch.backends.cudnn.deterministic = True
    # https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking
    torch.backends.cudnn.benchmark = False



def move_to_device(obj: T, device: Device) -> T:
    """
    Recursively moves an object to the specified PyTorch device.

    This function is designed to transfer various data structures and models to a
    PyTorch device (like GPU or CPU). It handles PyTorch tensors, modules (like neural network models),
    dataclasses, namedtuples, and general sequences and mappings. For composite structures like
    sequences and mappings, the operation is applied recursively. The function preserves the type
    of the input object, ensuring that the output has the same type as the input. If the object
    type is not directly handled, it's returned unchanged.

    Args:
        obj (T): The object to be moved. This can be a PyTorch tensor, module (nn.Module), dataclass,
                 namedtuple, any sequence (like lists and tuples), or mapping (like dictionaries).
        device (Device): The target PyTorch device (e.g., 'cpu', 'cuda:0').

    Returns:
        T: The object moved to the specified device, if applicable. If the object type is not
           directly handled by the function, the original object is returned unchanged.
    """
    if isinstance(obj, torch.Tensor | torch.nn.Module):
        return obj.to(device)
    if is_dataclass_instance(obj):
        return obj.__class__(*(move_to_device(x, device) for x in dataclasses.astuple(obj)))
    if is_namedtuple_instance(obj):
        return obj.__class__(*(move_to_device(x, device) for x in obj))
    if isinstance(obj, Sequence):
        return obj.__class__(move_to_device(x, device) for x in obj)
    if isinstance(obj, Mapping):
        return obj.__class__((k, move_to_device(v, device)) for k, v in obj.items())
    return obj



def unravel_index(
        index: int | torch.Tensor,
        shape: tuple[int, ...],
        order: Literal['C', 'F'] = 'C',
) -> tuple[torch.Tensor, ...]:
    """
    Unravels a flat index into coordinate indices for an array of the given shape.

    Args:
        index (Union[int, torch.Tensor]): The flat index or tensor of indices to unravel.
        shape (Tuple[int, ...]): The shape of the array for which the indices are to be unraveled.
        order (Literal['C', 'F']): The order of the array, 'C' for row-major (C-style) or
                                   'F' for column-major (Fortran-style) order.

    Returns:
        Tuple[torch.Tensor, ...]: A tuple of tensors representing the coordinates of the unraveled index.

    Raises:
        ValueError: If the index is out of bounds for the array with the given shape.
    """
    if isinstance(index, int):
        index = torch.tensor(index)

    # Validate index
    size = math.prod(shape)
    if not torch.all((index >= 0) & (index < size)):
        oob_index = index[(index < 0) | (index >= size)][0]
        raise ValueError(f'Index {oob_index.item()} is out of bounds for array with size {size}')

    # Unravel coordinates
    coords = []
    if order == 'C':
        shape = reversed(shape)
    for dim in shape:
        coords.append(index % dim)
        index //= dim
    if order == 'C':
        coords = reversed(coords)

    return tuple(coords)



def make_first_subword_mask(word_ids: Sequence[Optional[int]]) -> torch.BoolTensor:
    """
    Creates a boolean tensor indicating the positions of the first subword token
    for each word in a sequence, based on the provided list of word IDs.

    Args:
        word_ids (Sequence[Optional[int]]): A sequence of word IDs, where each word ID corresponds
            to a subword token. The word ID is None for special tokens.

    Returns:
        torch.BoolTensor: A boolean tensor where `True` indicates the first subword token of each word.

    Example:
        >>> word_ids = [None, 0, 1, 1, 2, None]
        >>> create_initial_subword_mask(word_ids)
        tensor([False,  True,  True, False,  True, False])
    """
    return torch.tensor([
        word_id is not None and (i == 0 or word_id != word_ids[i-1])
        for i, word_id in enumerate(word_ids)
    ])



###############
# PyTorch Hooks
###############



# Type Aliases
Grads = torch.Tensor | tuple[torch.Tensor, ...]
TensorHookCallable = Callable[['TensorHook', torch.Tensor], Optional[torch.Tensor]]
TensorPostAccumulateGradHookCallable = Callable[['TensorPostAccumulateGradHook', torch.Tensor], None]
ModuleForwardHookCallable = Callable[['ModuleForwardHook', torch.nn.Module, Args, KwArgs, Any], Optional[Any]]
ModulePreForwardHookCallable = Callable[['ModulePreForwardHook', torch.nn.Module, Args, KwArgs], Optional[tuple[Any, KwArgs]]]
ModuleBackwardHookCallable = Callable[['ModuleBackwardHook', torch.nn.Module, Grads, Grads], Optional[Grads]]
ModulePreBackwardHookCallable = Callable[['ModulePreBackwardHook', torch.nn.Module, Grads], Optional[Grads]]


class TorchHook(Hook):
    """
    A hook class for managing PyTorch hooks.

    This class extends the functionality of the Hook class for use with PyTorch,
    providing a way to manage hooks associated with PyTorch operations.
    """

    def __init__(self):
        """
        Initializes a TorchHook instance.
        """
        super().__init__()
        self._handle: Optional[RemovableHandle] = None

    def unregister_hook(self):
        """
        Unregisters the PyTorch hook.

        If the hook is registered, it removes the hook from the PyTorch system and resets the handle.
        """
        super().unregister_hook()
        if self._handle is None:
            raise HookNotRegisteredError('Hook is not currently registered in PyTorch')
        self._handle.remove()
        self._handle = None



class TensorHook(TorchHook):
    """
    A hook for PyTorch tensors.

    This class is used to attach a hook to a PyTorch tensor. The hook function can be provided
    either directly as an argument or by overriding the `hook_function` method in a subclass.
    The hook will be executed whenever the tensor participates in a backward pass.

    Attributes:
        tensor (torch.Tensor): The tensor to which the hook will be attached.
    """

    def __init__(
            self,
            tensor: torch.Tensor,
            *,
            hook_function: Optional[TensorHookCallable] = None,
    ):
        """
        Initializes a TensorHook instance.

        Args:
            tensor (torch.Tensor): The tensor to attach the hook to.
            hook_function (Optional[TensorHookCallable]): An optional callable that is invoked when the hook triggers.
                If not provided, the `hook_function` method must be overridden.
        """
        super().__init__()
        self.tensor = tensor
        self._hook_function = hook_function

    def register_hook(self):
        """
        Registers the hook with the PyTorch tensor.

        The hook function is attached to the tensor, and will be called whenever the tensor
        participates in a backward pass. If the hook function is not set, an error is raised.
        """
        super().register_hook()
        if self._handle is not None:
            raise HookAlreadyRegisteredError('Hook is already registered with the tensor')
        self.handle = self.tensor.register_hook(self.hook_function)

    def hook_function(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """
        The function to be called when the hook triggers.

        This method should be overridden if a hook function is not provided during initialization.
        The default implementation calls the provided hook function, if available, or raises
        NotImplementedError otherwise.

        Args:
            tensor (torch.Tensor): The tensor involved in the backward pass.

        Returns:
            Optional[torch.Tensor]: The result of the hook function, if any.

        Raises:
            NotImplementedError: If no hook function is provided or implemented.
        """
        if self._hook_function is None:
            raise NotImplementedError('Hook function is not implemented')
        return self._hook_function(self, tensor)



class TensorPostAccumulateGradHook(TorchHook):
    """
    A hook class for PyTorch tensors, specifically for post-accumulate gradient operations.

    This class is used to attach a hook that is triggered after gradients are accumulated in a tensor
    during the backward pass. The hook function can be provided directly as an argument or by overriding
    the `hook_function` method in a subclass.

    Attributes:
        tensor (torch.Tensor): The tensor to which the hook will be attached.
    """

    def __init__(
            self,
            tensor: torch.Tensor,
            *,
            hook_function: Optional[TensorPostAccumulateGradHookCallable] = None,
    ):
        """
        Initializes a TensorPostAccumulateGradHook instance.

        Args:
            tensor (torch.Tensor): The tensor to attach the hook to.
            hook_function (Optional[TensorPostAccumulateGradHookCallable]): An optional callable to be invoked
                when the hook triggers. If not provided, the `hook_function` method must be overridden.
        """
        super().__init__()
        self.tensor = tensor
        self._hook_function = hook_function

    def register_hook(self):
        """
        Registers the hook with the PyTorch tensor for post-accumulate gradient operations.

        The hook function is attached to the tensor, and will be called after gradient accumulation
        during the backward pass. If the hook is already registered, a HookAlreadyRegisteredException is raised.

        Raises:
            HookAlreadyRegisteredException: If the hook is already registered with the tensor.
        """
        super().register_hook()
        if self._handle is not None:
            raise HookAlreadyRegisteredError('Hook is already registered with the tensor')
        self.handle = self.tensor.register_post_accumulate_grad_hook(self.hook_function)

    def hook_function(self, tensor: torch.Tensor):
        """
        The function to be called when the hook triggers.

        This method invokes the hook function provided during the initialization of the object. If no hook function
        was provided, it raises a NotImplementedError. This function always returns None, regardless of
        the hook function's result.

        Args:
            tensor (torch.Tensor): The tensor involved in the post-accumulate gradient operation.

        Raises:
            NotImplementedError: If no hook function is provided.
        """
        if self._hook_function is None:
            raise NotImplementedError('Hook function is not implemented')
        self._hook_function(self, tensor)



class ModuleForwardHook(TorchHook):
    """
    A hook class for PyTorch modules, specifically for forward operations.

    This class is used to attach a hook to a PyTorch module's forward pass. The hook function can be provided
    directly as an argument or by overriding the `hook_function` method in a subclass. The hook will be executed
    during the forward pass of the module.

    Attributes:
        module (torch.nn.Module): The module to which the hook will be attached.
        prepend (bool): Determines if the hook should be executed before other registered forward hooks.
    """

    def __init__(
            self,
            module: torch.nn.Module,
            *,
            prepend: bool = False,
            hook_function: Optional[ModuleForwardHookCallable] = None,
    ):
        """
        Initializes a ModuleForwardHook instance.

        Args:
            module (torch.nn.Module): The module to attach the hook to.
            prepend (bool, optional): If True, the hook is added to the beginning of the hook list.
                Defaults to False.
            hook_function (Optional[ModuleForwardHookCallable], optional): An optional callable invoked
                during the module's forward pass. If not provided, the `hook_function` method must be overridden.
        """
        super().__init__()
        self.module = module
        self.prepend = prepend
        self._hook_function = hook_function

    def register_hook(self):
        """
        Registers the hook with the PyTorch module for the forward pass.

        The hook function is attached to the module, and will be called during the module's forward pass.
        If the hook is already registered, a HookAlreadyRegisteredException is raised.

        Raises:
            HookAlreadyRegisteredException: If the hook is already registered with the module.
        """
        super().register_hook()
        if self._handle is not None:
            raise HookAlreadyRegisteredError('Hook is already registered with the module')
        self.handle = self.module.register_forward_hook(
            hook=self.hook_function,
            prepend=self.prepend,
            with_kwargs=True,
        )

    def hook_function(
            self,
            module: torch.nn.Module,
            args: Args,
            kwargs: KwArgs,
            output: Any,
    ) -> Optional[Any]:
        """
        The function to be called during the module's forward pass.

        This method invokes the hook function provided during initialization. If no hook function
        was provided, it raises a NotImplementedError.

        Args:
            module (torch.nn.Module): The module executing the forward pass.
            args (Args): Positional arguments passed to the module's forward method.
            kwargs (KwArgs): Keyword arguments passed to the module's forward method.
            output (Any): The output of the module's forward method.

        Returns:
            Optional[Any]: The result of the hook function, if any.

        Raises:
            NotImplementedError: If no hook function is provided.
        """
        if self._hook_function is None:
            raise NotImplementedError('Hook function is not implemented')
        return self._hook_function(self, module, args, kwargs, output)



class ModulePreForwardHook(TorchHook):
    """
    A hook class for PyTorch modules, specifically for pre-forward operations.

    This class is used to attach a hook to a PyTorch module's pre-forward pass, allowing custom operations or
    modifications before the forward method of the module is called.

    Attributes:
        module (torch.nn.Module): The module to which the pre-forward hook will be attached.
        prepend (bool): Determines if the hook should be executed before other registered pre-forward hooks.
    """

    def __init__(
            self,
            module: torch.nn.Module,
            *,
            prepend: bool = False,
            hook_function: Optional[ModulePreForwardHookCallable] = None,
    ):
        """
        Initializes a ModulePreForwardHook instance.

        Args:
            module (torch.nn.Module): The module to attach the hook to.
            prepend (bool, optional): If True, the hook is added to the beginning of the hook list.
                Defaults to False.
            hook_function (Optional[ModulePreForwardHookCallable], optional): An optional callable invoked
                during the module's pre-forward pass.
        """
        super().__init__()
        self.module = module
        self.prepend = prepend
        self._hook_function = hook_function

    def register_hook(self):
        """
        Registers the hook with the PyTorch module for the pre-forward pass.

        The hook function is attached to the module and will be called before the module's forward pass.
        If the hook is already registered, a HookAlreadyRegisteredException is raised.

        Raises:
            HookAlreadyRegisteredException: If the hook is already registered with the module.
        """
        super().register_hook()
        if self._handle is not None:
            raise HookAlreadyRegisteredError('Hook is already registered with the module')
        self.handle = self.module.register_forward_pre_hook(
            hook=self.hook_function,
            prepend=self.prepend,
            with_kwargs=True,
        )

    def hook_function(
            self,
            module: torch.nn.Module,
            args: Args,
            kwargs: KwArgs,
    ) -> Optional[tuple[Any, dict[str, Any]]]:
        """
        The function to be called during the module's pre-forward pass.

        This method invokes the hook function provided during initialization. If no hook function
        was provided, it raises a NotImplementedError.

        Args:
            module (torch.nn.Module): The module executing the pre-forward pass.
            args (Args): Positional arguments passed to the module's forward method.
            kwargs (KwArgs): Keyword arguments passed to the module's forward method.

        Returns:
            Optional[Tuple[Any, Dict[str, Any]]]: The result of the hook function, which can modify
            the arguments passed to the forward method.

        Raises:
            NotImplementedError: If no hook function is provided.
        """
        if self._hook_function is None:
            raise NotImplementedError('Hook function is not implemented')
        return self._hook_function(self, module, args, kwargs)



class ModuleBackwardHook(TorchHook):
    """
    A hook class for PyTorch modules, specifically for backward operations.

    This class is used to attach a hook to a PyTorch module's backward pass. The hook function can be provided
    either directly as an argument or by overriding the `hook_function` method in a subclass. The hook will be
    executed during the backward pass of the module.

    Attributes:
        module (torch.nn.Module): The module to which the backward hook will be attached.
        prepend (bool): Determines if the hook should be executed before other registered backward hooks.
    """

    def __init__(
            self,
            module: torch.nn.Module,
            *,
            prepend: bool = False,
            hook_function: Optional[ModuleBackwardHookCallable] = None,
    ):
        """
        Initializes a ModuleBackwardHook instance.

        Args:
            module (torch.nn.Module): The module to attach the hook to.
            prepend (bool, optional): If True, the hook is added to the beginning of the hook list.
                Defaults to False.
            hook_function (Optional[ModuleBackwardHookCallable], optional): An optional callable
                invoked during the module's backward pass.
        """
        super().__init__()
        self.module = module
        self.prepend = prepend
        self._hook_function = hook_function

    def register_hook(self):
        """
        Registers the hook with the PyTorch module for the backward pass.

        The hook function is attached to the module, and will be called during the module's backward pass.
        If the hook is already registered, a HookAlreadyRegisteredException is raised.

        Raises:
            HookAlreadyRegisteredException: If the hook is already registered with the module.
        """
        super().register_hook()
        if self._handle is not None:
            raise HookAlreadyRegisteredError('Hook is already registered with the module')
        self.handle = self.module.register_full_backward_hook(
            hook=self.hook_function,
            prepend=self.prepend,
        )

    def hook_function(
            self,
            module: torch.nn.Module,
            grad_input: Grads,
            grad_output: Grads,
    ) -> Optional[Grads]:
        """
        The function to be called during the module's backward pass.

        This method invokes the hook function provided during initialization. If no hook function
        was provided, it raises a NotImplementedError. The hook can potentially modify the gradients.

        Args:
            module (torch.nn.Module): The module executing the backward pass.
            grad_input (Grads): The gradients input to the module's backward method.
            grad_output (Grads): The gradients output from the module's backward method.

        Returns:
            Optional[Grads]: The result of the hook function, which can modify the gradients.

        Raises:
            NotImplementedError: If no hook function is provided.
        """
        if self._hook_function is None:
            raise NotImplementedError('Hook function is not implemented')
        return self._hook_function(self, module, grad_input, grad_output)



class ModulePreBackwardHook(TorchHook):
    """
    A hook class for PyTorch modules, specifically for operations before the backward pass.

    This class is used to attach a hook to a PyTorch module right before its backward pass begins. The hook
    function can be provided either directly as an argument or by overriding the `hook_function` method in a
    subclass. The hook will be executed just before the backward pass of the module.

    Attributes:
        module (torch.nn.Module): The module to which the pre-backward hook will be attached.
        prepend (bool): Determines if the hook should be executed before other registered pre-backward hooks.
    """

    def __init__(
            self,
            module: torch.nn.Module,
            *,
            prepend: bool = False,
            hook_function: Optional[ModulePreBackwardHookCallable] = None,
    ):
        """
        Initializes a ModulePreBackwardHook instance.

        Args:
            module (torch.nn.Module): The module to attach the hook to.
            prepend (bool, optional): If True, the hook is added to the beginning of the hook list.
                Defaults to False.
            hook_function (Optional[ModulePreBackwardHookCallable], optional): An optional callable invoked
                just before the module's backward pass.
        """
        super().__init__()
        self.module = module
        self.prepend = prepend
        self._hook_function = hook_function

    def register_hook(self):
        """
        Registers the hook with the PyTorch module for the pre-backward pass.

        The hook function is attached to the module and will be called just before the module's backward pass.
        If the hook is already registered, a HookAlreadyRegisteredException is raised.

        Raises:
            HookAlreadyRegisteredException: If the hook is already registered with the module.
        """
        super().register_hook()
        if self._handle is not None:
            raise HookAlreadyRegisteredError('Hook is already registered with the module')
        self.handle = self.module.register_full_backward_pre_hook(
            hook=self.hook_function,
            prepend=self.prepend,
        )

    def hook_function(
            self,
            module: torch.nn.Module,
            grad_output: Grads,
    ) -> Optional[Grads]:
        """
        The function to be called just before the module's backward pass.

        This method invokes the hook function provided during initialization. If no hook function
        was provided, it raises a NotImplementedError. The hook can potentially modify the gradients
        before the backward pass.

        Args:
            module (torch.nn.Module): The module executing the pre-backward pass.
            grad_output (Grads): The gradients output that will be used in the module's backward method.

        Returns:
            Optional[Grads]: The result of the hook function, which can modify the gradients before the backward pass.

        Raises:
            NotImplementedError: If no hook function is provided.
        """
        if self._hook_function is None:
            raise NotImplementedError('Hook function is not implemented')
        return self._hook_function(self, module, grad_output)
