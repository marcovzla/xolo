import os
import random
from typing import Optional
from collections.abc import Callable
import torch
from torch.utils.hooks import RemovableHandle
import numpy as np
from xolo.utils.hooks import Hook, HookNotRegisteredException, HookAlreadyRegisteredException



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



def enable_full_determinism(seed: int, warn_only: bool = False):
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



###############
# PyTorch Hooks
###############



TensorHookCallable = Callable[['TensorHook', torch.Tensor], Optional[torch.Tensor]]
TensorPostAccumulateGradHookCallable = Callable[['TensorPostAccumulateGradHook', torch.Tensor], None]



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
            raise HookNotRegisteredException('Hook is not currently registered in PyTorch')
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
            raise HookAlreadyRegisteredException('Hook is already registered in PyTorch')
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
            hook_function: Optional[TensorPostAccumulateGradHookCallable] = None,
    ):
        """
        Initializes a TensorPostAccumulateGradHook instance.

        Args:
            tensor (torch.Tensor): The tensor to attach the hook to.
            hook_function (Optional[TensorPostAccumulateGradHookCallable]): An optional callable to be invoked when the hook triggers.
                If not provided, the `hook_function` method must be overridden.
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
            raise HookAlreadyRegisteredException('Hook is already registered in PyTorch')
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
