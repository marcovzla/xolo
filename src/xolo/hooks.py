from collections.abc import Iterable, Iterator
from typing import Generic, Optional, Self, TypeVar

H = TypeVar('H', bound='Hook')



class Hook:
    """
    A class that represents a hook that can be registered and unregistered.

    This class provides a context manager interface for safely registering and unregistering hooks.
    It ensures that a hook is only registered once and unregistered before deletion if it was registered.
    """

    def __init__(self):
        """Initialize a Hook instance with the registered flag set to False."""
        self._registered: bool = False

    def __enter__(self) -> Self:
        """
        Context manager entry method to register the hook.

        Registers the hook when entering the context. If the hook is already registered,
        an exception is raised.

        Returns:
            Hook: The instance itself.
        """
        self.register_hook()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Context manager exit method to unregister the hook.

        Unregisters the hook when exiting the context. If the hook is not registered,
        an exception is raised.

        Args:
            exc_type: The type of the exception if an exception has occurred, otherwise None.
            exc_value: The exception instance if an exception has occurred, otherwise None.
            traceback: A traceback instance if an exception has occurred, otherwise None.
        """
        self.unregister_hook()

    def __del__(self):
        """
        Destructor method to ensure the hook is unregistered before deletion.

        If the hook is registered at the time of object deletion, it is unregistered.
        """
        if self.is_registered:
            self.unregister_hook()

    @property
    def is_registered(self) -> bool:
        """
        Indicates whether the hook is currently registered.

        Returns:
            bool: True if the hook is registered, False otherwise.
        """
        return self._registered

    def register_hook(self):
        """
        Registers the hook.

        Sets the _registered flag to True. If the hook is already registered,
        raises an exception.

        Raises:
            HookAlreadyRegisteredException: If the hook is already registered.
        """
        if self.is_registered:
            raise HookAlreadyRegisteredError
        self._registered = True

    def unregister_hook(self):
        """
        Unregisters the hook.

        Sets the _registered flag to False. If the hook is not registered,
        raises an exception.

        Raises:
            HookNotRegisteredException: If the hook is not currently registered.
        """
        if not self.is_registered:
            raise HookNotRegisteredError
        self._registered = False



class HookManager(Hook, Generic[H]):
    """
    A manager for handling a collection of Hook instances.

    This class extends the Hook class to manage multiple hook instances collectively.
    It provides methods to register and unregister all managed hooks at once, and supports
    iteration over and modification of the hook collection.
    """

    def __init__(self, hooks: Optional[Iterable[H]] = None):
        """
        Initializes a HookManager with an optional iterable of hooks.

        Args:
            hooks (Optional[Iterable[H]]): An iterable of Hook instances to be managed. Defaults to None.
        """
        super().__init__()
        self.hooks = hooks

    def __len__(self) -> int:
        """
        Returns the number of hooks being managed.

        Returns:
            int: The number of hooks in the manager.
        """
        return len(self.hooks)

    def __iter__(self) -> Iterator[H]:
        """
        Returns an iterator over the managed hooks.

        Returns:
            Iterator[H]: An iterator over the Hook instances.
        """
        return iter(self.hooks)

    @property
    def hooks(self) -> list[H]:
        """
        Gets the list of managed hooks.

        Returns:
            list[H]: A list containing the managed Hook instances.
        """
        return self._hooks

    @hooks.setter
    def hooks(self, hooks: Optional[Iterable[H]]):
        """
        Sets the hooks to be managed.

        If the HookManager is currently registered, attempting to set new hooks will raise an exception.

        Args:
            hooks (Optional[Iterable[H]]): An iterable of Hook instances to be managed.

        Raises:
            HookAlreadyRegisteredException: If trying to set hooks while the manager is registered.
        """
        if self.is_registered:
            raise HookAlreadyRegisteredError('Cannot swap hooks while manager is registered')
        self._hooks = list(hooks) if hooks is not None else []

    def register_hook(self):
        """
        Registers all managed hooks.

        Before registering the individual hooks, this method registers the manager itself.
        If any hook is already registered, a HookAlreadyRegisteredException is raised.

        Raises:
            HookAlreadyRegisteredException: If any hook is already registered.
        """
        super().register_hook()
        for hook in self.hooks:
            hook.register_hook()

    def unregister_hook(self):
        """
        Unregisters all managed hooks.

        Before unregistering the individual hooks, this method unregisters the manager itself.
        If any hook is not registered, a HookNotRegisteredException is raised.

        Raises:
            HookNotRegisteredException: If any hook is not currently registered.
        """
        super().unregister_hook()
        for hook in self.hooks:
            hook.unregister_hook()



class HookError(Exception):
    """
    Base exception for all hook-related errors.

    This class serves as a base for specific hook exceptions, allowing users of the Hook class
    to catch all hook-related exceptions in a single except block if needed. It inherits from the
    standard Exception class.
    """



class HookNotRegisteredError(HookError):
    """Exception raised when attempting to unregister a hook that is not registered."""

    def __init__(self, message='Hook is not currently registered'):
        super().__init__(message)



class HookAlreadyRegisteredError(HookError):
    """Exception raised when attempting to register a hook that is already registered."""

    def __init__(self, message='Hook is already registered'):
        super().__init__(message)
