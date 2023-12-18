class MovingAverage:
    """
    A class for calculating a simple moving average.

    This class maintains a running total and count of values to efficiently compute the moving average
    of a sequence of numbers.

    Attributes:
        total (float): The cumulative sum of added values.
        count (int): The number of values added.
    """

    def __init__(self):
        """Initialize a new moving average with no values."""
        self.reset()

    def reset(self):
        """Reset the moving average, clearing all added values."""
        self.total = 0.0
        self.count = 0

    @property
    def value(self) -> float:
        """
        Get the current moving average value.

        The moving average is calculated as the total sum of values divided by the number of values.

        Returns:
            float: The current moving average.

        Raises:
            ZeroDivisionError: If no values have been added (count is 0).
        """
        if self.count == 0:
            raise ZeroDivisionError('Cannot calculate the average of zero values')
        return self.total / self.count

    def add(self, value: float, n: int = 1):
        """
        Add a value or multiple instances of the same value to the moving average.

        Args:
            value (float): The value to be added.
            n (int, optional): The number of times the value is to be added. Defaults to 1.
        """
        self.total += n * value
        self.count += n



class ExponentialMovingAverage:
    """
    A class for calculating an Exponential Moving Average (EMA).

    EMA applies more weight to recent data points, making it more responsive to new information.
    It uses a smoothing factor to determine the weight of older data points.

    Attributes:
        beta (float): The smoothing factor, which determines the weight of older data. Typically between 0 and 1.
        count (int): The number of values added.
        biased_avg (float): The current biased EMA value.
    """

    def __init__(self, beta: float = 0.98):
        """
        Initialize a new Exponential Moving Average with a specified smoothing factor.

        Args:
            beta (float, optional): The smoothing factor, determining the weight of older data. Defaults to 0.98.
        """
        self.beta = beta
        self.reset()

    def reset(self):
        """Reset the Exponential Moving Average, clearing all added values."""
        self.count = 0
        self.biased_avg = 0.0

    @property
    def value(self) -> float:
        """
        Get the current Exponential Moving Average value with bias correction.

        EMA value is calculated using a bias correction formula to account for initial values.

        Returns:
            float: The bias-corrected EMA value.

        Raises:
            ZeroDivisionError: If no values have been added (count is 0).
        """
        if self.count == 0:
            raise ZeroDivisionError("Cannot calculate the EMA of zero values")
        return self.biased_avg / (1 - self.beta ** self.count)

    def add(self, value: float, n: int = 1):
        """
        Add a new value (or multiple instances of it) to the Exponential Moving Average.

        The EMA is updated for each instance of the value using the specified smoothing factor.

        Args:
            value (float): The new value to be added to the EMA calculation.
            n (int, optional): The number of times the value is to be added. Defaults to 1.
        """
        for _ in range(n):
            self.count += 1
            self.biased_avg = self.beta * self.biased_avg + (1 - self.beta) * value
