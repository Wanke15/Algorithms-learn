from abc import ABC, abstractmethod


class BaseDP(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def recursion(self, *args, **kwargs):
        pass

    @abstractmethod
    def dp(self, *args, **kwargs):
        pass


class Stair(BaseDP):
    def __init__(self):
        super().__init__()

    def recursion(self, n):
        if n <= 2:
            return n
        return self.recursion(n - 1) + self.recursion(n - 2)

    def dp(self, n):
        ways = [1, 2]
        if n <= 2:
            return n
        for i in range(2, n):
            ways.append(ways[i-1]+ways[i-2])
        return ways[-1]


class Stock(BaseDP):
    def __init__(self):
        super().__init__()

    def recursion(self, prices):
        if len(prices) < 2:
            return 0
        return max(prices[-1] - min(prices[:-1]), self.recursion(prices[:-1]))

    def dp(self, prices):
        result = [0]
        if len(prices) < 2:
            return 0
        minPrice = prices[0]
        for i in range(1, len(prices)):
            minPrice = min(minPrice, prices[i - 1])
            result.append(max(prices[i] - minPrice, result[i - 1]))
        return result[-1]
