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
