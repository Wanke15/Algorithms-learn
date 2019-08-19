from abc import ABC, abstractmethod

from functools import lru_cache


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

    @lru_cache(maxsize=32)
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


class LCS(BaseDP):
    def __init__(self):
        super().__init__()


    def get_all_subsequence(self, seq):
        pass

    def violence(self, seq1, seq2):
        res = ''
        for id1, s1 in enumerate(seq1):
            for id2, s2 in enumerate(seq2):
                pass



    def recursion(self, seq1, seq2):
        pass

    def dp(self, seq1, seq2):
        pass
