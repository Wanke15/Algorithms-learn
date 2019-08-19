import unittest

from DinamicPrpgramming.dp import Stair, Stock, LCS
from utils.decorator import timer


class TestDP(unittest.TestCase):
    def setUp(self):
        pass

    @timer
    def test_stair(self):
        stair = Stair()
        steps = 38
        res = stair.recursion(steps)
        print(res)

        res = stair.dp(steps)
        print(res)

    @timer
    def test_stock(self):
        stock = Stock()
        prices = [7, 1, 5, 3, 6, 4]
        prices = [_ for _ in range(950)]
        res = stock.recursion(prices)
        print(res)

        res = stock.dp(prices)
        print(res)

    @timer
    def test_lcs(self):
        lcs = LCS()
        seq1 = 'abazdc'
        seq2 = 'bacbad'
        res = lcs.violence(seq1, seq2)
        print(res)

