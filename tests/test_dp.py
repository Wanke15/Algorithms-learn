import unittest

from DinamicPrpgramming.dp import Stair
from utils.timer import timer


class TestDP(unittest.TestCase):
    def setUp(self):
        self.stair = Stair()
        self.steps = 35

    @timer
    def test_recursion(self):
        res = self.stair.recursion(self.steps)
        print(res)

    @timer
    def test_dp(self):
        res = self.stair.dp(self.steps)
        print(res)
