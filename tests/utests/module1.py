import unittest

from src.examples.module1 import Number


class TestSimple(unittest.TestCase):

    def test_add(self):
        self.assertEqual((Number(5) + Number(6)).value, 11)
