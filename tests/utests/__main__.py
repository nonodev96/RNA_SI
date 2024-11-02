import unittest

from tests.utests.test_mock import RandomTest
from tests.utests.test_module1 import TestSimple


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TestSimple())
    suite.addTest(RandomTest())
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())