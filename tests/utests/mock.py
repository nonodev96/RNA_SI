import unittest
import mock

from src.examples.module1 import Random


class RandomTest(unittest.TestCase):

    @mock.patch("os.urandom")
    def test_Random(self, mocker):
        mocker.return_value = "aaa"
        self.assertEqual(Random(2), "2aaa")
