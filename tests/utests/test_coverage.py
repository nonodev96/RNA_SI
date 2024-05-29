import unittest
from unittest.mock import MagicMock


class MyClass:
    def my_method(self, a, b):
        # Some complex operation
        pass


class TestMyClass(unittest.TestCase):
    def test_my_method(self):
        my_instance = MyClass()
        my_instance.my_method = MagicMock(return_value=42)

        result = my_instance.my_method(2, 3)

        self.assertEqual(result, 42)
        my_instance.my_method.assert_called_with(2, 3)
