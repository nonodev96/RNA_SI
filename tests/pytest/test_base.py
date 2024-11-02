from src.base import BASE_NAME
from src.hello_world import hello_world


def test_base():
    assert BASE_NAME == "rna_si"


def test_hello():
    hello_world()
