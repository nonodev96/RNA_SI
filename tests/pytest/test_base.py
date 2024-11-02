from src.rna_si.base import BASE_NAME
from src.rna_si.hello_world import hello_world


def test_base():
    assert BASE_NAME == "rna_si"


def test_hello():
    hello_world()
