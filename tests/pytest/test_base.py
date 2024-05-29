from src.tfm_sai.base import BASE_NAME
from src.tfm_sai.hello_world import hello_world


def test_base():
    assert BASE_NAME == "tfm_sai"


def test_hello():
    hello_world()
