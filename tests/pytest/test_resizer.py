def test_box():
    from src.tfm_sai.resizer import box

    x = 0.7
    box_rate = box(x)
    assert box_rate == 0.0


def test_dataframe():
    import pandas as pd

    data = {"col1": [1, 2], "col2": [3, 4]}
    df = pd.DataFrame(data)
    assert df.empty == False
