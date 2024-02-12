import pandas as pd


def test_dataframe():
    data = {"col1": [1, 2], "col2": [3, 4]}
    df = pd.DataFrame(data)

    assert df.empty() == False
