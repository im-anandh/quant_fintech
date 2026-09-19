import numpy as np
import pandas as pd
from src.indicators import sma, max_drawdown


def test_sma():
    assert sma(pd.Series([1, 2, 3, 4]), 2).iloc[-1] == 3.5


def test_drawdown():
    assert max_drawdown(pd.Series([100, 120, 90, 110])) == -0.25
