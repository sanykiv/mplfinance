"""
Regression tests for the `volume_color_updown_dependancy` market-colors setting,
which controls how the up/down color of each volume bar is decided:

    "Open vs Close"    (default) : today's open  vs today's close (same as candles)
    "Previous Close"             : today's close vs the previous close
    "Previous Volume"            : today's volume vs the previous volume

Value matching is case- and space-insensitive.
Also verifies backward compatibility with the legacy `vcdopcod` setting.
"""

import numpy             as np
import pandas            as pd
import mplfinance        as mpf
import matplotlib.pyplot as plt
import pytest

print('mpf.__version__ =', mpf.__version__)                  # for the record


def _sample_df():
    idx   = pd.date_range('2024-01-01', periods=8, freq='D')
    close = np.array([10, 11, 10.5, 12, 11.8, 13, 12.5, 14.0])
    # Volume deliberately rises/falls independently of price, so that the
    # 'Previous Volume' coloring is distinguishable from the price-based modes.
    vol   = np.array([100, 80, 120, 90, 150, 60, 200, 50.0])
    df = pd.DataFrame({'Open': close - 0.2, 'High': close + 0.5,
                       'Low': close - 0.5, 'Close': close, 'Volume': vol}, index=idx)
    df.index.name = 'Date'
    return df


def _volume_bar_colors(df, **mktcolor_kwargs):
    """Return the list of face colors of the volume bars for the given marketcolors."""
    mc    = mpf.make_marketcolors(up='g', down='r', volume={'up': 'g', 'down': 'r'},
                                  **mktcolor_kwargs)
    style = mpf.make_mpf_style(marketcolors=mc)
    fig, axlist = mpf.plot(df, type='candle', volume=True, style=style, returnfig=True)
    volume_ax = axlist[2]  # panel holding the volume bars
    colors = [tuple(np.round(patch.get_facecolor(), 3)) for patch in volume_ax.patches]
    plt.close(fig)
    return colors


def test_volume_color_modes_differ():
    df = _sample_df()
    open_vs_close  = _volume_bar_colors(df, volume_color_updown_dependancy='Open vs Close')
    previous_vol   = _volume_bar_colors(df, volume_color_updown_dependancy='Previous Volume')

    # The whole point of the feature: coloring by volume differs from the default.
    assert previous_vol != open_vs_close
    assert len(previous_vol) == len(df)


def test_case_and_space_insensitive():
    df = _sample_df()
    canonical = _volume_bar_colors(df, volume_color_updown_dependancy='Previous Volume')
    for variant in ('previous volume', 'PREVIOUSVOLUME', '  Previous  Volume '):
        assert _volume_bar_colors(df, volume_color_updown_dependancy=variant) == canonical


def test_backward_compatible_with_vcdopcod():
    df = _sample_df()
    legacy   = _volume_bar_colors(df, vcdopcod=True)
    explicit = _volume_bar_colors(df, volume_color_updown_dependancy='Previous Close')
    assert legacy == explicit


def test_invalid_value_raises():
    with pytest.raises(Exception):
        mpf.make_marketcolors(volume_color_updown_dependancy='not a real mode')
