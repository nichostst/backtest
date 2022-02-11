import numpy as np
from vectorbt import IndicatorFactory as vi
from find import ExtremeFind


class MomentumSignals(object):
    def __init__(self, price, mom_period=24):
        self.ef = ExtremeFind()
        self.price = price
        self.mom = vi.from_talib('MOM').run(price, timeperiod=mom_period)

    def in_trough_signal(self, mom_trough=True):
        '''
        :param mom_trough bool: To use momentum trough as signal
        :param n int: Order of trough
        '''
        # Troughs (Buy signals)
        if mom_trough:
            trough_underlying = self.mom.real
        else:
            trough_underlying = self.price

        trough_idx = self.ef.find_trough(trough_underlying)[0]
        in_trough_idx = self.ef.get_in_troughs(trough_underlying, n=2)[0]

        # In-trough confirmation
        buy_idx = [trough_idx[trough_idx > x][0] for x in in_trough_idx]
        in_trough_confirmation = self.mom.real.iloc[buy_idx]
        in_trough_confirmation = (np.sign(-in_trough_confirmation) + 1)/2

        trough_signal = in_trough_confirmation.reindex(self.mom.real.index).fillna(0).astype(bool)
        return trough_signal

    def momentum_consistency(self, ma_period):
        momentum_consistency = vi.from_talib('MA').run(np.sign(self.mom.real), timeperiod=ma_period)
        return momentum_consistency

    def momentum_buy_signal(self, ma_period=24, thresh=0.9):
        return self.momentum_consistency(ma_period).real_crossed_above(thresh)

    def momentum_sell_signal(self, ma_period=24, thresh=0.1):
        return self.momentum_consistency(ma_period).real_crossed_below(thresh)

    def dying_momentum_signal(self, ma_period=168):
        # For now use absolute value
        mom_deviation = np.abs(self.mom.real)
        ma_md = vi.from_talib('MA').run(mom_deviation, timeperiod=ma_period)
        return ma_md.real_crossed_below(self.mom)

class DumpSignals(object):
    def __init__(self, price, ma_period=84, dma_period=3):
        self.price = price
        self.ma_ratio = vi.from_talib('MA').run(price, timeperiod=ma_period).real/price
        self.dma_ratio = vi.from_talib('MA').run(self.ma_ratio, timeperiod=dma_period).real/self.ma_ratio

    def dump_sell_signal(self, thres=1.2):
        return self.ma_ratio.vbt.crossed_above(thres)

    def recovery_buy_signal(self, ma_thres=1.15, dma_thres=1.05):
        return (self.ma_ratio > ma_thres) & (self.dma_ratio > dma_thres)

class DrawdownSignals(object):
    def __init__(self, price):
        self.price = price

    def lm_drawdown(self, dd_period):
        # Limited-memory drawdown
        lmdrawdown = 1 - self.price/self.price.rolling(dd_period).max()
        return lmdrawdown

    def drawdown_buy_signal(self, dd_period=84, thres=0.2):
        return self.lm_drawdown(dd_period).vbt.crossed_above(thres)

class VolumeSignals(object):
    def __init__(self, vol, ma_period=42, dma_period=3):
        self.vol = vol
        self.ma_ratio = vol/vi.from_talib('MA').run(vol, timeperiod=ma_period).real
        self.dma_ratio = self.ma_ratio/vi.from_talib('MA').run(self.ma_ratio, timeperiod=dma_period).real

    def vol_up_signal(self, thres=1.2):
        return self.ma_ratio > thres

    def vol_stabilise_signal(self, ma_thres=1.1, dma_thres=1.05):
        return (self.ma_ratio > ma_thres) & self.dma_ratio.vbt.crossed_below(dma_thres)

class OscillatorSignals(object):
    def __init__(self, price):
        self.price = price
        self.bband = vi.from_talib('BBANDS').run(price)
        self.rsi = vi.from_talib('RSI').run(price).real

    def bband_position(self):
        bandrange = self.bband.upperband - self.bband.lowerband
        bandpos = (self.price - self.bband.middleband)/bandrange
        return bandpos
