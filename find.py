import numpy as np


class ExtremeFind(object):
    def __init__(self):
        pass

    @staticmethod
    def find_peak(x):
        direction = 0
        peak_idx = []
        peak = []
        for i in range(len(x)-1):
            if x[i] <= x[i+1]:
                direction = 1
            else:
                if direction == 1:
                    peak_idx.append(i)
                    peak.append(x[i])
                direction = -1
        return np.array(peak_idx), np.array(peak)

    @staticmethod
    def find_trough(x):
        trough_idx, trough_neg = ExtremeFind.find_peak(-x)
        return trough_idx, -trough_neg

    @staticmethod
    def get_in_peaks(x, n=2):
        peak_idx, in_peak = ExtremeFind.find_peak(x)
        loop_no = 1
        while loop_no < n:
            in_peak_idx, in_peak = ExtremeFind.find_peak(in_peak)
            peak_idx = peak_idx[in_peak_idx]
            loop_no += 1
        return peak_idx, in_peak

    @staticmethod
    def get_in_troughs(x, n=2):
        trough_idx, in_trough_neg = ExtremeFind.get_in_peaks(-x, n)
        return trough_idx, -in_trough_neg

    def _get_peak_trough(self, series):
        i = 1
        p = {}
        t = {}
        pflag = True
        tflag = True
        while pflag or tflag:
            try:
                if pflag:
                    p[i] = self.get_in_peaks(series, n=i)[0]
            except IndexError:
                pflag = False
                print(f'Stopped at {i}th degree in peak finding.')

            try:
                if tflag:
                    t[i] = self.get_in_troughs(series, n=i)[0]
            except IndexError:
                tflag = False
                print(f'Stopped at {i}th degree in trough finding.')
            i += 1
        return p, t

    def annotate_direction(self, series):
        p, t = self._get_peak_trough(series)
        out = series.reset_index()
        out = out.assign(peak=0, trough=0, pt=0)

        for i, pind in p.items():
            out.loc[pind, 'peak'] = i
            out.loc[pind, 'pt'] = i

        for i, tind in t.items():
            out.loc[tind, 'trough'] = i
            out.loc[tind, 'pt'] = -i

        out = out.set_index('time')
        return out

def get_optimal_signals(prices):
    ef = ExtremeFind()
    in_peak_idx, _ = ef.get_in_peaks(prices['close'], n=3)
    in_trough_idx, _ = ef.get_in_troughs(prices['close'], n=3)

    pt = prices.copy().assign(signal=None)
    pt.loc[pt.index[in_peak_idx], 'signal'] = -1
    pt.loc[pt.index[in_trough_idx], 'signal'] = 1
    pt['pos'] = pt['signal'].bfill()
    pt = pt.fillna(0)
    signals = pt[pt['signal'] != 0]
    signals = signals[signals['signal'].diff() != 0]
    return signals