# Backtest

Creatively named to intelligently mask the creativity of the author. Ha, reverse psychology!

Part of my strategy backtesting framework. A combination of automatic annotation of buy/sell signals and statistical learning algorithms.
I sort of believe that cryptocurrency market is low in signal-to-noise ratio, with how erratic the prices can move, most of the times breaking
all your hypotheses. However, it serves as a good testbed for the strategies and everybody (both the wise and unwise in decision making and
their uncles and their goldfishes) is into it..

Is that a possible *alpha* source I see? :o

## User Guide

In case I lose my memory in the near future..

```
Dear future me, the IPython notebook technicals.ipynb contains random technical indicators backtested and overfit.
The responsibility to cross-validate is yours.

The IPython notebook backtest.ipynb contains annotation (currently implemented: kernelized n-degree peak/trough signals).
There are some machine learning algorithms blended distastefully for the inference - luckily it's properly cross validated.
Your job is to provide useful data and signals so it works well.
```

---

Footnote: I need a good data source because the performance of my models with even my steroid-injected technical indicators aren't exactly encouraging.
Pardon the cheese, I am bored.
