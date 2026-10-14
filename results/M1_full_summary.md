# M1_full -- full channel set (76 inputs, 58 scored)

* total notebook time: 5.58 h
* stages: preflight 1 min (ok), train 22 min (ok), errors 5 min (ok), search 296 min (ok), test 10 min (ok)
* pre-registered config (chosen on TRAIN, F0.5 0.8815): `{'p': 0.3, 'zmode': 'global', 'tmode': 'perchan', 'q': 0.99999, 'close': 60, 'minlen': 3, 'gapL': 10, 'gapmult': 6.0, 'boost': 0.6}`
* search: 305 configs, gpu assembly
* inputs: 76 channels + telecommands; scored: 58 Target=YES channels
* positive rows predicted: 819,081

| metric | ours | best published (full M1) |
|---|---|---|
| Event-wise F0.5   (headline) | 0.4394 | 0.061 (TM-Pruned) |
| Event-wise precision | 0.4452 | 0.050 (TM-Pruned) |
| Event-wise recall | 0.4176 | 0.967 (iForest) |
| Subsystem-aware F0.5 | 0.3429 | 0.689 (TM) |
| Channel-aware F0.5 | 0.1806 | 0.477 (TM) |
| Alarming precision | 0.8085 | 0.206 (TM-Pruned) |
| ADTQC score | 0.4650 | 0.752 (DC-VAE-ESA STD3) |
| Affiliation F0.5 | 0.5870 | 0.607 (TM-Pruned) |

Anomaly-only selection (ESA reports this separately):

* anom_global_EW_F_0.50: 0.3370
* anom_global_EW_precision: 0.3243
* anom_global_EW_recall: 0.4000
* anom_chan_channel_F0.50: 0.1578
* anom_adtqc_Total: 0.3207
