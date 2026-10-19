| Configuration | Seeds | Test PSNR (dB) | Gain vs baseline (dB) | Val MSE measured | Val MSE in cone | s/epoch |
|---|---|---|---|---|---|---|
| consistency_direct_e20_lr0.003_final | 3 | 16.29 ± 0.09 | +0.04 ± 0.09 | 7.53e-06 | 2.65e-02 | 65 |
| equivariant_residual_a3_detach_e20_lr0.003_final | 3 | 34.34 ± 0.33 | +18.09 ± 0.33 | 1.53e-04 | 3.69e-04 | 127 |
| supervised_residual_e20_lr0.003_final | 3 | 35.80 ± 0.47 | +19.55 ± 0.47 | 1.14e-04 | 3.03e-04 | 61 |

Baseline test PSNR (x_hat = y): 16.25 dB. ± is the sample standard deviation (n-1) over seeds.
