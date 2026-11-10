metric: `val_bpb` (lower is better)

```
              n_seeds  val_bpb_mean  val_bpb_std  params_M  train_min  delta_vs_gpt_textbook  paired_p
name                                                                                                  
gpt_modern          2        2.1052       0.0012    3.1626     0.5922                -0.0839    0.0248
lstm                2        2.1351       0.0057    1.0698     0.6693                -0.0539    0.0958
gru                 2        2.1639       0.0017    0.8067     0.6237                -0.0251    0.1338
lstm_512            2        2.1747       0.0088    4.2368     0.5659                -0.0143    0.3993
gpt_textbook        2        2.1891       0.0059    3.4383     0.4197                 0.0000       NaN
rnn                 2        2.4384       0.0136    0.2803     0.1180                 0.2493    0.0351
```
