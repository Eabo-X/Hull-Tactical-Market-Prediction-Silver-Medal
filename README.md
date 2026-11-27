# Hull Tactical – Market Prediction (Silver Medal)

A Kaggle submission for predicting daily S&P 500 excess returns and turning each forecast into an index position. A position of `0` holds cash, `1` tracks the index, and `2` doubles the index exposure. Kaggle evaluates the daily positions with an adjusted Sharpe ratio. [Competition overview](https://www.kaggle.com/competitions/hull-tactical-market-prediction)

## Model

The notebook uses up to 800 of the most recent labeled trading days with `date_id >= 37`. The target is `market_forward_excess_returns`. It keeps numbered market factors with at least 50% coverage in that window and adds five features when their inputs are available.

| Feature | Definition |
| --- | --- |
| `U1` | `I2 - I1` |
| `U2` | `M11 / ((I2 + I7 + I9) / 3)` |
| Cross-products | `V1 * S1`, `M11 * V1`, `I9 * S1` |

During fitting, interest-rate factors (`I*`) are forward-filled and then backward-filled within the training window. Other gaps, including undefined ratios, use training medians. A derived feature with no finite training values falls back to zero. At prediction time, the same medians and scaler are used for both models.

| Regressor | Trees | Learning rate |
| --- | ---: | ---: |
| XGBoost (depth 6) | 300 | 0.05 |
| LightGBM | 200 | 0.03 |

## Daily allocation

The models are fitted once per submission. Each day, their forecasts receive equal weight. The position starts at zero.

```text
forecast[t] = (xgboost[t] + lightgbm[t]) / 2
target[t]   = clip(500 × forecast[t], 0, 2)
position[t] = 0.75 × target[t] + 0.25 × position[t-1]
```

## Run

1. Import the notebook into a Kaggle Code session. [maincode.ipynb](maincode.ipynb)
2. Attach the competition data, including `train.csv`, `test.csv`, and the supplied `kaggle_evaluation` package. [Competition data](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data)
3. Run all cells. An interactive run sends the example test days through the local gateway. A competition rerun starts the inference server. Set `HULL_DATA_DIR` if the data is mounted elsewhere.

For a chronological check, set `HOLDOUT_DAYS = 120` in the first cell. The replay leaves a one-trading-day gap after training, then processes the last 120 days in order. It reports compounded strategy and buy-and-hold returns before costs. The default value of `0` skips the replay.
