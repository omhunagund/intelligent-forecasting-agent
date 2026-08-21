# Overall Forecast Analysis

This document contains project-derived information about the overall weekly revenue forecasting system.

It does not contain external industry benchmarks or unstated business assumptions.

## Production model decision

XGBoost is the selected production model for the overall forecast because it provides consistent TreeSHAP explanations across all production forecasting levels. LSTM remains a benchmark in the model comparison.

## Model comparison

| model                 |   mean_mae |   mean_rmse |   mean_mape |   mean_interval_coverage_pct |   rank |
|:----------------------|-----------:|------------:|------------:|-----------------------------:|-------:|
| lstm                  |    37281.4 |     43829.3 |     16.1074 |                       81.25  |      1 |
| weighted_ensemble     |    36602.6 |     43384.5 |     17.2322 |                       75     |      2 |
| exponential_smoothing |    42570.1 |     50412.8 |     19.7587 |                       68.75  |      3 |
| prophet               |    44045.2 |     50923.1 |     20.241  |                       46.875 |      4 |
| xgboost               |    41257.9 |     50268.3 |     20.4118 |                       81.25  |      5 |

### Current ranking

The model ranked first in the stored comparison is **lstm**, with mean MAPE of **16.11%**.

The production-model choice is documented separately from the benchmark ranking so that the agent does not misrepresent the model-selection decision.

## Historical ensemble forecast stream

The stored walk-forward ensemble contains **32 forecast observations**.

Validation period covered: **2017-12-31** to **2018-08-05**.

## Source artifacts

- `reports\stage1_overall_model_comparison.csv`
- `reports\stage1_weighted_ensemble_predictions.csv`
