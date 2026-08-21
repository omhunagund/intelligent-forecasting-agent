# Regional Forecast Analysis

This document contains project-derived forecast information for the 5 approved regional series.

All regional forecasting is performed using XGBoost with the approved walk-forward validation design.

## Regional model performance

| series_id    |   mean_mape |   mean_rmse |   mean_mae |   mean_interval_coverage_pct |   rank_within_type |
|:-------------|------------:|------------:|-----------:|-----------------------------:|-------------------:|
| Northeast    |     21.6634 |     7020.65 |    5999.27 |                       71.875 |                  1 |
| Southeast    |     24.2391 |    42221.4  |   37250.1  |                       68.75  |                  2 |
| Central-West |     29.3419 |     4954.53 |    4025.56 |                       62.5   |                  3 |
| South        |     34.8885 |    12378.4  |   10693    |                       65.625 |                  4 |
| North        |     49.951  |     2347.72 |    1994.65 |                       78.125 |                  5 |

## Latest regional forecasts

| series_id    | timestamp           |   forecast_revenue |   lower_80 |   upper_80 |
|:-------------|:--------------------|-------------------:|-----------:|-----------:|
| Central-West | 2018-09-02 00:00:00 |           14908    |    8008.4  |   19532.4  |
| Central-West | 2018-09-09 00:00:00 |           14280.2  |    7380.61 |   18904.6  |
| Central-West | 2018-09-16 00:00:00 |           16676.5  |    9776.96 |   21301    |
| Central-West | 2018-09-23 00:00:00 |           12310.7  |    5411.17 |   16935.2  |
| North        | 2018-09-02 00:00:00 |            4616.28 |    2512.9  |    8444.81 |
| North        | 2018-09-09 00:00:00 |            5318.46 |    3215.08 |    9146.99 |
| North        | 2018-09-16 00:00:00 |            5621.38 |    3517.99 |    9449.91 |
| North        | 2018-09-23 00:00:00 |            3984.3  |    1880.92 |    7812.83 |
| Northeast    | 2018-09-02 00:00:00 |           25858.9  |   18112.9  |   35847.7  |
| Northeast    | 2018-09-09 00:00:00 |           34560.8  |   26814.8  |   44549.6  |
| Northeast    | 2018-09-16 00:00:00 |           34958.8  |   27212.9  |   44947.6  |
| Northeast    | 2018-09-23 00:00:00 |           21904.8  |   14158.9  |   31893.6  |
| South        | 2018-09-02 00:00:00 |           27466.7  |    8804.29 |   41697.9  |
| South        | 2018-09-09 00:00:00 |           24408.2  |    5745.82 |   38639.4  |
| South        | 2018-09-16 00:00:00 |           34201.6  |   15539.2  |   48432.8  |
| South        | 2018-09-23 00:00:00 |           31406.6  |   12744.1  |   45637.7  |
| Southeast    | 2018-09-02 00:00:00 |          143387    |   89486.2  |  194956    |
| Southeast    | 2018-09-09 00:00:00 |          184949    |  131048    |  236518    |
| Southeast    | 2018-09-16 00:00:00 |          203007    |  149107    |  254576    |
| Southeast    | 2018-09-23 00:00:00 |          194897    |  140996    |  246466    |

## Interpretation boundary

The values in this document are generated exclusively from the project's forecasting outputs. No external regional assumptions are added.

## Source artifacts

- `reports\secondary\secondary_model_summary.csv`
- `reports\secondary\secondary_latest_forecasts.csv`
