# Monitoring Context

This document contains project-derived model and data monitoring results.

These results are monitoring signals, not external industry assessments.

## Current monitoring summary

- Overall monitoring status: **alert**
- Drift alerts: **103**
- Drift warnings: **2**
- Performance alerts: **12**
- Performance warnings: **8**

## Model performance monitoring

| series_type   | series_id                   | baseline_start   | baseline_end   | recent_start   | recent_end   |   baseline_mae |   baseline_rmse |   baseline_mape |   recent_mae |   recent_rmse |   recent_mape | status   |
|:--------------|:----------------------------|:-----------------|:---------------|:---------------|:-------------|---------------:|----------------:|----------------:|-------------:|--------------:|--------------:|:---------|
| overall       | overall                     | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |       27587.4  |       35802.3   |         12.0872 |    51627.9   |     66329     |       25.8074 | alert    |
| category      | Automotive                  | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        2911.64 |        3794.06  |         34.2447 |     3254.84  |      3850.8   |       35.8766 | alert    |
| category      | Beauty & Health             | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        7035.58 |        7993.55  |         26.2532 |     6413.58  |      7915.42  |       20.7944 | stable   |
| category      | Books & Media               | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        2664.64 |        3242.63  |         51.8719 |     2705.65  |      3384.53  |       73.0612 | alert    |
| category      | Electronics & Computing     | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        9808.84 |       12247.5   |         36.3964 |     5788.31  |      6851.61  |       31.1225 | warning  |
| category      | Fashion & Accessories       | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |       10120.2  |       11936.3   |         36.6664 |    11169.7   |     14011.8   |       51.0052 | alert    |
| category      | Food & Beverage             | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |         579.91 |         821.954 |         33.6203 |      761.669 |       929.124 |      127.68   | alert    |
| category      | Gifts, Arts & Seasonal      | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        1913.59 |        2309.45  |         19.0047 |     3161.97  |      4073.82  |       64.777  | alert    |
| category      | Home & Furniture            | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        9651.23 |       12405.2   |         21.0088 |    11500.5   |     13819.4   |       25.2797 | warning  |
| category      | Home Improvement & Garden   | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        4226.95 |        5588.08  |         23.8368 |     3826.45  |      4707.33  |       25.618  | warning  |
| category      | Kids & Baby                 | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        6519.88 |        7967.48  |         43.8165 |     4362.88  |      5186.28  |       40.6498 | alert    |
| category      | Kitchen & Appliances        | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        4835.7  |        6395.62  |         42.7092 |     3413.93  |      4055.26  |       34.9691 | warning  |
| category      | Office, Business & Services | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        4658.88 |        5821.28  |         31.5428 |     3216.18  |      3712.93  |       30.1896 | warning  |
| category      | Pet Supplies                | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        1329.3  |        1693.76  |         35.5864 |     1425.81  |      1878.64  |       27.0927 | warning  |
| category      | Phones & Telecom            | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        2043.07 |        2722.78  |         34.563  |     2978.72  |      4364.21  |       37.5799 | alert    |
| category      | Sports & Leisure            | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        5508.56 |        7805.84  |         27.5407 |     4262.66  |      5351.05  |       37.1644 | alert    |
| region        | Central-West                | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        3275.8  |        4808.29  |         19.44   |     5275.16  |      6910.26  |       45.8451 | alert    |
| region        | North                       | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        1508.04 |        1843.95  |         24.415  |     2805.68  |      3750.84  |       92.5111 | alert    |
| region        | Northeast                   | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        5292.63 |        6573.27  |         18.5865 |     7177.01  |      8632.79  |       26.7915 | warning  |
| region        | South                       | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |        9796.87 |       11811.9   |         28.2482 |    12186.6   |     15106.3   |       45.9558 | alert    |
| region        | Southeast                   | 2017-12-31       | 2018-05-13     | 2018-05-20     | 2018-08-05   |       34243.1  |       47220.8   |         21.3661 |    42261.7   |     49182     |       29.0274 | warning  |

## Data drift monitoring

| series_type   | series_id                   | feature        |   ks_statistic |   ks_p_value |       psi | status   |
|:--------------|:----------------------------|:---------------|---------------:|-------------:|----------:|:---------|
| overall       | overall                     | target_revenue |       0.961538 |  2.09712e-13 | 12.0966   | alert    |
| overall       | overall                     | lag_1          |       1        |  4.03292e-15 | 12.0966   | alert    |
| overall       | overall                     | lag_4          |       1        |  4.03292e-15 | 12.0966   | alert    |
| overall       | overall                     | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| overall       | overall                     | rolling_std_4  |       0.615385 |  6.38009e-05 |  4.83795  | alert    |
| category      | Automotive                  | target_revenue |       0.884615 |  8.91275e-11 |  9.76662  | alert    |
| category      | Automotive                  | lag_1          |       0.923077 |  5.34765e-12 | 10.7199   | alert    |
| category      | Automotive                  | lag_4          |       0.923077 |  5.34765e-12 | 11.1416   | alert    |
| category      | Automotive                  | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| category      | Automotive                  | rolling_std_4  |       0.307692 |  0.172029    |  6.00052  | alert    |
| category      | Beauty & Health             | target_revenue |       0.961538 |  2.09712e-13 | 11.1416   | alert    |
| category      | Beauty & Health             | lag_1          |       0.961538 |  2.09712e-13 | 11.1416   | alert    |
| category      | Beauty & Health             | lag_4          |       0.961538 |  2.09712e-13 | 10.7199   | alert    |
| category      | Beauty & Health             | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| category      | Beauty & Health             | rolling_std_4  |       0.807692 |  1.04814e-08 |  9.76662  | alert    |
| category      | Books & Media               | target_revenue |       0.692308 |  3.03493e-06 |  6.18948  | alert    |
| category      | Books & Media               | lag_1          |       0.730769 |  5.39542e-07 |  6.66886  | alert    |
| category      | Books & Media               | lag_4          |       0.692308 |  3.03493e-06 |  7.13276  | alert    |
| category      | Books & Media               | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| category      | Books & Media               | rolling_std_4  |       0.653846 |  1.48374e-05 |  7.01264  | alert    |
| category      | Electronics & Computing     | target_revenue |       0.807692 |  1.04814e-08 |  8.3935   | alert    |
| category      | Electronics & Computing     | lag_1          |       0.846154 |  1.09181e-09 |  9.76662  | alert    |
| category      | Electronics & Computing     | lag_4          |       0.846154 |  1.09181e-09 |  9.76662  | alert    |
| category      | Electronics & Computing     | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| category      | Electronics & Computing     | rolling_std_4  |       0.576923 |  0.000243603 |  5.28401  | alert    |
| category      | Fashion & Accessories       | target_revenue |       0.884615 |  8.91275e-11 |  9.76662  | alert    |
| category      | Fashion & Accessories       | lag_1          |       0.923077 |  5.34765e-12 | 10.7199   | alert    |
| category      | Fashion & Accessories       | lag_4          |       0.846154 |  1.09181e-09 | 10.5365   | alert    |
| category      | Fashion & Accessories       | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| category      | Fashion & Accessories       | rolling_std_4  |       0.538462 |  0.000832312 |  3.85424  | alert    |
| category      | Food & Beverage             | target_revenue |       0.846154 |  1.09181e-09 |  9.76662  | alert    |
| category      | Food & Beverage             | lag_1          |       0.846154 |  1.09181e-09 |  9.76662  | alert    |
| category      | Food & Beverage             | lag_4          |       0.846154 |  1.09181e-09 |  9.76662  | alert    |
| category      | Food & Beverage             | rolling_mean_4 |       0.961538 |  2.09712e-13 | 12.0966   | alert    |
| category      | Food & Beverage             | rolling_std_4  |       0.769231 |  8.21043e-08 |  8.63242  | alert    |
| category      | Gifts, Arts & Seasonal      | target_revenue |       0.192308 |  0.732737    |  0.234059 | warning  |
| category      | Gifts, Arts & Seasonal      | lag_1          |       0.153846 |  0.926019    |  0.223906 | warning  |
| category      | Gifts, Arts & Seasonal      | lag_4          |       0.307692 |  0.172029    |  2.65346  | alert    |
| category      | Gifts, Arts & Seasonal      | rolling_mean_4 |       0.230769 |  0.500995    |  4.39352  | alert    |
| category      | Gifts, Arts & Seasonal      | rolling_std_4  |       0.307692 |  0.172029    |  4.15988  | alert    |
| category      | Home & Furniture            | target_revenue |       0.884615 |  8.91275e-11 |  9.76662  | alert    |
| category      | Home & Furniture            | lag_1          |       0.923077 |  5.34765e-12 | 11.1416   | alert    |
| category      | Home & Furniture            | lag_4          |       0.961538 |  2.09712e-13 | 10.7199   | alert    |
| category      | Home & Furniture            | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| category      | Home & Furniture            | rolling_std_4  |       0.576923 |  0.000243603 |  2.90726  | alert    |
| category      | Home Improvement & Garden   | target_revenue |       0.961538 |  2.09712e-13 | 10.7199   | alert    |
| category      | Home Improvement & Garden   | lag_1          |       0.961538 |  2.09712e-13 | 10.7199   | alert    |
| category      | Home Improvement & Garden   | lag_4          |       0.961538 |  2.09712e-13 | 12.0966   | alert    |
| category      | Home Improvement & Garden   | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| category      | Home Improvement & Garden   | rolling_std_4  |       0.807692 |  1.04814e-08 |  7.26617  | alert    |
| category      | Kids & Baby                 | target_revenue |       0.769231 |  8.21043e-08 |  7.26617  | alert    |
| category      | Kids & Baby                 | lag_1          |       0.769231 |  8.21043e-08 |  8.54217  | alert    |
| category      | Kids & Baby                 | lag_4          |       0.692308 |  3.03493e-06 |  7.15047  | alert    |
| category      | Kids & Baby                 | rolling_mean_4 |       0.846154 |  1.09181e-09 |  9.45857  | alert    |
| category      | Kids & Baby                 | rolling_std_4  |       0.846154 |  1.09181e-09 | 10.1617   | alert    |
| category      | Kitchen & Appliances        | target_revenue |       0.884615 |  8.91275e-11 | 11.1416   | alert    |
| category      | Kitchen & Appliances        | lag_1          |       0.923077 |  5.34765e-12 | 12.0966   | alert    |
| category      | Kitchen & Appliances        | lag_4          |       0.884615 |  8.91275e-11 | 11.1416   | alert    |
| category      | Kitchen & Appliances        | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| category      | Kitchen & Appliances        | rolling_std_4  |       0.5      |  0.00256096  |  5.73136  | alert    |
| category      | Office, Business & Services | target_revenue |       0.884615 |  8.91275e-11 | 10.1883   | alert    |
| category      | Office, Business & Services | lag_1          |       0.923077 |  5.34765e-12 | 11.1416   | alert    |
| category      | Office, Business & Services | lag_4          |       0.923077 |  5.34765e-12 | 11.1416   | alert    |
| category      | Office, Business & Services | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| category      | Office, Business & Services | rolling_std_4  |       0.461538 |  0.0071341   |  4.94128  | alert    |
| category      | Pet Supplies                | target_revenue |       0.730769 |  5.39542e-07 |  7.14383  | alert    |
| category      | Pet Supplies                | lag_1          |       0.730769 |  5.39542e-07 |  8.08933  | alert    |
| category      | Pet Supplies                | lag_4          |       0.730769 |  5.39542e-07 |  8.19752  | alert    |
| category      | Pet Supplies                | rolling_mean_4 |       0.961538 |  2.09712e-13 | 12.0966   | alert    |
| category      | Pet Supplies                | rolling_std_4  |       0.346154 |  0.0885018   |  1.4827   | alert    |
| category      | Phones & Telecom            | target_revenue |       0.807692 |  1.04814e-08 |  8.3935   | alert    |
| category      | Phones & Telecom            | lag_1          |       0.846154 |  1.09181e-09 |  9.34498  | alert    |
| category      | Phones & Telecom            | lag_4          |       0.846154 |  1.09181e-09 | 11.1416   | alert    |
| category      | Phones & Telecom            | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| category      | Phones & Telecom            | rolling_std_4  |       0.576923 |  0.000243603 |  4.53459  | alert    |
| category      | Sports & Leisure            | target_revenue |       0.807692 |  1.04814e-08 |  9.45857  | alert    |
| category      | Sports & Leisure            | lag_1          |       0.846154 |  1.09181e-09 |  9.16335  | alert    |
| category      | Sports & Leisure            | lag_4          |       0.846154 |  1.09181e-09 |  9.62725  | alert    |
| category      | Sports & Leisure            | rolling_mean_4 |       0.923077 |  5.34765e-12 | 12.0966   | alert    |
| category      | Sports & Leisure            | rolling_std_4  |       0.5      |  0.00256096  |  5.4504   | alert    |
| region        | Central-West                | target_revenue |       0.884615 |  8.91275e-11 |  8.3935   | alert    |
| region        | Central-West                | lag_1          |       0.923077 |  5.34765e-12 | 10.7199   | alert    |
| region        | Central-West                | lag_4          |       0.884615 |  8.91275e-11 |  9.34498  | alert    |
| region        | Central-West                | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| region        | Central-West                | rolling_std_4  |       0.461538 |  0.0071341   |  4.33703  | alert    |
| region        | North                       | target_revenue |       0.615385 |  6.38009e-05 |  6.54873  | alert    |
| region        | North                       | lag_1          |       0.692308 |  3.03493e-06 |  7.63892  | alert    |
| region        | North                       | lag_4          |       0.692308 |  3.03493e-06 |  8.5154   | alert    |
| region        | North                       | rolling_mean_4 |       0.961538 |  2.09712e-13 | 12.0966   | alert    |
| region        | North                       | rolling_std_4  |       0.538462 |  0.000832312 |  5.52458  | alert    |
| region        | Northeast                   | target_revenue |       0.884615 |  8.91275e-11 |  9.76662  | alert    |
| region        | Northeast                   | lag_1          |       0.923077 |  5.34765e-12 | 11.1416   | alert    |
| region        | Northeast                   | lag_4          |       0.961538 |  2.09712e-13 | 10.7199   | alert    |
| region        | Northeast                   | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| region        | Northeast                   | rolling_std_4  |       0.653846 |  1.48374e-05 |  5.77889  | alert    |
| region        | South                       | target_revenue |       0.807692 |  1.04814e-08 |  8.63541  | alert    |
| region        | South                       | lag_1          |       0.846154 |  1.09181e-09 |  9.16335  | alert    |
| region        | South                       | lag_4          |       0.884615 |  8.91275e-11 | 10.7199   | alert    |
| region        | South                       | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| region        | South                       | rolling_std_4  |       0.538462 |  0.000832312 |  5.28571  | alert    |
| region        | Southeast                   | target_revenue |       1        |  4.03292e-15 | 12.0966   | alert    |
| region        | Southeast                   | lag_1          |       1        |  4.03292e-15 | 12.0966   | alert    |
| region        | Southeast                   | lag_4          |       1        |  4.03292e-15 | 12.0966   | alert    |
| region        | Southeast                   | rolling_mean_4 |       1        |  4.03292e-15 | 12.0966   | alert    |
| region        | Southeast                   | rolling_std_4  |       0.653846 |  1.48374e-05 |  6.04092  | alert    |

## Source artifacts

- `reports\monitoring\monitoring_summary.json`
- `reports\monitoring\model_performance_report.csv`
- `reports\monitoring\data_drift_report.csv`
