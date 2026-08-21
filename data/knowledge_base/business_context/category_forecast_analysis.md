# Category Forecast Analysis

This document contains project-derived forecast information for the 15 approved business categories.

All category forecasting is performed using XGBoost with the approved walk-forward validation design.

## Category model performance

| series_id                   |   mean_mape |   mean_rmse |   mean_mae |   mean_interval_coverage_pct |   rank_within_type |
|:----------------------------|------------:|------------:|-----------:|-----------------------------:|-------------------:|
| Home & Furniture            |     22.6104 |   11756.1   |   10344.7  |                       71.875 |                  1 |
| Beauty & Health             |     24.2061 |    7687.87  |    6802.33 |                       65.625 |                  2 |
| Home Improvement & Garden   |     24.5047 |    4973.68  |    4076.76 |                       75     |                  3 |
| Office, Business & Services |     31.0353 |    4592.82  |    4117.87 |                       56.25  |                  4 |
| Sports & Leisure            |     31.1496 |    5853.99  |    5041.34 |                       56.25  |                  5 |
| Pet Supplies                |     32.4013 |    1542.92  |    1365.49 |                       75     |                  6 |
| Electronics & Computing     |     34.4187 |    9336.66  |    8301.14 |                       84.375 |                  7 |
| Automotive                  |     34.8566 |    3519.65  |    3040.34 |                       65.625 |                  8 |
| Phones & Telecom            |     35.6943 |    2966.55  |    2393.94 |                       75     |                  9 |
| Gifts, Arts & Seasonal      |     36.1693 |    2709.97  |    2381.73 |                       81.25  |                 10 |
| Kitchen & Appliances        |     39.8066 |    4976.63  |    4302.54 |                       53.125 |                 11 |
| Fashion & Accessories       |     42.0435 |   11736.1   |   10513.8  |                       59.375 |                 12 |
| Kids & Baby                 |     42.629  |    6406.34  |    5711    |                       68.75  |                 13 |
| Books & Media               |     59.8179 |    3025.77  |    2680.02 |                       53.125 |                 14 |
| Food & Beverage             |     68.8927 |     789.141 |     648.07 |                       62.5   |                 15 |

## Latest category forecasts

| series_id                   | timestamp           |   forecast_revenue |   lower_80 |   upper_80 |
|:----------------------------|:--------------------|-------------------:|-----------:|-----------:|
| Automotive                  | 2018-09-02 00:00:00 |            9803.17 |  7570.01   |   15817.8  |
| Automotive                  | 2018-09-09 00:00:00 |           12793.5  | 10560.4    |   18808.1  |
| Automotive                  | 2018-09-16 00:00:00 |           12027.5  |  9794.3    |   18042.1  |
| Automotive                  | 2018-09-23 00:00:00 |           14574.7  | 12341.5    |   20589.3  |
| Beauty & Health             | 2018-09-02 00:00:00 |           28674.1  | 23057.6    |   40353.6  |
| Beauty & Health             | 2018-09-09 00:00:00 |           32545.8  | 26929.3    |   44225.2  |
| Beauty & Health             | 2018-09-16 00:00:00 |           29002.5  | 23386      |   40681.9  |
| Beauty & Health             | 2018-09-23 00:00:00 |           26605.9  | 20989.4    |   38285.4  |
| Books & Media               | 2018-09-02 00:00:00 |            6288.96 |  3657.86   |   10228.9  |
| Books & Media               | 2018-09-09 00:00:00 |            4016.02 |  1384.91   |    7955.92 |
| Books & Media               | 2018-09-16 00:00:00 |            4705.55 |  2074.44   |    8645.46 |
| Books & Media               | 2018-09-23 00:00:00 |            2838.26 |   207.148  |    6778.16 |
| Electronics & Computing     | 2018-09-02 00:00:00 |           39612.9  | 26580.4    |   50050.3  |
| Electronics & Computing     | 2018-09-09 00:00:00 |           30145.9  | 17113.5    |   40583.4  |
| Electronics & Computing     | 2018-09-16 00:00:00 |           24562.6  | 11530.2    |   35000    |
| Electronics & Computing     | 2018-09-23 00:00:00 |           15476.5  |  2444.1    |   25914    |
| Fashion & Accessories       | 2018-09-02 00:00:00 |           26277.4  | 15636.7    |   40566.6  |
| Fashion & Accessories       | 2018-09-09 00:00:00 |           23355.7  | 12715      |   37644.9  |
| Fashion & Accessories       | 2018-09-16 00:00:00 |           27294.1  | 16653.4    |   41583.3  |
| Fashion & Accessories       | 2018-09-23 00:00:00 |           24063.4  | 13422.7    |   38352.6  |
| Food & Beverage             | 2018-09-02 00:00:00 |            3597.2  |  3121.49   |    4588.71 |
| Food & Beverage             | 2018-09-09 00:00:00 |            3009.73 |  2534.02   |    4001.25 |
| Food & Beverage             | 2018-09-16 00:00:00 |            3536.93 |  3061.22   |    4528.45 |
| Food & Beverage             | 2018-09-23 00:00:00 |            3099.66 |  2623.95   |    4091.18 |
| Gifts, Arts & Seasonal      | 2018-09-02 00:00:00 |            4791.57 |     0      |    8453.92 |
| Gifts, Arts & Seasonal      | 2018-09-09 00:00:00 |            5847.52 |   907.892  |    9509.88 |
| Gifts, Arts & Seasonal      | 2018-09-16 00:00:00 |            4997.97 |    58.3495 |    8660.33 |
| Gifts, Arts & Seasonal      | 2018-09-23 00:00:00 |            4553.79 |     0      |    8216.15 |
| Home & Furniture            | 2018-09-02 00:00:00 |           47548.6  | 32097.9    |   63578.7  |
| Home & Furniture            | 2018-09-09 00:00:00 |           41076    | 25625.3    |   57106.1  |
| Home & Furniture            | 2018-09-16 00:00:00 |           45177.1  | 29726.4    |   61207.2  |
| Home & Furniture            | 2018-09-23 00:00:00 |           58585.4  | 43134.7    |   74615.5  |
| Home Improvement & Garden   | 2018-09-02 00:00:00 |           16099.2  | 12479.8    |   23027.4  |
| Home Improvement & Garden   | 2018-09-09 00:00:00 |           14360.1  | 10740.7    |   21288.3  |
| Home Improvement & Garden   | 2018-09-16 00:00:00 |           13090.7  |  9471.29   |   20018.9  |
| Home Improvement & Garden   | 2018-09-23 00:00:00 |           17039.6  | 13420.2    |   23967.8  |
| Kids & Baby                 | 2018-09-02 00:00:00 |           13344.1  |  7685.46   |   22481.2  |
| Kids & Baby                 | 2018-09-09 00:00:00 |           11288    |  5629.33   |   20425.1  |
| Kids & Baby                 | 2018-09-16 00:00:00 |           13738.3  |  8079.66   |   22875.4  |
| Kids & Baby                 | 2018-09-23 00:00:00 |           13519.4  |  7860.72   |   22656.5  |
| Kitchen & Appliances        | 2018-09-02 00:00:00 |            9421.02 |  5777.06   |   15265    |
| Kitchen & Appliances        | 2018-09-09 00:00:00 |            9956.14 |  6312.18   |   15800.1  |
| Kitchen & Appliances        | 2018-09-16 00:00:00 |            8923.77 |  5279.81   |   14767.8  |
| Kitchen & Appliances        | 2018-09-23 00:00:00 |            5700.36 |  2056.4    |   11544.3  |
| Office, Business & Services | 2018-09-02 00:00:00 |           11705.6  |  7546.3    |   18921.3  |
| Office, Business & Services | 2018-09-09 00:00:00 |           10928.1  |  6768.8    |   18143.8  |
| Office, Business & Services | 2018-09-16 00:00:00 |            9693.84 |  5534.59   |   16909.6  |
| Office, Business & Services | 2018-09-23 00:00:00 |           10002    |  5842.74   |   17217.7  |
| Pet Supplies                | 2018-09-02 00:00:00 |            6458.5  |  5262.38   |    9077.4  |
| Pet Supplies                | 2018-09-09 00:00:00 |            6621.53 |  5425.42   |    9240.43 |
| Pet Supplies                | 2018-09-16 00:00:00 |            6533.27 |  5337.16   |    9152.17 |
| Pet Supplies                | 2018-09-23 00:00:00 |            6663.97 |  5467.86   |    9282.87 |
| Phones & Telecom            | 2018-09-02 00:00:00 |           12185    |  9041.15   |   16713.6  |
| Phones & Telecom            | 2018-09-09 00:00:00 |           13879.8  | 10735.9    |   18408.4  |
| Phones & Telecom            | 2018-09-16 00:00:00 |           12289.6  |  9145.75   |   16818.2  |
| Phones & Telecom            | 2018-09-23 00:00:00 |            4803.74 |  1659.91   |    9332.34 |
| Sports & Leisure            | 2018-09-02 00:00:00 |           12046.4  |  7717.64   |   19664.1  |
| Sports & Leisure            | 2018-09-09 00:00:00 |           13986    |  9657.3    |   21603.8  |
| Sports & Leisure            | 2018-09-16 00:00:00 |           13474.3  |  9145.6    |   21092.1  |
| Sports & Leisure            | 2018-09-23 00:00:00 |           16787.9  | 12459.1    |   24405.6  |

## Interpretation boundary

The forecast values in this document come directly from the project's saved ML outputs. No external market benchmark is assumed.

## Source artifacts

- `reports\secondary\secondary_model_summary.csv`
- `reports\secondary\secondary_latest_forecasts.csv`
