# Changelog
## [2.2.0] - 2026/01/xx
-Stats tab
-Location estimates metrics
-Histogram & boxplots for features
-Shapiro Wilk test for normality of distributions
-Outliers detection IQR & z-score modified

## [2.1.1] -2026/01/28
-Tag HRV metric on home tab as Excellent, Good or Bad
-Time exercised metric on home tab with a goal of 4 hours
-Recovery plot with nap on home tab
-Home tab recovery charts with insights
-Error handling with naps metrics
>>>>>>> main

## [2.1.0] - 2026/01/20
-Naps added to sleep tab
-Home tab with nap metrics (average, nap days & nap frequency)
-Naps classified based on the duration & the hour it was taken
-Sigmoid recovery function modified by naps
-Home tab with naps summary (sigmoid modified, delta & status)

## [2.0.4] -2026/01/18
-Fixed data governance for better accuracy
-Visuals improved

## [2.0.3] -2026/01/17
- Sliders with moving average up to 11 days added for plot recovery and sleep charts in home tab
-Dark mode always ON
-Standardize y-axis scaling for consistent interpretation


## [2.0.2] - 2026/01/16
- Data freshness badge (workouts / sleep / recovery)
- Home tab data status indicator (slightly delayed / fresh)

## [2.0.1]
- Data integrity added as a visible metric in Home tab

## [2.0.0]
- Fixed major bugs
- Collapsed file upload panel
- Added data integrity checks
- Added pd.merge logic in import_sleep_data.py
- Refactored app architecture

## [1.5.0]
- Added 5 tabs: Home, Workouts, Sleep, Recovery, Correlations
- Introduced sigmoid recovery score (experimental)

## [1.0.0]
- Initial app creatioN

