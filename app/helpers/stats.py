"""Helpers estadísticos puros (sin Streamlit).

Extraídos de streamlit_app.py sin cambiar la lógica.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def normality_test(series):
    """Run the Shapiro-Wilk normality test on a series.

    Args:
        series: Numeric series to test; missing values are dropped first.

    Returns:
        A ``(p_value, interpretation)`` tuple, where ``interpretation`` is a
        human-readable verdict using the 0.05 significance threshold.
    """
    series = series.dropna()
    stat, p_value = stats.shapiro(series)
    if p_value > 0.05:
        interpretation = "Data is normally distributed (fail to reject H0)"
    else:
        interpretation = "Data is not normally distributed (reject H0)"
    return p_value, interpretation


def outlier_dectection_iqr(series):
    """Detect outliers with the interquartile-range (Tukey) rule.

    Flags values below ``Q1 - 1.5*IQR`` or above ``Q3 + 1.5*IQR``.

    Args:
        series: Numeric series; missing values are dropped first.

    Returns:
        A series containing only the outlier values.
    """
    series = series.dropna()
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return outliers


def outlier_detection_zscore_modified(series, threshold=3):
    """Detect outliers with the median-absolute-deviation modified z-score.

    Uses ``0.6745 * (x - median) / MAD`` and flags values whose absolute
    score exceeds ``threshold``. More robust to outliers than the classic
    mean/standard-deviation z-score.

    Args:
        series: Numeric series; missing values are dropped first.
        threshold: Absolute modified-z cutoff. Defaults to 3.

    Returns:
        A series containing only the outlier values.
    """
    series = series.dropna()
    mad = np.abs(series - series.median()).median()
    modified_z_scores = 0.6745 * (series - series.median()) / mad
    outliers = series[np.abs(modified_z_scores) > threshold]
    return outliers


def fit_distribution(data):
    """Fit several candidate distributions and rank them by AIC.

    Fits normal, Student-t, Cauchy, Laplace, skew-normal, lognormal, gamma,
    exponential and Weibull distributions via maximum likelihood, computing
    AIC and BIC for each. Distributions that fail to fit are skipped.

    Args:
        data: Array-like sample of values.

    Returns:
        A DataFrame with columns ``distribution``, ``params``, ``AIC`` and
        ``BIC``, sorted ascending by ``AIC`` (best fit first).
    """
    data = np.array(data)
    distribution = {
        "normal": stats.norm,
        "student-t": stats.t,
        "cauchy": stats.cauchy,
        "laplace": stats.laplace,
        "skewnorm": stats.skewnorm,
        # Recommended additions for positive skewed data
        "lognormal": stats.lognorm,  # Very common for flight hours
        "gamma": stats.gamma,  # Positive values, flexible shape
        "exponential": stats.expon,  # Simple right-skewed
        "weibull": stats.weibull_min,  # Reliability/lifetime data
    }
    results = []

    for name, dist in distribution.items():
        try:
            params = dist.fit(data)
            ll = np.sum(dist.logpdf(data, *params))
            k = len(params)
            aic = 2 * k - 2 * ll
            bic = np.log(len(data)) * k - 2 * ll

            results.append(
                {
                    "distribution": name,
                    "params": params,
                    "AIC": aic,
                    "BIC": bic,
                }
            )
        except Exception:
            # Handle exceptions for distributions that fail to fit
            pass
    return pd.DataFrame(results).sort_values("AIC")


def compute_ecdf(data, x, complementary=True):
    """Evaluate the empirical CDF (or its complement) at a point.

    Args:
        data: Array-like sample of values.
        x: Threshold at which to evaluate.
        complementary: If True, return ``P(data > x)`` (the survival
            function); if False, return ``P(data <= x)``. Defaults to True.

    Returns:
        The empirical probability, rounded to 4 decimals.
    """
    counter = np.sum(data > x) if complementary else np.sum(data <= x)
    return np.round(counter / len(data), 4)


def metrics_learning_curve(df, sample_size, predictors, H=30, min_train=10):
    """Fit an OLS model on a data prefix and score train/test at one checkpoint.

    Takes the first ``sample_size`` rows, holds out the last ``H`` as a test
    window, fits an HC3-robust OLS on the rest, and reports fit/error metrics
    for one point of a learning curve.

    Args:
        df: Ordered DataFrame containing the predictors and the ``Score``
            target.
        sample_size: Number of leading rows to use for this checkpoint.
        predictors: Predictor column names.
        H: Size of the held-out test window. Defaults to 30.
        min_train: Minimum training rows required. Defaults to 10.

    Returns:
        A dict of train/test sizes and MAE/RMSE/MSE/R² metrics, or None if
        the prefix has fewer than ``H + min_train`` rows.
    """
    df_sample = df.iloc[:sample_size].copy()

    # Require enough rows to form a stable train/test
    if len(df_sample) < (H + min_train):
        return None  # skip this checkpoint

    train = df_sample.iloc[:-H].copy()
    test = df_sample.iloc[-H:].copy()

    X = sm.add_constant(train[predictors], has_constant="add")
    y = train["Score"]
    model = sm.OLS(y, X).fit(cov_type="HC3")

    X_test = sm.add_constant(test[predictors], has_constant="add")
    y_test = test["Score"]
    y_pred = model.predict(X_test)

    r2_train = model.rsquared
    r2_test = r2_score(y_test, y_pred)
    mse_train = mean_squared_error(y, model.fittedvalues)
    mse_test = mean_squared_error(y_test, y_pred)
    mae_train = mean_absolute_error(y, model.fittedvalues)
    mae_test = mean_absolute_error(y_test, y_pred)
    rmse_train = np.sqrt(mse_train)
    rmse_test = np.sqrt(mse_test)

    return {
        "Model_samples": sample_size,
        "Train size": len(train),
        "Test size": len(test),
        "Train MAE": mae_train,
        "Test MAE": mae_test,
        "Train RMSE": rmse_train,
        "Test RMSE": rmse_test,
        "Train MSE": mse_train,
        "Test MSE": mse_test,
        "Train R²": r2_train,
        "Test R²": r2_test,
    }
