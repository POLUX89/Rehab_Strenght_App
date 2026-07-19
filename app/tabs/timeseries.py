"""Tab Time Series Analysis — extraída de streamlit_app.py sin cambiar la lógica.

render devuelve time_series_analysis ("Stationary"/"Non-Stationary"/"Inconclusive"),
que la tab Models (tab6) reutiliza. Los imports de statsmodels, que en el monolito
estaban dentro del bloque with, se subieron al top del módulo.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss


def render(recovery):
    """Render the Time Series Analysis tab (ADF/KPSS, ACF/PACF).

    Args:
        recovery: Recovery DataFrame.

    Returns:
        The overall stationarity verdict — ``"Stationary"``,
        ``"Non-Stationary"`` or ``"Inconclusive"`` — which the Models tab
        reuses.
    """
    st.header("🔗 Time Series Analysis")

    tsa_col = [
        "Date",
        "Start",
        "End",
        "InBed hrs",
        "Asleep hrs",
        "Awake",
        "REM hrs",
        "Light hrs",
        "Deep hrs",
        "Efficiency",
        "Fall Asleep",
        "Score",
    ]
    tsa_df = (
        recovery[tsa_col].dropna(subset=["Score"]).copy().sort_values(by="Date", ascending=True)
        if recovery is not None
        else None
    )
    time_series_analysis = ""
    tsa_series = (
        tsa_df.set_index("Date")["Score"]
        .asfreq("D")  # insert NaN for missing nights
        .interpolate("time")  # fill gaps proportionally to time distance
    )

    with st.expander("📋 Time Series Data", expanded=False):
        st.write("Length of time series data:", tsa_df.shape[0] if tsa_df is not None else "N/A")
        st.dataframe(tsa_df.head(10))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            tsa_df["Date"], tsa_df["Score"], marker="x", markersize=2, color="salmon", linewidth=0.5
        )
        ax.set_title("Sleep Score over time", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=45, labelsize=6)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
        st.pyplot(fig)

    with st.expander("📊 ACF & PACF of Sleep Score", expanded=True):
        max_lags = min(100, len(tsa_series) // 2 - 1)
        LAGS = st.slider("Select number of lags for ACF/PACF", 10, max_lags, min(30, max_lags), 5)
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_acf(tsa_series, lags=LAGS, title="ACF of Sleep Score", ax=ax)
        st.pyplot(fig)
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_pacf(tsa_series, lags=LAGS, title="PACF of Sleep Score", ax=ax)
        st.pyplot(fig)

    with st.expander("📈 Stationary Tests of Sleep Score", expanded=True):
        # ADF Test — H₀: series has a unit root (NON-stationary)
        st.info("ADF Test — H₀: series has a unit root")
        adf_result = adfuller(tsa_series)
        st.write(f"ADF statistic: {adf_result[0]:.4f}")
        st.write(f"ADF p-value:   {adf_result[1]:.4f}")
        # p < 0.05 → reject H₀ → stationary ✅
        pvalue_adf = adf_result[1]
        # KPSS Test — H₀: series IS stationary (OPPOSITE null!)
        st.info("KPSS Test — H₀: series IS stationary")
        kpss_result = kpss(tsa_series, regression="ct", nlags="auto")
        st.write(f"KPSS statistic: {kpss_result[0]:.4f}")
        st.write(f"KPSS p-value:   {kpss_result[1]:.4f}")
        # p > 0.05 → fail to reject H₀ → stationary ✅
        pvalue_kpss = kpss_result[1]

        if pvalue_adf < 0.05 and pvalue_kpss > 0.05:
            st.success("Both tests indicate the series is likely stationary.")
            time_series_analysis = "Stationary"
        elif pvalue_adf >= 0.05 and pvalue_kpss <= 0.05:
            st.warning("Both tests indicate the series is likely non-stationary.")
            time_series_analysis = "Non-Stationary"
        else:
            st.info(
                "Tests are inconclusive or conflicting. Consider differencing or further analysis."
            )
            time_series_analysis = "Inconclusive"
        st.write(f"Overall Time Series Analysis: **{time_series_analysis}**")

    return time_series_analysis
