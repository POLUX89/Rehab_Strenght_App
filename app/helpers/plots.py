"""Helpers de visualización (matplotlib + Streamlit).

Extraídos de streamlit_app.py sin cambiar la lógica. A diferencia de stats/
transforms, estas funciones renderizan directamente en Streamlit (st.pyplot,
st.success/info/warning), así que dependen de streamlit.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def plot_line(
    dfx,
    x,
    y,
    title,
    ylabel,
    xlabel="Date",
    marker="o",
    markersize=4,
    color=None,
    show_grid=True,
    despine=True,
    rotate_x=False,
    date_locator=None,
    date_formatter=None,
    linewidth=1.5,
):
    """Render a single-series line chart into the Streamlit page.

    Args:
        dfx: DataFrame holding the data to plot.
        x: Column name for the x-axis.
        y: Column name for the y-axis.
        title: Chart title.
        ylabel: Label for the y-axis.
        xlabel: Label for the x-axis. Defaults to ``"Date"``.
        marker: Matplotlib marker style. Defaults to ``"o"``.
        markersize: Marker size in points. Defaults to 4.
        color: Line color, or None for the matplotlib default.
        show_grid: Whether to draw a light horizontal grid. Defaults to True.
        despine: Whether to remove the top/right spines. Defaults to True.
        rotate_x: Whether to rotate x tick labels 45°. Defaults to False.
        date_locator: Optional matplotlib major locator for the x-axis.
        date_formatter: Optional matplotlib major formatter for the x-axis.
        linewidth: Line width in points. Defaults to 1.5.

    Returns:
        None. The figure is drawn with ``st.pyplot``.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dfx[x], dfx[y], marker=marker, markersize=markersize, color=color, linewidth=linewidth)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if show_grid:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    if rotate_x:
        ax.tick_params(axis="x", rotation=45)

    if date_locator:
        ax.xaxis.set_major_locator(date_locator)
    if date_formatter:
        ax.xaxis.set_major_formatter(date_formatter)
    if despine:
        sns.despine(ax=ax)

    st.pyplot(fig)


def plot_two_axis(dfx, x, y1, y2, title, y1_label, y2_label):
    """Render a dual-y-axis line chart into the Streamlit page.

    The first series is drawn on the left axis (solid) and the second on a
    twin right axis (dashed), sharing the same x-axis.

    Args:
        dfx: DataFrame holding the data to plot.
        x: Column name for the shared x-axis.
        y1: Column name for the left (primary) axis.
        y2: Column name for the right (secondary) axis.
        title: Chart title.
        y1_label: Label for the left axis.
        y2_label: Label for the right axis.

    Returns:
        None. The figure is drawn with ``st.pyplot``.
    """
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(dfx[x], dfx[y1], marker="o")
    ax1.set_xlabel("Date")
    ax1.set_ylabel(y1_label)
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(dfx[x], dfx[y2], marker="o", linestyle="--")
    ax2.set_ylabel(y2_label)

    ax1.set_title(title)
    st.pyplot(fig)


def correlation_insight(df, col1, col2):
    """Render a Streamlit callout describing the correlation of two columns.

    Computes the Pearson correlation over rows where both columns are
    present and emits a success/info/warning box whose wording reflects the
    strength and sign of the coefficient.

    Args:
        df: Source DataFrame, or None.
        col1: First column name.
        col2: Second column name.

    Returns:
        The Streamlit element returned by ``st.success``/``st.info``/
        ``st.warning``, or a plain message string when the columns are
        missing or the data is insufficient.
    """
    if df is None or col1 not in df.columns or col2 not in df.columns:
        return "Insufficient data for correlation analysis."
    corr_coef = df[[col1, col2]].dropna().corr().iloc[0, 1]
    if corr_coef == 1:
        return st.success(f"Perfect positive correlation (1.00) between {col1} and {col2}.")
    elif corr_coef > 0.7:
        return st.success(
            f"Strong positive correlation ({corr_coef:.2f}) between {col1} and {col2}."
        )
    elif corr_coef > 0.49:
        return st.info(
            f"Moderate positive correlation ({corr_coef:.2f}) between {col1} and {col2}."
        )
    elif corr_coef > 0:
        return st.warning(
            f"Weak or no significant correlation ({corr_coef:.2f}) between {col1} and {col2}."
        )
    elif corr_coef == -1:
        return st.success(f"Perfect negative correlation (1.00) between {col1} and {col2}.")
    elif corr_coef < -0.7:
        return st.success(
            f"Strong negative correlation ({corr_coef:.2f}) between {col1} and {col2}."
        )
    elif corr_coef < -0.49:
        return st.info(
            f"Moderate negative correlation ({corr_coef:.2f}) between {col1} and {col2}."
        )
    else:
        return st.warning(
            f"Weak or no significant correlation ({corr_coef:.2f}) between {col1} and {col2}."
        )
