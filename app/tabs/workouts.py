"""Tab Workouts — extraída de streamlit_app.py sin cambiar el comportamiento.

workouts se muta in-place (añade la columna 'Week'); como el DataFrame se pasa
por referencia, esa mutación se refleja en la variable global igual que en el
monolito, así que render no necesita devolver nada.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from app.helpers.plots import plot_line
from app.helpers.transforms import daily_ma, weekly_bucket


def render(workouts, cva_dt, smooth_days):
    st.header("🏋️ Workouts")

    if workouts is None:
        st.info("Upload your cleaned workouts CSV to see charts.")
        return

    req = {"Date", "EXERCISE_NAME"}
    if not req.issubset(workouts.columns):
        st.error(f"Workouts CSV must include at least: {req}")
        return

    cva_ts = pd.to_datetime(cva_dt)

    # pick exercise
    ex_list = sorted(workouts["EXERCISE_NAME"].dropna().unique())
    chosen_ex = st.selectbox("Choose an exercise:", ex_list)

    w = workouts[workouts["EXERCISE_NAME"] == chosen_ex].copy()
    w = w.dropna(subset=["Date"]).sort_values("Date")

    # -------- 1) Pre vs Post (Estimated 1RM mean)
    st.subheader("📊 Pre vs Post (Estimated 1RM)")
    if "est_1RM" in w.columns:
        pre = w[w["Date"] < cva_ts]["est_1RM"].mean()
        post = w[w["Date"] >= cva_ts]["est_1RM"].mean()

        fig, ax = plt.subplots(figsize=(6, 4))
        vals = [pre, post]
        labs = ["Pre-CVA", "Post-CVA"]
        ax.bar(np.arange(2), vals, width=0.6, edgecolor="black")
        ax.set_xticks(np.arange(2))
        ax.set_xticklabels(labs)
        ax.set_ylabel("Estimated 1RM (lb)")
        ax.set_title(chosen_ex, fontsize=14, fontweight="bold", pad=15)
        sns.despine(ax=ax)
        for i, v in enumerate(vals):
            if not np.isnan(v):
                ax.text(i, v + 2, f"{v:.1f}", ha="center", va="bottom")
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        st.pyplot(fig)
    else:
        st.info("No est_1RM found. Make sure WEIGHT_LBS and REPS exist in the workouts file.")

    # -------- 2) Progress over time (daily + MA)
    st.subheader("⏳ Progress over time (Daily + Moving Avg)")
    if "est_1RM" in w.columns:
        # daily mean est_1RM
        daily = w.groupby("Date", as_index=False)["est_1RM"].mean().sort_values("Date")
        daily["MA"] = daily_ma(daily["est_1RM"], smooth_days)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(
            daily["Date"],
            daily["est_1RM"],
            marker="s",
            label="Daily mean est.1RM",
            color="salmon",
            markersize=4,
        )
        ax.plot(
            daily["Date"],
            daily["MA"],
            linestyle="--",
            label=f"{smooth_days}-day MA",
            color="yellow",
        )
        ax.axvline(cva_ts, linestyle=":", linewidth=1)
        ax.set_title(
            f"{chosen_ex} — Comparative Pre & Post CVA", fontsize=14, fontweight="bold", pad=15
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("lb")
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
        sns.despine(ax=ax)
        ax.legend()
        st.pyplot(fig)

    # -------- 3) Weekly Volume chart (per exercise + total)
    st.subheader("📦 Weekly Volume (Exercise and Total)")
    if "VOLUME" in workouts.columns:
        workouts["Week"] = weekly_bucket(workouts["Date"])
        w_ex_week = (
            workouts[workouts["EXERCISE_NAME"] == chosen_ex]
            .groupby("Week", as_index=False)["VOLUME"]
            .sum()
        )
        w_all_week = workouts.groupby("Week", as_index=False)["VOLUME"].sum()

        cA, cB = st.columns(2)
        with cA:
            plot_line(
                w_ex_week.sort_values("Week"),
                "Week",
                "VOLUME",
                f"Weekly Volume — {chosen_ex}",
                "Total Volume (lb·reps)",
                xlabel="Week",
            )
        with cB:
            plot_line(
                w_all_week.sort_values("Week"),
                "Week",
                "VOLUME",
                "Weekly Volume — ALL Exercises",
                "Total Volume (lb·reps)",
                xlabel="Week",
            )

    # -------- 4) RPE trend (daily mean)
    st.subheader("🔥 RPE Trend (Daily)")
    if "RPE" in w.columns:
        rpe_daily = w.groupby("Date", as_index=False)["RPE"].mean().sort_values("Date")
        if rpe_daily["RPE"].notna().sum() == 0:
            st.info("No RPE values recorded for this exercise yet.")
        else:
            rpe_daily["MA"] = daily_ma(rpe_daily["RPE"], smooth_days)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(
                rpe_daily["Date"],
                rpe_daily["RPE"],
                marker="o",
                label="Daily mean RPE",
                color="salmon",
                markersize=4,
            )
            ax.plot(
                rpe_daily["Date"],
                rpe_daily["MA"],
                linestyle="--",
                label=f"{smooth_days}-day MA",
                color="yellow",
            )
            ax.axvline(cva_ts, linestyle=":", linewidth=1)
            ax.set_title(
                f"{chosen_ex} — RPE Trend & Post CVA", fontsize=14, fontweight="bold", pad=15
            )
            ax.set_xlabel("Date")
            ax.set_ylabel("RPE")
            ax.grid(axis="y", alpha=0.25)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", rotation=45)
            sns.despine(ax=ax)
            ax.legend()
            st.pyplot(fig)

    # -------- Summary table (exercise)
    with st.expander("📋 Show raw sets for this exercise"):
        st.dataframe(w.sort_values("DATE" if "DATE" in w.columns else "Date"))
