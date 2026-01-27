# app.py
# Rehab Strength Dashboard (Workouts + Sleep + Recovery)
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
import io
from datetime import date

st.set_page_config(page_title="Rehab Strength APP", layout="wide")
st.title("🏋️‍♂️ Rehab Strength APP", text_alignment="center")
st.caption("Workouts (Strong) • Sleep (Sheets) • Recovery (Sigmoid)")
app_version = "V2.1.1"
st.caption(f"App Version: {app_version} • Updated: {datetime.now():%Y-%m-%d %H:%M}")
st.markdown("---")

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("⚙️ Settings")
cva_dt = st.sidebar.date_input("CVA split date", value=datetime(2025, 5, 14))
smooth_days = st.sidebar.slider("Smoothing window (days)", 3, 30, 15, 1)
show_dark = st.sidebar.toggle("🌙 Dark mode", value=True)

if show_dark:
    st.markdown(
        """
        <style>
        html, body, [data-testid="stAppViewContainer"] { background-color:#0e1117 !important; color:#e5e7eb !important; }
        [data-testid="stHeader"] { background: rgba(0,0,0,0); }
        </style>
        """,
        unsafe_allow_html=True
    )
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#0e1117",
        "savefig.facecolor": "#0e1117",
        "text.color": "#e5e7eb",
        "axes.labelcolor": "#e5e7eb",
        "xtick.color": "#e5e7eb",
        "ytick.color": "#e5e7eb",
        "axes.edgecolor": "#e5e7eb",
        "grid.color": "#2d3748"
    })
else:
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

# -------------------------
# Helpers
# -------------------------

def make_unique_columns(cols):
    """Fix duplicate column names by suffixing .1 .2 ... (Streamlit upload sometimes preserves dupes)."""
    seen = {}
    out = []
    for c in cols:
        c = str(c).strip()
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
    return out
def pick_col(df, candidates):
    """Return the first column in candidates that exists in df, else None."""
    if df is None:
        return None
    for c in candidates:
        if c in df.columns:
            return c
    return None

def coerce_date(df, col="Date"):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.floor("D")
    return df

def epley_1rm(w, r):
    try:
        w = float(w); r = float(r)
        if np.isnan(w) or np.isnan(r):
            return np.nan
        return w * (1.0 + r/30.0)
    except Exception:
        return np.nan

def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def daily_ma(series, window_days):
    # series is indexed by datetime daily (or resampled)
    return series.rolling(window=window_days, min_periods=max(1, window_days//2)).mean()

def weekly_bucket(dt_series):
    # Monday-start week
    return dt_series.dt.to_period("W-MON").dt.start_time

def plot_line(dfx, x, y, title, ylabel, xlabel="Date", marker="o", markersize=4, color=None, 
              show_grid=True, despine=True, rotate_x=False, date_locator=None, 
              date_formatter=None, linewidth=1.5):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dfx[x], dfx[y], marker=marker, markersize=markersize, color=color, linewidth=linewidth)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if show_grid:
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    
    if rotate_x:
        ax.tick_params(axis='x', rotation=45)
    
    if date_locator:
        ax.xaxis.set_major_locator(date_locator)
    if date_formatter:
        ax.xaxis.set_major_formatter(date_formatter)
    if despine:
        sns.despine(ax=ax)
    
    st.pyplot(fig)

def plot_two_axis(dfx, x, y1, y2, title, y1_label, y2_label):
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

def week_bounds(today=None):
    """Monday -> Sunday"""
    if today is None:
        today = pd.Timestamp.today().normalize()
    else:
        today = pd.to_datetime(today).normalize()
    start = today - pd.Timedelta(days=today.weekday())
    end = start + pd.Timedelta(days=6)
    return start, end

def safe_minimal_last(df, date_col, value_col):
    if df is None or value_col is None:
        return None
    if date_col not in df.columns or value_col not in df.columns:
        return None
    tmp = df[[date_col, value_col]].copy()
    tmp = tmp.dropna(subset=[date_col, value_col]).sort_values(date_col)
    if tmp.empty:
        return None
    return tmp[value_col].iloc[-1]
def recovery_zone(x):
    if x >= 0.7: return "🟢 ⬆️ Ready"
    if x >= 0.55: return "🟡 Moderate"
    return "🔴 ⬇️ Low"



# -------------------------
# Uploads (hidden after loaded)
# -------------------------

# ---------- helpers ----------
def all_loaded() -> bool:
    return all(st.session_state.get(k) is not None for k in ["df_workouts", "df_sleep", "df_recovery"])

def load_df_from_upload(uploaded_file):
    data = uploaded_file.getvalue()
    return pd.read_csv(io.BytesIO(data))

def normalize_workouts(df):
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    if "EXERCISE_NAME" in df.columns:
        df["EXERCISE_NAME"] = df["EXERCISE_NAME"].astype(str).str.strip()
    for col in ["WEIGHT_LBS", "REPS", "RPE", "VOLUME"]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    if set(["WEIGHT_LBS", "REPS"]).issubset(df.columns):
        df["est_1RM"] = df.apply(lambda r: epley_1rm(r["WEIGHT_LBS"], r["REPS"]), axis=1)

    if "DATE" in df.columns:
        df["Date"] = df["DATE"].dt.floor("D")
        df["DAY"] = df["Date"]
    return df

def normalize_sleep(df):
    df.columns = make_unique_columns(df.columns)
    if "Date" not in df.columns:
        for cand in ["DATE", "day", "date"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "Date"})
                break
    df = coerce_date(df, "Date")
    for cand in ["Score", "Wake Count", "Efficiency", "Asleep hrs", "InBed hrs",
                 "REM hrs", "Light hrs", "Deep hrs"]:
        if cand in df.columns:
            df[cand] = safe_numeric(df[cand])
    return df

def normalize_recovery(df):
    df.columns = make_unique_columns(df.columns)
    if "Date" not in df.columns:
        for cand in ["DATE", "day", "date"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "Date"})
                break
    df = coerce_date(df, "Date")
    for cand in ["Sigmoid Recovery Score", "RECOVERY_SCORE_RAW", "Stress_prev_day",
                 "Overnight HRV", "Resting Heart Rate", "Score"]:
        if cand in df.columns:
            df[cand] = safe_numeric(df[cand])
    return df

# ---------- init state ----------
if "show_uploads" not in st.session_state:
    st.session_state.show_uploads = True

st.subheader("📥 Upload your cleaned CSVs")

# status + button
left, right = st.columns([3, 1])
with left:
    w_ok = "✅" if st.session_state.get("df_workouts") is not None else "⬜️"
    s_ok = "✅" if st.session_state.get("df_sleep") is not None else "⬜️"
    r_ok = "✅" if st.session_state.get("df_recovery") is not None else "⬜️"
    st.caption(f"{w_ok} Workouts • {s_ok} Sleep • {r_ok} Recovery")

with right:
    if all_loaded():
        st.session_state.show_uploads = False
        if st.button("Change files"):
            for k in ["workouts", "sleep", "recovery", "df_workouts", "df_sleep", "df_recovery"]:
                st.session_state.pop(k, None)
            st.session_state.show_uploads = True
            st.rerun()
    else:
        st.session_state.show_uploads = True

# uploaders (fully hidden once loaded)
if st.session_state.show_uploads:
    with st.expander("Upload panel", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            up_workouts = st.file_uploader("Workouts: clean_strong_workouts.csv", type=["csv"], key="workouts")
        with c2:
            up_sleep = st.file_uploader("Sleep: clean_sleep_data.csv", type=["csv"], key="sleep")
        with c3:
            up_recovery = st.file_uploader("Recovery: clean_recovery_data.csv", type=["csv"], key="recovery")

        # IMPORTANT: persist + normalize immediately when uploads exist
        if up_workouts is not None and st.session_state.get("df_workouts") is None:
            st.session_state.df_workouts = normalize_workouts(load_df_from_upload(up_workouts))

        if up_sleep is not None and st.session_state.get("df_sleep") is None:
            st.session_state.df_sleep = normalize_sleep(load_df_from_upload(up_sleep))

        if up_recovery is not None and st.session_state.get("df_recovery") is None:
            st.session_state.df_recovery = normalize_recovery(load_df_from_upload(up_recovery))


# Stop until all 3 are loaded (AFTER the persist step above)
# ---------- downstream dataframes (always from session_state) ---------
if not all_loaded():
    st.info("Upload the 3 CSVs to unlock the dashboard.")
    st.stop()
# Use dataframes ONLY from session_state from here onward
workouts = st.session_state.df_workouts
sleep    = st.session_state.df_sleep
recovery = st.session_state.df_recovery

# -------------------------
# Tabs
# -------------------------
tab0, tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🏋️ Workouts", "😴 Sleep", "🧠 Recovery", "🔗 Correlations"])
# =========================
# TAB 0 — HOME
# =========================

with tab0:
    st.header("🏠 Weekly Snapshot")
    start_wk, end_wk = week_bounds()
    st.caption(f"Week: {start_wk.date()} → {end_wk.date()}")
    st.markdown("---")
    # compute last dates safely
    st.subheader("🔐 Data Integrity")
    last_workouts = workouts["Date"].max() if "Date" in workouts.columns else workouts["DATE"].max()
    last_sleep    = sleep["Date"].max() if sleep is not None and "Date" in sleep.columns else None
    last_recovery = recovery["Date"].max() if recovery is not None and "Date" in recovery.columns else None

    # freshness (days old)
    today = pd.Timestamp.today().normalize()
    def age_days(ts):
        if ts is None or pd.isna(ts): 
            return None
        return int((today - pd.to_datetime(ts).normalize()).days)

    a_w = age_days(last_workouts)
    a_s = age_days(last_sleep)
    a_r = age_days(last_recovery)

    c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.2, 2.4])
    c1.metric("Workouts last", f"{pd.to_datetime(last_workouts):%b %d}", f"{a_w}d old" if a_w is not None else "—",
              delta_arrow="off")
    c2.metric("Sleep last",    f"{pd.to_datetime(last_sleep):%b %d}" if last_sleep is not None else "—",
            f"{a_s}d old" if a_s is not None else "—", delta_arrow="off")
    c3.metric("Recovery last", f"{pd.to_datetime(last_recovery):%b %d}" if last_recovery is not None else "—",
            f"{a_r}d old" if a_r is not None else "—", delta_arrow="off")

    # optional: overall status label
    overall_age = max([x for x in [a_w, a_s, a_r] if x is not None], default=None)
    label = "🟢 Fresh" if (overall_age is not None and overall_age <= 1) else ("🟡 Slightly delayed" if (overall_age is not None and overall_age <= 3) else "🔴 Outdated")
    c4.markdown(f"**Data status:** {label}")
    st.markdown("---")
     # -------------------------
    # Weekly workouts snapshot
    # -------------------------
    if workouts is None:
        st.info("Upload workouts CSV to see weekly snapshot.")
    else:
        if "DAY" not in workouts.columns or workouts["DAY"].isna().all():
            st.warning("Workouts CSV needs a DATE/Date column so the app can create 'DAY'.")
        else:
            wk = workouts[(workouts["DAY"] >= start_wk) & (workouts["DAY"] <= end_wk)].copy()

            # Workouts (sessions): unique (DAY, WORKOUT_NAME) if available
            if "WORKOUT_NAME" in wk.columns:
                workouts_count = wk.dropna(subset=["WORKOUT_NAME"]).groupby(["DAY", "WORKOUT_NAME"]).ngroups
            else:
                workouts_count = wk["DAY"].nunique()

            # Time exercised: MAX duration per session, then sum
            total_minutes = None
            if "DURATION_MIN" in wk.columns:
                wk["DURATION_MIN"] = pd.to_numeric(wk["DURATION_MIN"], errors="coerce")

                if "WORKOUT_NAME" in wk.columns:
                    total_minutes = (
                        wk.dropna(subset=["DURATION_MIN"])
                          .groupby(["DAY", "WORKOUT_NAME"])["DURATION_MIN"]
                          .max()
                          .sum()
                    )
                else:
                    total_minutes = wk.groupby("DAY")["DURATION_MIN"].max().sum()

            total_hours = None if total_minutes is None else float(total_minutes) / 60.0

            # -------------------------
            # Latest values (sleep/recovery)
            # -------------------------
            st.subheader("📊 Key Metrics")
            last_sigmoid = safe_minimal_last(recovery, "Date", "Sigmoid Recovery Score") if recovery is not None else None
            sleep_score_col = pick_col(recovery, ["Score", "Sleep Score", "SleepScore", "SCORE"])
            sleep_hrv_col   = pick_col(recovery, ["Overnight HRV", "Avg. HRV", "HRV", "7d Avg"])

            last_sleep_score = safe_minimal_last(recovery, "Date", sleep_score_col)
            last_hrv        = safe_minimal_last(recovery, "Date", sleep_hrv_col)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Workouts (week)", int(workouts_count) if workouts_count is not None else "—")
            c2.metric("Time exercised (hrs)", f"{total_hours:.1f}" if total_hours is not None else "—",
                      delta=f"4 Hrs goal", delta_arrow="off", 
                      delta_color="normal" if total_hours is not None and total_hours >=4 else "inverse")
            if last_sigmoid is None or pd.isna(last_sigmoid):
                c3.metric("Last Recovery", "—", "No data")
            else:
                c3.metric(
                    "Last Recovery",
                    f"{last_sigmoid:.3f}",
                    recovery_zone(last_sigmoid), delta_arrow="off",
                    delta_color="normal" if last_sigmoid is not None and last_sigmoid >= 0.7 else ("inverse" if last_sigmoid is not None and last_sigmoid >= 0.55 else "inverse"))            
            c4.metric("Last sleep score %", f"{float(last_sleep_score):.0f}" if last_sleep_score is not None else "—", f"Excellent" if last_sleep_score is not None and last_sleep_score >= 85 else ("Fair" if last_sleep_score is not None and last_sleep_score >= 70 else "Poor"), delta_arrow="off")
            c5.metric("Last HRV (ms)", f"{float(last_hrv):.0f}" if last_hrv is not None and str(last_hrv) != "nan" else "—",
                      delta="Bad" if last_hrv is not None and last_hrv < 45 else "Good" if last_hrv is not None and last_hrv <= 60 else "Excellent", 
                      delta_arrow="off", delta_color="normal" if last_hrv is not None and last_hrv >= 45 else ("inverse" if last_hrv is not None and last_hrv < 60 else "off"))

            st.subheader("📈 Recent Trends")
            last_sigmoid_nap = safe_minimal_last(recovery, "Date", "Sigmoid with Nap") if recovery is not None else None
            last_delta = safe_minimal_last(recovery, "Date", "DELTA_NAP") if recovery is not None else None
            last_nap_status = safe_minimal_last(recovery, "Date", "Nap_Status") if recovery is not None else None
            c1, c2, c3 = st.columns(3)
            c1.metric("Recovery with Nap", f"{last_sigmoid_nap:.3f}" if last_sigmoid_nap is not None else "—",
                      recovery_zone(last_sigmoid_nap), delta_arrow="off",
                      delta_color="normal" if last_sigmoid_nap is not None and last_sigmoid_nap > last_sigmoid else ("inverse" if last_sigmoid_nap is not None and last_sigmoid_nap < last_sigmoid else "off"))
            c2.metric("Δ Nap Effect", f"{last_delta:.3f}" if last_delta is not None else "—",
                      f"{'⬆️ Positive' if last_delta is not None and last_delta > 0 else ('⬇️ Negative' if last_delta is not None and last_delta < 0 else 'No effect')}",
                      delta_arrow="off", delta_color="normal" if last_delta is not None and last_delta > 0 else ("inverse" if last_delta is not None and last_delta < 0 else "off"))
            c3.metric("Last Nap Status", last_nap_status if last_nap_status is not None else "No nap", delta_arrow="off")
    st.markdown("---")

    #---------------------------
    # Naps summary
    #--------------------------
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("💤 Naps logged")
        window = st.segmented_control(
            "Select nap days window for avg:",
            options=[7, 14, 30, 60, 90],
            default=14,
            key="nap_avg_window",
            selection_mode="single", format_func=lambda x: f"{x} days")
        if window is None:
            window = 14
            st.markdown("Please select a number of days")
        sleep = sleep.sort_values("Date", ascending=True)
        avg_nap = sleep.tail(window)["Asleep_Nap"].dropna().mean()
        try:
            if pd.isna(avg_nap):
                col1.metric(f"💤 Avg nap time (last {window} days)", " 0 minutes")
                st.write("No Naps Logged")
            else:
                col1.metric(f"💤 Avg nap time (last {window} days)", f"{avg_nap:.1f} minutes")
        except Exception as e:
            st.error(f"Error computing nap frequency: {e}")
            st.error("Please select a number")

    with col2:
        st.subheader("📅 Nap Days")
        window2 = st.segmented_control(
            "Select nap days window for avg:",
            options=[7, 14, 30, 60, 90],
            default=14,
            key="nap_avg_window2",
            selection_mode="single", format_func=lambda x: f"{x} days")
        try:
            if pd.isna(avg_nap):
                st.write("No Day Naps Logged")
            else:
                start_date = sleep["Date"].max() - pd.Timedelta(days=window2)
                nap_data = sleep[sleep["Date"] >= start_date].dropna(subset=["Asleep_Nap"])
                n_naps = nap_data.loc[nap_data["Asleep_Nap"].notna() & (nap_data["Asleep_Nap"] > 0)].shape[0]
                col2.metric("Naps logged", f"{n_naps}")
        except Exception as e:
            st.error(f"Error computing nap frequency: {e}")
            st.error("Please select a number")
    with col3:
        st.subheader("📈 Nap Frequency")
        window3 = st.segmented_control(
            "Select nap days window for avg:",
            options=[7, 14, 30, 60, 90],
            default=14,
            key="nap_avg_window3",
            selection_mode="single", format_func=lambda x: f"{x} days")
        try:
            if pd.isna(avg_nap):
                st.write("No Frequency Naps Logged")
            else:
                start_date = sleep["Date"].max() - pd.Timedelta(days=window3)
                nap_data = sleep[sleep["Date"] >= start_date].dropna(subset=["Asleep_Nap"])
                sleep_filtered = sleep[sleep["Date"] >= start_date]
                freq_val = (nap_data[nap_data["Asleep_Nap"].notna() & (nap_data["Asleep_Nap"] > 0)].shape[0] / sleep_filtered.shape[0]) * 100
                def freq_label(x):
                    if pd.isna(x):
                        return "No data"
                    if x <= 15: return "🔴 Low"
                    if x <= 30: return "🟡 Moderate"
                    return "🟢 High"
                col3.metric("Nap frequency", f"{freq_val:.1f} %", delta=freq_label(freq_val))
                st.caption(f"Naps on {nap_data[nap_data['Asleep_Nap'].notna() & (nap_data['Asleep_Nap'] > 0)].shape[0]} of the last {window3} days")
        except Exception as e:
            st.error(f"Error computing nap frequency: {e}")
            st.error("Please select a number")
    st.markdown("---")
    # -------------------------
    # Quick trends (independent)
    # -------------------------
    left, right = st.columns(2)

    with left:
        st.subheader("🧠 Recovery (last 14 days)")
        if recovery is not None and {"Date", "Sigmoid Recovery Score"}.issubset(recovery.columns):
            tmp = recovery.dropna(subset=["Date", "Sigmoid Recovery Score"]).sort_values("Date").tail(14)
            if tmp.empty:
                st.info("Recovery CSV loaded but no usable rows.")
            else:
                fig, ax = plt.subplots(figsize=(7,3))
                tmp_avg = tmp["Sigmoid Recovery Score"].mean()
                sld1 = st.slider("Select days for moving average", 2, 11, 5, 1, width=250)
                ax.plot(tmp["Date"], tmp["Sigmoid Recovery Score"], marker="o", markersize=3, color="green")
                roll_avg_recovery = tmp["Sigmoid Recovery Score"].rolling(window=sld1).mean()
                ax.plot(tmp["Date"], roll_avg_recovery, color="orange", linestyle="--", 
                        alpha=0.25, label=f"MA {sld1} days {roll_avg_recovery.iloc[-1]:.2f}")
                ax.axhline(tmp_avg, color="blue", linestyle=":", alpha=0.6, label=f"Avg {tmp_avg:.2f}")
                ax.set_xlabel("")
                ax.set_ylim(0, 1)
                ax.legend(loc="lower left")
                ax.tick_params(axis='x', rotation=45, labelsize=6)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.set_ylabel("Sigmoid")
                sns.despine(ax=ax)
                st.pyplot(fig)
        else:
            st.info("No recovery data uploaded yet.")

    with right:
        st.subheader("🧠 Recovery (last 14 days) with Nap")
        if recovery is not None and {"Date", "Sigmoid with Nap"}.issubset(recovery.columns):
            tmp2 = recovery.dropna(subset=["Date", "Sigmoid with Nap"]).sort_values("Date").tail(14)
            last_recovery_nap = tmp2["Sigmoid with Nap"].iloc[-1] if not tmp2.empty else None
            if tmp2.empty:
                st.info("Recovery CSV loaded but no usable rows.")
            else:
                fig, ax = plt.subplots(figsize=(7,3))
                tmp2_avg = tmp2["Sigmoid with Nap"].mean()
                sld_sig_nap = st.slider("Select days for moving average", 2, 11, 5, 1, width=250, key="sig_nap_ma_slider")
                ax.plot(tmp2["Date"], tmp2["Sigmoid with Nap"], marker="o", markersize=3, color="seagreen")
                roll_avg_recovery_nap = tmp2["Sigmoid with Nap"].rolling(window=sld_sig_nap).mean()
                ax.plot(tmp2["Date"], roll_avg_recovery_nap, color="orange", linestyle="--", 
                        alpha=0.25, label=f"MA {sld_sig_nap} days {roll_avg_recovery_nap.iloc[-1]:.2f}")
                ax.axhline(tmp2_avg, color="blue", linestyle=":", alpha=0.6, label=f"Avg {tmp2_avg:.2f}")
                ax.set_xlabel("")
                ax.set_ylim(0,1)
                ax.legend(loc="lower left")
                ax.tick_params(axis='x', rotation=45, labelsize=6)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
                ax.set_ylabel("Sigmoid")
                sns.despine(ax=ax)
                st.pyplot(fig)
        else:
            st.info("No recovery data uploaded yet.")
    with st.expander("Recovery Insights", icon="🧠",expanded=False):
        vol_no_nap = recovery["Sigmoid Recovery Score"].tail(14).std()
        vol_nap = recovery["Sigmoid with Nap"].tail(14).std()
        delta_window = st.slider("Select days window for Δ average", 2, 14, 7, 1, key="delta_avg_window", width=200)
        recovery_naps = recovery.dropna(subset=["Date", "DELTA_NAP"]).sort_values("Date").tail(delta_window)
        avg_delta = recovery_naps["DELTA_NAP"].mean() if not recovery_naps.empty else None
        st.write(f"Average Δ {delta_window} days: {avg_delta:.2f}" if avg_delta is not None else "Average Δ not available")
        if last_sigmoid is not None and last_sigmoid >= tmp_avg:
            st.write(f"14-day Last Recovery without Nap: {last_sigmoid:.2f} above Avg")
        else:
            st.write(f"14-day Last Recovery without Nap: {last_sigmoid_nap:.2f} below Avg")
        if last_recovery_nap is not None and last_recovery_nap >=tmp2_avg:
            st.write(f"14-day Last Recovery with Nap: {last_recovery_nap:.2f} above Avg")
        else:
            st.write(f"14-day Last Recovery with Nap: {last_recovery_nap:.2f} below Avg")
        st.write(f"Volatility (STD) without Nap (14 days): {vol_no_nap:.4f}")
        st.write(f"Volatility (STD) with Nap (14 days): {vol_nap:.4f}")

    st.caption("Note: Recovery with Nap may be higher than without nap depending on nap effect.", help="Nap effect is computed based on the duration of the nap and the hour it was taken. A positive nap effect indicates that the nap contributed positively to recovery, while a negative effect suggests it may have disrupted sleep patterns.")
    st.markdown("---")
    st.subheader("😴 Sleep score (last 14 days)")
    sleep_score_col = pick_col(recovery, ["Score", "Sleep Score", "SleepScore", "SCORE", "Score.1", "Score.2"]) if recovery is not None else None

    if recovery is not None and sleep_score_col is not None and "Date" in recovery.columns:
        tmp = recovery.dropna(subset=["Date", sleep_score_col]).sort_values("Date").tail(14)
        if tmp.empty:
            st.info("Sleep CSV loaded but no usable rows.")
        else:
            fig, ax = plt.subplots(figsize=(7,3))
            sld2 = st.slider("Select days for moving average", 2, 11, 5, 1, key="sleep_ma_slider", width=250)
            roll_avg_sleep = tmp[sleep_score_col].rolling(window=sld2).mean()
            tmp_avg_sleep = tmp[sleep_score_col].mean()
            ax.axhline(tmp_avg_sleep, color="blue", linestyle=":", alpha=0.6, label=f"Avg {tmp_avg_sleep:.0f}")
            ax.plot(tmp["Date"], tmp[sleep_score_col], marker="o", markersize=3, color="purple")
            ax.plot(tmp["Date"],roll_avg_sleep, color="orange", linestyle="--", 
                    alpha=0.4, label=f"MA {sld2} days {roll_avg_sleep.iloc[-1]:.0f}")
            ax.set_xlabel("")
            ax.legend(loc="lower left")
            ax.set_ylim(50,100)
            ax.set_ylabel(sleep_score_col)
            ax.tick_params(axis='x', rotation=45, labelsize=6)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
            sns.despine(ax=ax)
            st.pyplot(fig)
    else:
        st.info("No sleep data uploaded yet (or score column not detected).")
    st.markdown("---")

    with st.expander("🛠 Debug"):
        st.write("Workouts cols:", list(workouts.columns))
        st.write("Sleep cols:", list(sleep.columns))
        st.write("Recovery cols:", list(recovery.columns))
    st.markdown("--")
# =========================
# TAB 1 — WORKOUTS
# =========================
with tab1:
    st.header("🏋️ Workouts")

    if workouts is None:
        st.info("Upload your cleaned workouts CSV to see charts.")
    else:
        # basic checks
        req = {"Date", "EXERCISE_NAME"}
        if not req.issubset(workouts.columns):
            st.error(f"Workouts CSV must include at least: {req}")
        else:
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
                ax.set_xticks(np.arange(2)); ax.set_xticklabels(labs)
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
                ax.plot(daily["Date"], daily["est_1RM"], marker="s", label="Daily mean est.1RM", color="salmon", markersize=4)
                ax.plot(daily["Date"], daily["MA"], linestyle="--", label=f"{smooth_days}-day MA", color="yellow")
                ax.axvline(cva_ts, linestyle=":", linewidth=1)
                ax.set_title(f"{chosen_ex} — Comparative Pre & Post CVA", fontsize=14, fontweight="bold", pad=15)
                ax.set_xlabel("Date"); ax.set_ylabel("lb")
                ax.grid(axis="y", alpha=0.25)
                ax.set_axisbelow(True)
                sns.despine(ax=ax)
                ax.legend()
                st.pyplot(fig)

            # -------- 3) Weekly Volume chart (per exercise + total)
            st.subheader("📦 Weekly Volume (Exercise and Total)")
            if "VOLUME" in workouts.columns:
                workouts["Week"] = weekly_bucket(workouts["Date"])
                w_ex_week = workouts[workouts["EXERCISE_NAME"] == chosen_ex].groupby("Week", as_index=False)["VOLUME"].sum()
                w_all_week = workouts.groupby("Week", as_index=False)["VOLUME"].sum()

                cA, cB = st.columns(2)
                with cA:
                    plot_line(w_ex_week.sort_values("Week"), "Week", "VOLUME",
                              f"Weekly Volume — {chosen_ex}", "Total Volume (lb·reps)", xlabel="Week")
                with cB:
                    plot_line(w_all_week.sort_values("Week"), "Week", "VOLUME",
                              "Weekly Volume — ALL Exercises", "Total Volume (lb·reps)", xlabel="Week")

            # -------- 4) RPE trend (daily mean)
            st.subheader("🔥 RPE Trend (Daily)")
            if "RPE" in w.columns:
                rpe_daily = w.groupby("Date", as_index=False)["RPE"].mean().sort_values("Date")
                if rpe_daily["RPE"].notna().sum() == 0:
                    st.info("No RPE values recorded for this exercise yet.")
                else:
                    rpe_daily["MA"] = daily_ma(rpe_daily["RPE"], smooth_days)
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(rpe_daily["Date"], rpe_daily["RPE"], marker="o", label="Daily mean RPE", color="salmon", markersize=4)
                    ax.plot(rpe_daily["Date"], rpe_daily["MA"], linestyle="--", label=f"{smooth_days}-day MA", color="yellow")
                    ax.axvline(cva_ts, linestyle=":", linewidth=1)
                    ax.set_title(f"{chosen_ex} — RPE Trend & Post CVA", fontsize=14, fontweight="bold", pad=15)
                    ax.set_xlabel("Date"); ax.set_ylabel("RPE")
                    ax.grid(axis="y", alpha=0.25)
                    ax.set_axisbelow(True)
                    ax.tick_params(axis='x', rotation=45)
                    sns.despine(ax=ax)
                    ax.legend()
                    st.pyplot(fig)

            # -------- Summary table (exercise)
            with st.expander("📋 Show raw sets for this exercise"):
                st.dataframe(w.sort_values("DATE" if "DATE" in w.columns else "Date"))

# =========================
# TAB 2 — SLEEP
# =========================
with tab2:
    st.header("😴 Sleep")

    if sleep is None:
        st.info("Upload your clean sleep CSV to see charts.")
    else:
        if "Date" not in sleep.columns:
            st.error("Sleep CSV must include a Date column.")
        else:
            sleep = sleep.sort_values("Date")
            st.subheader("📋 Sleep table")
            st.dataframe(sleep)

            # Score
            if "Score" in sleep.columns:
                st.subheader("⭐ Sleep Score")
                plot_line(sleep.dropna(subset=["Score"]), "Date", "Score", "Sleep Score over time", "Score")

            # Stages
            stage_cols = [c for c in ["REM hrs", "Light hrs", "Deep hrs"] if c in sleep.columns]
            if stage_cols:
                st.subheader("🧱 Sleep Stages (hrs)")
                df_s = sleep[["Date"] + stage_cols].dropna(subset=["Date"]).copy()

                fig, ax = plt.subplots(figsize=(10, 4))
                bottom = np.zeros(len(df_s))
                for col in stage_cols:
                    vals = df_s[col].fillna(0).to_numpy()
                    ax.bar(df_s["Date"], vals, bottom=bottom, width=0.8, label=col)
                    bottom += vals
                ax.set_title("Sleep stages (stacked hours)", fontsize=14, fontweight="bold", pad=15)
                ax.set_xlabel("")
                ax.set_ylabel("Hours")
                sns.despine(ax=ax)
                ax.grid(axis="y", alpha=0.25)
                ax.tick_params(axis='x', rotation=45, labelsize=6)
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
                ax.set_axisbelow(True)
                ax.legend()
                st.pyplot(fig)

            # Wake Count
            if "Wake Count" in sleep.columns:
                st.subheader("🌙 Wake Count")
                plot_line(sleep.dropna(subset=["Wake Count"]), "Date", "Wake Count", 
                        "Wake Count over time", "Count", 
                        marker=None, color="purple", xlabel="",
                        rotate_x=True, date_locator=mdates.MonthLocator(interval=2), linewidth=0.7)
            # Naps
            left, right = st.columns(2)
            if "Asleep_Nap" in sleep.columns:
                with left:
                    st.subheader("💤 Nap Asleep (min)")
                    slider_nap2 = st.slider("Select number of days to show for Nap Asleep plot", 60, 365, 365, 1, key="nap_asleep_days")
                    recent_date = sleep["Date"].max()
                    start_date = recent_date - pd.Timedelta(days=slider_nap2)
                    filtered_nap = sleep[(sleep["Date"] >= start_date) & (sleep["Date"] <= recent_date)].copy()
                    df_plot = filtered_nap.dropna(subset=["Asleep_Nap"])
                    roll_avg_nap = df_plot["Asleep_Nap"].rolling(window=7, min_periods=1).mean()
                    

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(df_plot["Date"], df_plot["Asleep_Nap"], color="teal", linewidth=1.5, label="Nap Asleep")
                    ax.plot(df_plot["Date"], roll_avg_nap, marker="o", markersize=2, color="salmon", label=f"7-day MA", 
                            linewidth=1, linestyle="--")
                    ax.set_title("Nap Asleep over time")
                    ax.set_ylabel("Minutes")
                    ax.tick_params(axis='x', rotation=45)
                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
                    ax.grid(axis="y", alpha=0.25)
                    ax.set_axisbelow(True)
                    ax.legend()
                    sns.despine(ax=ax)
                    st.pyplot(fig)

                    # Monthly total naps 
                # TBD

            else:
                st.info("Column 'InBed_Nap' not found in sleep data.")
            with right:
                st.subheader("🛏️ Nap Asleep (min)")
                if "Asleep_Nap" in sleep.columns:
                    slider_nap = st.slider("Select number of days to show for Nap Asleep plot", 60, 365, 365, 1, key="nap_asleep")
                    recent_date = sleep["Date"].max()
                    start_date = recent_date - pd.Timedelta(days=slider_nap)
                    sleep_filtered = sleep[(sleep["Date"] >= start_date) & (sleep["Date"] <= recent_date)].copy()
                    df_nap = sleep_filtered.dropna(subset=["Asleep_Nap"])[["Date", "Asleep_Nap"]].copy()
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.bar(df_nap["Date"], df_nap["Asleep_Nap"], color="teal", width=0.8)
                    ax.set_title("Nap Asleep over time")
                    ax.set_ylabel("Minutes")
                    ax.tick_params(axis='x', rotation=45)
                    ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
                    ax.grid(axis="y", alpha=0.25)
                    ax.set_axisbelow(True)
                    sns.despine(ax=ax)
                    st.pyplot(fig)

                    # Monthly total nap inbed
                    sleep_monthly = sleep_filtered.set_index("Date").resample("M")["Asleep_Nap"].sum().reset_index()
                    st.subheader("🗓️ Monthly Nap Asleep Total (min)")
                    plot_line(sleep_monthly.dropna(subset=["Asleep_Nap"]), "Date", "Asleep_Nap",
                            "Monthly Nap Asleep Total", "Minutes", 
                            marker="o", color="coral", xlabel="",
                            rotate_x=True, date_locator=mdates.MonthLocator(interval=1), show_grid=True, date_formatter=mdates.DateFormatter('%b-%Y'))
                else:
                    st.info("Column 'InBed_Nap' not found in sleep data.")

 
# =========================
# TAB 3 — RECOVERY
# =========================
with tab3:
    st.header("🧠 Recovery")

    if recovery is None:
        st.info("Upload your clean recovery CSV to see charts.")
    else:
        if "Date" not in recovery.columns:
            st.error("Recovery CSV must include a Date column.")
        else:
            recovery = recovery.sort_values("Date")

            st.subheader("📋 Recovery table")
            st.dataframe(recovery)

            # Main recovery score (sigmoid)
            if "Sigmoid Recovery Score" in recovery.columns:
                st.subheader("🧠 Sigmoid Recovery Score (0–1)")
                plot_line(
                    recovery.dropna(subset=["Sigmoid Recovery Score"]),
                    "Date",
                    "Sigmoid Recovery Score",
                    "Sigmoid Recovery Score over time",
                    "Score", xlabel="", color="seagreen", rotate_x=True, 
                    date_locator=mdates.DayLocator(interval=2), 
                    date_formatter=mdates.DateFormatter('%b-%d')
                )

            # Components (choose what you want)
            st.subheader("🧩 Components")
            candidates = [
                "Stress_prev_day",
                "Overnight HRV",
                "Resting Heart Rate",
                "Score",
                "RECOVERY_SCORE_RAW",
            ]
            available = [c for c in candidates if c in recovery.columns]
            if available:
                chosen = st.multiselect("Pick component(s) to plot:", available, default=available[:3])
                for col in chosen:
                    plot_line(recovery.dropna(subset=[col]), "Date", col, f"{col} over time", col)
            else:
                st.info("No component columns detected (Stress_prev_day / Overnight HRV / etc.).")

# =========================
# TAB 4 — CORRELATIONS
# =========================
with tab4:
    st.header("🔗 Correlations (Workout vs Recovery/Sleep)")

    if workouts is None or (sleep is None and recovery is None):
        st.info("Upload workouts + (sleep and/or recovery) to compute correlations.")
    else:
        # Workout weekly aggregation (works even if you train 3–4x/week)
        if "VOLUME" in workouts.columns:
            tmp = workouts.copy()
            tmp["Week"] = weekly_bucket(tmp["Date"])
            wk_total = tmp.groupby("Week", as_index=False).agg(
                WeeklyVolume=("VOLUME", "sum"),
                WeeklyRPE=("RPE", "mean") if "RPE" in tmp.columns else ("VOLUME", "size")
            )

            st.subheader("📦 Weekly Training Load")
            st.dataframe(wk_total.sort_values("Week"))

            # Recovery weekly aggregation
            if recovery is not None and "Sigmoid Recovery Score" in recovery.columns:
                rec = recovery.copy()
                rec["Week"] = weekly_bucket(rec["Date"])
                wk_rec = rec.groupby("Week", as_index=False).agg(
                    WeeklyRecovery=("Sigmoid Recovery Score", "mean"),
                    WeeklyStressPrev=("Stress_prev_day", "mean") if "Stress_prev_day" in rec.columns else ("Sigmoid Recovery Score", "size")
                )

                merged = pd.merge(wk_total, wk_rec, on="Week", how="inner").sort_values("Week")
                st.subheader("🔗 Weekly Volume vs Weekly Recovery")
                st.dataframe(merged)

                if not merged.empty:
                    corr = merged["WeeklyVolume"].corr(merged["WeeklyRecovery"])
                    st.write(f"**Correlation (WeeklyVolume vs WeeklyRecovery):** {corr:.2f} (n={len(merged)})")

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.scatter(merged["WeeklyVolume"], merged["WeeklyRecovery"])
                    ax.set_xlabel("WeeklyVolume (lb·reps)")
                    ax.set_ylabel("WeeklyRecovery (mean sigmoid)")
                    ax.set_title("Weekly Volume vs Weekly Recovery")
                    ax.grid(True, alpha=0.25)
                    st.pyplot(fig)

                    # Optional: show both over time
                    st.subheader("📈 Weekly Volume + Recovery (time)")
                    plot_two_axis(
                        merged, "Week",
                        "WeeklyVolume", "WeeklyRecovery",
                        "Weekly Volume vs Recovery over time",
                        "WeeklyVolume", "WeeklyRecovery"
                    )

            # Sleep weekly aggregation
            sleep_score_col = pick_col(sleep, ["Score", "Sleep Score", "SleepScore", "SCORE", "Score.1", "Score.2"]) if sleep is not None else None

            if sleep is not None and sleep_score_col is not None:
                sl = sleep.copy()
                sl["Week"] = weekly_bucket(sl["Date"])
                wk_sl = sl.groupby("Week", as_index=False).agg(
                    WeeklySleepScore=(sleep_score_col, "mean"),
                    WeeklyAsleep=("Asleep hrs", "mean") if "Asleep hrs" in sl.columns else (sleep_score_col, "size"),
                )

            else:
                st.info("Workouts file needs a VOLUME column to compute weekly load correlations.")

st.caption(
    "Tip: If you only train 3–4 days/week, use weekly aggregation (Volume / mean Recovery / mean Sleep) "
    "to avoid the mismatch between daily sleep and training frequency."
)