"""Local dashboard for EdgeAuth verification, PAD and latency JSON reports."""

import json
from statistics import NormalDist

import altair as alt
import streamlit as st

from edgeauth.evaluation import REPORT_SCHEMA, synthetic_report


def percent(value):
    return "—" if value is None else f"{value:.1%}"


def curve_chart(points, *, det: bool):
    if det:
        normal = NormalDist()

        def probit(value):
            return normal.inv_cdf(min(0.9999, max(0.0001, value)))

        marks = [
            {
                "x": probit(point["fmr"]),
                "y": probit(point["fnmr"]),
                "FMR": point["fmr"],
                "FNMR": point["fnmr"],
                "step": index,
            }
            for index, point in enumerate(points)
        ]
        axis = alt.Axis(
            values=[-3.09, -2.33, -1.64, -1.28, -0.67, 0, 0.67, 1.28, 1.64, 2.33, 3.09],
            labelExpr="['0.1%','1%','5%','10%','25%','50%','75%','90%','95%','99%','99.9%']"
            "[indexof([-3.09,-2.33,-1.64,-1.28,-0.67,0,0.67,1.28,1.64,2.33,3.09],datum.value)]",
        )
        x = alt.X("x:Q", title="False match rate", axis=axis)
        y = alt.Y("y:Q", title="False non-match rate", axis=axis)
        tips = [alt.Tooltip("FMR:Q", format=".2%"), alt.Tooltip("FNMR:Q", format=".2%")]
    else:
        marks = [
            {"FMR": point["fmr"], "TMR": 1 - point["fnmr"], "step": index}
            for index, point in enumerate(points)
        ]
        axis = alt.Axis(format=".0%")
        x = alt.X("FMR:Q", title="False match rate", axis=axis)
        y = alt.Y("TMR:Q", title="True match rate", axis=axis)
        tips = [alt.Tooltip("FMR:Q", format=".2%"), alt.Tooltip("TMR:Q", format=".2%")]
    return (
        alt.Chart(alt.Data(values=marks))
        .mark_line(color="#00a88f", strokeWidth=2)
        .encode(x=x, y=y, order="step:Q", tooltip=tips)
        .properties(height=290)
    )


def verification(report):
    if report["source"].startswith("synthetic"):
        st.info("Synthetic example scores. Upload a verification report to view measured results.")
    options = {"Calibration": report["calibration"]}
    options.update({f"Held-out · {key}": val for key, val in report["test"].items()})
    name = st.selectbox("Trial set", list(options), index=1)
    subset = options[name]
    point = subset["operating_point"]
    first, second, third, fourth = st.columns(4)
    first.metric("ROC AUC", percent(subset["roc_auc"]))
    second.metric("Equal error rate", percent(subset["eer"]))
    third.metric("False accept rate", percent(point["far"] if point else None))
    fourth.metric("False reject rate", percent(point["frr"] if point else None))
    st.caption(
        f"{subset['attempts']:,} attempts · {subset['screened_out']:,} screened out "
        f"· Calibration FMR target {report['target_fmr']:.2%}"
    )
    if point:
        st.write(
            f"Cosine threshold **{point['threshold']:.4f}** · "
            f"**{point['impostor_scored']:,}** impostor trials scored · "
            f"**{point['false_matches']:,}** false matches"
        )
    else:
        st.warning("Calibration did not select an operating point.")
    if subset["curve"]:
        roc, det = st.tabs(["ROC", "DET"])
        with roc:
            st.altair_chart(curve_chart(subset["curve"], det=False), width="stretch")
        with det:
            st.altair_chart(curve_chart(subset["curve"], det=True), width="stretch")
            st.caption("Normal-deviate axes. Plot endpoints are clipped to 0.01% and 99.99%.")
    else:
        st.info("The curve requires at least one scored genuine and impostor trial.")
    st.subheader("Held-out performance by capture condition")
    st.dataframe(
        [
            {
                "Condition": condition,
                "Attempts": item["attempts"],
                "Screened out": item["screened_out"],
                "EER": percent(item["eer"]),
                "FAR": percent(item["operating_point"]["far"] if item["operating_point"] else None),
                "FRR": percent(item["operating_point"]["frr"] if item["operating_point"] else None),
            }
            for condition, item in report["test"].items()
        ],
        hide_index=True,
    )


def pad(report):
    st.subheader("Presentation attack detection")
    st.caption(f"Bona-fide score cutoff {report['pad_cutoff']:.2f}")
    st.dataframe(
        [
            {
                "Class": row["label"],
                "Species": row["species"],
                "Presentations": row["presentations"],
                "Quality rejected": row["quality_rejected"],
                "Classifier error": percent(row["classifier_error_rate"]),
                "Pipeline error": percent(row["pipeline_error_rate"]),
            }
            for row in report["results"]
        ],
        hide_index=True,
    )


def benchmark(report):
    st.subheader("CPU inference benchmark")
    st.caption(
        f"{report['machine']} · OpenCV {report['opencv']} · "
        f"{report['measured_runs']} timed runs · {report['opencv_threads']} thread"
    )
    frame = [
        {
            "Model": variant,
            "Weights (MB)": round(item["recognition_weights_bytes"] / 1e6, 2),
            "Pipeline p50 (ms)": round(item["pipeline_p50_ms"], 2),
            "Pipeline p95 (ms)": round(item["pipeline_p95_ms"], 2),
            "Cosine with FP32": round(item["cosine_with_fp32_embedding"], 4),
        }
        for variant, item in report["variants"].items()
    ]
    st.dataframe(frame, hide_index=True)
    st.bar_chart(frame, x="Model", y=["Pipeline p50 (ms)", "Pipeline p95 (ms)"])


st.set_page_config(page_title="EdgeAuth · Verification Lab", page_icon="◉", layout="wide")
st.title("EdgeAuth")
st.caption("Face verification and edge inference lab")
uploaded = st.sidebar.file_uploader(
    "Load a verification, PAD or benchmark report",
    type="json",
    max_upload_size=10,
)
views = {
    REPORT_SCHEMA: verification,
    "edgeauth/pad-report/v1": pad,
    "edgeauth/benchmark/v1": benchmark,
}
try:
    report = synthetic_report() if uploaded is None else json.load(uploaded)
    views[report["schema"]](report)
except (KeyError, IndexError, TypeError, ValueError):
    st.error("This file is not a valid EdgeAuth report.")
