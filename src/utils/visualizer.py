"""
Interactive Visualizations for Speech Emotion Recognition
Generates Plotly charts for Confidence Scores, Emotion Timeline,
Intensity Gauges, Radar Fingerprints, and Waveform visualizers.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.config import EMOTIONS, EMOTION_META

def plot_confidence_distribution(probabilities: dict[str, float], predicted_emotion: str) -> go.Figure:
    """
    Creates an interactive bar chart of confidence scores across all emotions.
    """
    emotions = list(probabilities.keys())
    scores = [probabilities[e] * 100 for e in emotions]
    colors = [EMOTION_META.get(e, {}).get("color", "#4A5568") for e in emotions]
    
    # Add border/opacity highlight for predicted emotion
    opacity = [1.0 if e == predicted_emotion else 0.65 for e in emotions]
    
    fig = go.Figure(
        data=[
            go.Bar(
                x=emotions,
                y=scores,
                marker=dict(
                    color=colors,
                    opacity=opacity,
                    line=dict(
                        color=["#FFFFFF" if e == predicted_emotion else "rgba(0,0,0,0)" for e in emotions],
                        width=[2.5 if e == predicted_emotion else 0 for e in emotions]
                    )
                ),
                text=[f"{s:.1f}%" for s in scores],
                textposition="outside",
                hoverinfo="x+y",
                hovertemplate="<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>"
            )
        ]
    )
    
    fig.update_layout(
        title=dict(text="<b>Emotion Confidence Distribution</b>", font=dict(size=18)),
        xaxis=dict(title="Emotion Category", tickfont=dict(size=13, family="sans-serif")),
        yaxis=dict(title="Probability (%)", range=[0, max(scores) + 15]),
        margin=dict(l=20, r=20, t=50, b=30),
        template="plotly_dark",
        height=340
    )
    return fig


def plot_emotion_radar(probabilities: dict[str, float]) -> go.Figure:
    """
    Generates a polar radar chart displaying the emotional fingerprint.
    """
    emotions = list(probabilities.keys())
    # Close polygon by repeating first value
    r_vals = [probabilities[e] * 100 for e in emotions] + [probabilities[emotions[0]] * 100]
    theta_vals = emotions + [emotions[0]]
    
    fig = go.Figure(
        data=go.Scatterpolar(
            r=r_vals,
            theta=theta_vals,
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.35)',
            line=dict(color='#6366F1', width=2.5),
            hoverinfo="r+theta"
        )
    )
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%"),
            angularaxis=dict(tickfont=dict(size=12))
        ),
        margin=dict(l=30, r=30, t=30, b=30),
        template="plotly_dark",
        height=340,
        showlegend=False
    )
    return fig


def plot_emotion_timeline(timeline_data: list[dict]) -> go.Figure:
    """
    Builds an interactive multi-line / area time-series chart showing
    how emotion probabilities and acoustic intensity evolve over the audio duration.
    """
    if not timeline_data:
        return go.Figure()

    df = pd.DataFrame(timeline_data)
    fig = go.Figure()

    # Add emotion probability curves
    for emo in EMOTIONS:
        col = f"prob_{emo.lower()}"
        if col in df.columns:
            color = EMOTION_META.get(emo, {}).get("color", "#4A5568")
            fig.add_trace(
                go.Scatter(
                    x=df["center_time"],
                    y=df[col] * 100,
                    mode="lines+markers",
                    name=f"{EMOTION_META.get(emo, {}).get('emoji', '')} {emo}",
                    line=dict(color=color, width=2.5),
                    marker=dict(size=5),
                    hovertemplate=f"<b>{emo}</b>: %{{y:.1f}}% at %{{x:.1f}}s<extra></extra>"
                )
            )

    # Add Emotion Intensity overlay (dotted gold line)
    if "intensity_percentage" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["center_time"],
                y=df["intensity_percentage"],
                mode="lines",
                name="⚡ Intensity",
                line=dict(color="#F6E05E", width=2, dash="dash"),
                hovertemplate="Intensity: %{y:.1f}%<extra></extra>"
            )
        )

    fig.update_layout(
        title=dict(text="<b>Dynamic Emotion Timeline (Fluctuations Over Time)</b>", font=dict(size=18)),
        xaxis=dict(title="Time (seconds)", showgrid=True),
        yaxis=dict(title="Intensity & Probability (%)", range=[0, 105], showgrid=True),
        margin=dict(l=20, r=20, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark",
        height=380,
        hovermode="x unified"
    )
    return fig


def plot_intensity_gauge(intensity_percentage: float, level_en: str, level_hi: str) -> go.Figure:
    """
    Renders an interactive dial/gauge for vocal emotion intensity.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=intensity_percentage,
            title={'text': f"<b>Vocal Intensity: {level_en} ({level_hi})</b>", 'font': {'size': 16}},
            number={'suffix': "%", 'font': {'size': 26, 'color': "#FFFFFF"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#6366F1", 'thickness': 0.3},
                'bgcolor': "rgba(0,0,0,0)",
                'borderwidth': 2,
                'bordercolor': "#4A5568",
                'steps': [
                    {'range': [0, 35], 'color': 'rgba(72, 187, 120, 0.4)'},     # Subtle (Green)
                    {'range': [35, 65], 'color': 'rgba(236, 201, 75, 0.4)'},    # Moderate (Yellow)
                    {'range': [65, 85], 'color': 'rgba(237, 137, 54, 0.4)'},    # High (Orange)
                    {'range': [85, 100], 'color': 'rgba(229, 62, 62, 0.4)'}     # Intense (Red)
                ],
                'threshold': {
                    'line': {'color': "#E53E3E", 'width': 4},
                    'thickness': 0.75,
                    'value': intensity_percentage
                }
            }
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=220
    )
    return fig


def plot_waveform(y: np.ndarray, sr: int) -> go.Figure:
    """
    Renders an audio waveform chart.
    """
    # Downsample for snappy browser rendering
    step = max(1, len(y) // 1000)
    y_sub = y[::step]
    time_axis = np.linspace(0, len(y) / sr, len(y_sub))
    
    fig = go.Figure(
        data=go.Scatter(
            x=time_axis,
            y=y_sub,
            mode='lines',
            line=dict(color='#38BDF8', width=1.2),
            hovertemplate="Time: %{x:.2f}s, Amp: %{y:.2f}<extra></extra>"
        )
    )
    fig.update_layout(
        title=dict(text="Audio Waveform (Amplitude vs Time)", font=dict(size=14)),
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Amplitude", range=[-1.05, 1.05]),
        margin=dict(l=20, r=20, t=35, b=20),
        template="plotly_dark",
        height=180,
        showlegend=False
    )
    return fig
