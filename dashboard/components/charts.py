"""
Reusable Plotly charts for PongRL Pro Dashboard.
Designed with a cohesive modern dark theme.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Professional Dark Theme Template
PLOTLY_DARK_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        plot_bgcolor="rgba(15, 17, 35, 0.6)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c4c7d5", family="'Inter', -apple-system, BlinkMacSystemFont, sans-serif", size=12),
        margin=dict(l=45, r=30, t=45, b=45),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#8e92a8")
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.06)",
            zerolinecolor="rgba(255, 255, 255, 0.12)",
            tickfont=dict(color="#8e92a8")
        ),
        legend=dict(
            bgcolor="rgba(20, 24, 45, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1,
            font=dict(color="#e2e4ef")
        )
    )
)


def plot_real_reward_curve(df: pd.DataFrame, title: str = "Empirical Training Reward Progression") -> go.Figure:
    """Plot actual recorded training reward curves."""
    fig = go.Figure()
    
    if "running_reward" in df.columns:
        # Raw / recorded running rewards
        fig.add_trace(go.Scatter(
            x=df['episode'],
            y=df['running_reward'],
            mode='lines',
            line=dict(color='rgba(0, 212, 255, 0.45)', width=1.5),
            name='Running Reward'
        ))
        
        # Smoothed moving averages
        if "moving_avg_50" in df.columns:
            fig.add_trace(go.Scatter(
                x=df['episode'],
                y=df['moving_avg_50'],
                mode='lines',
                line=dict(color='#00d4ff', width=2.5),
                name='50-Ep Moving Avg'
            ))
        if "moving_avg_100" in df.columns:
            fig.add_trace(go.Scatter(
                x=df['episode'],
                y=df['moving_avg_100'],
                mode='lines',
                line=dict(color='#00ffaa', width=3),
                name='100-Ep Moving Avg'
            ))
            
    elif "reward" in df.columns:
        fig.add_trace(go.Scatter(
            x=df['episode'],
            y=df['reward'],
            mode='lines',
            line=dict(color='rgba(0, 212, 255, 0.3)', width=1),
            name='Raw Reward'
        ))
        if "avg_reward" in df.columns:
            fig.add_trace(go.Scatter(
                x=df['episode'],
                y=df['avg_reward'],
                mode='lines',
                line=dict(color='#00d4ff', width=2.5),
                name='Moving Avg'
            ))

    # Add reference benchmark horizontal lines for Atari Pong
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255, 255, 255, 0.3)", annotation_text="Break-even (0.0)")
    fig.add_hline(y=21, line_dash="dot", line_color="rgba(0, 255, 170, 0.4)", annotation_text="Perfect Win (+21.0)")
    fig.add_hline(y=-21, line_dash="dot", line_color="rgba(255, 0, 110, 0.4)", annotation_text="Total Loss (-21.0)")

    fig.update_layout(
        title=f"📈 {title}",
        xaxis_title="Training Episode",
        yaxis_title="Score / Running Reward",
        template=PLOTLY_DARK_TEMPLATE,
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02)
    )
    return fig


def plot_algorithm_comparison_bar(perf_df: pd.DataFrame) -> go.Figure:
    """Bar chart comparing peak or documented reward across algorithms."""
    colors = {
        'REINFORCE': '#00d4ff',
        'DQN': '#7c3aed',
        'PPO': '#00ffaa'
    }
    bar_colors = [colors.get(algo, '#00d4ff') for algo in perf_df['Algorithm']]
    
    fig = go.Figure(data=[
        go.Bar(
            x=perf_df['Algorithm'],
            y=perf_df['Best Reward'],
            marker=dict(
                color=bar_colors,
                line=dict(color='rgba(255, 255, 255, 0.2)', width=1.5)
            ),
            text=[f"{v:+.2f}" for v in perf_df['Best Reward']],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="🏆 Algorithm Peak Performance Comparison (Pong)",
        xaxis_title="RL Algorithm",
        yaxis_title="Best Reward (Scale: -21 to +21)",
        template=PLOTLY_DARK_TEMPLATE,
    )
    return fig


def plot_algorithm_radar() -> go.Figure:
    """Radar chart comparing algorithm architectural trade-offs."""
    categories = [
        'Sample Efficiency',
        'Training Stability',
        'Simplicity of Tuning',
        'Peak Performance',
        'Visual Feature Learning'
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[4, 8, 7, 8, 9],
        theta=categories,
        fill='toself',
        fillcolor='rgba(124, 58, 237, 0.2)',
        line=dict(color='#7c3aed', width=2),
        name='DQN (Double Q-Learning)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[7, 9, 8, 9, 9],
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 255, 170, 0.2)',
        line=dict(color='#00ffaa', width=2),
        name='PPO (Actor-Critic)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[3, 5, 9, 6, 7],
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 212, 255, 0.2)',
        line=dict(color='#00d4ff', width=2),
        name='REINFORCE (Policy Gradient)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], color="#8e92a8", gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(color="#c4c7d5", gridcolor="rgba(255,255,255,0.08)"),
            bgcolor="rgba(15, 17, 35, 0.5)"
        ),
        showlegend=True,
        title="⚡ Algorithmic Capability & Trade-off Radar",
        template=PLOTLY_DARK_TEMPLATE
    )
    return fig
