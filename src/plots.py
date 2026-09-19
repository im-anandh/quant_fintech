import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def line(df, title, y_title=None):
    fig = go.Figure()
    for c in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], name=str(c), mode="lines"))
    return fig.update_layout(title=title, template="plotly_white", yaxis_title=y_title,
                             legend=dict(orientation="h", y=-0.2))


def heatmap(df, title):
    return px.imshow(df, text_auto=".2f", title=title, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)


def param_heatmap(sweep, x, y, z, title):
    grid = sweep.pivot(index=y, columns=x, values=z)
    return px.imshow(grid, text_auto=".2f", aspect="auto", title=title,
                     color_continuous_scale="RdYlGn", labels=dict(x=x, y=y, color=z))


def signal_chart(close, trades, title):
    fig = go.Figure(go.Scatter(x=close.index, y=close, name="Price", mode="lines"))
    if len(trades):
        fig.add_trace(go.Scatter(x=trades["Entry Date"], y=trades["Entry Price"], name="Buy",
                                 mode="markers", marker=dict(symbol="triangle-up", size=10, color="green")))
        closed = trades[~trades["Open"]]
        fig.add_trace(go.Scatter(x=closed["Exit Date"], y=closed["Exit Price"], name="Sell",
                                 mode="markers", marker=dict(symbol="triangle-down", size=10, color="red")))
    return fig.update_layout(title=title, template="plotly_white", legend=dict(orientation="h", y=-0.2))


def fan_chart(paths, title):
    x = np.arange(1, paths.shape[1] + 1)
    p5, p50, p95 = np.percentile(paths, [5, 50, 95], axis=0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=p95, name="95th pct", line=dict(width=0)))
    fig.add_trace(go.Scatter(x=x, y=p5, name="5th-95th pct range", fill="tonexty",
                             line=dict(width=0), fillcolor="rgba(70,130,180,0.25)"))
    fig.add_trace(go.Scatter(x=x, y=p50, name="Median", line=dict(color="steelblue")))
    return fig.update_layout(title=title, template="plotly_white",
                             xaxis_title="Trading days ahead", yaxis_title="Simulated price")
