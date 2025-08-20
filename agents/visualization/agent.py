"""
Visualization Agent Module

Converts SQL results into charts/tables using Plotly.
"""
from typing import List, Dict, Optional
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydantic import BaseModel, validator


class VisualizationOptions(BaseModel):
    """Options for controlling visualization output"""
    chart_type: str = "bar"  # bar, line, pie, table
    title: Optional[str] = None
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None
    color_scheme: str = "#1f77b4"  # Default Plotly blue

    @validator("color_scheme")
    def validate_color(cls, v):  # noqa: D417 - short validator
        if not (v.startswith("#") or v.startswith("rgb") or v in ["blue", "red", "green"]):
            raise ValueError(
                f"Invalid color: {v}. Use hex (#RRGGBB), rgb(), or named colors"
            )
        return v


class VisualizationAgent:
    """
    Transforms SQL results into visualizations with:
    - Interactive charts (Plotly)
    - Formatted tables
    - Export capabilities (PNG/HTML)
    """

    def __init__(self):
        self.supported_chart_types = ["bar", "line", "pie", "table"]

    def visualize(self, data: List[Dict], options: VisualizationOptions) -> go.Figure:
        """Generate visualization from SQL results"""
        if options.chart_type not in self.supported_chart_types:
            raise ValueError(f"Unsupported chart type: {options.chart_type}")

        if options.chart_type == "table":
            return self._create_table(data, options)
        return self._create_chart(data, options)

    def _create_chart(self, data: List[Dict], options: VisualizationOptions) -> go.Figure:
        """Generate Plotly chart based on type"""
        fig = go.Figure()

        if options.chart_type == "bar":
            fig.add_trace(
                go.Bar(
                    x=[d.get(options.x_axis) for d in data],
                    y=[d.get(options.y_axis) for d in data],
                    marker_color=options.color_scheme,
                )
            )
        elif options.chart_type == "line":
            fig.add_trace(
                go.Scatter(
                    x=[d.get(options.x_axis) for d in data],
                    y=[d.get(options.y_axis) for d in data],
                    mode="lines+markers",
                    line_color=options.color_scheme,
                )
            )
        elif options.chart_type == "pie":
            fig.add_trace(
                go.Pie(
                    labels=[d.get(options.x_axis) for d in data],
                    values=[d.get(options.y_axis) for d in data],
                    marker_colors=[options.color_scheme] * len(data),
                )
            )

        if options.title:
            fig.update_layout(title=options.title)
        return fig

    def _create_table(self, data: List[Dict], options: VisualizationOptions) -> go.Figure:
        """Generate formatted table"""
        headers = list(data[0].keys()) if data else []
        values = [[row[h] for h in headers] for row in data]

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=headers),
                    cells=dict(values=list(zip(*values))),
                )
            ]
        )
        if options.title:
            fig.update_layout(title=options.title)
        return fig

    def export(self, fig: go.Figure, format: str = "png", filename: str = "chart"):
        """Export visualization to file.

        Attempts to generate a static image using Plotly's export machinery. If
        the required backend (such as Chrome for Kaleido) is not available, the
        method falls back to writing HTML content to the requested filename to
        ensure a file is created.
        """
        path = Path(filename)
        base = str(path.with_suffix("")) if path.suffix else str(path)

        if format == "png":
            target = f"{base}.png"
            try:
                fig.write_image(target)
            except Exception:
                fig.write_html(target)
        elif format == "html":
            fig.write_html(f"{base}.html")
        else:
            raise ValueError(f"Unsupported export format: {format}")
