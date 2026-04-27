from __future__ import annotations

from html import escape
from typing import Sequence


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if isinstance(value, tuple):
        return " x ".join(str(v) for v in value)
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def html_card(
    title: str,
    metrics: dict[str, object],
    *,
    details: dict[str, object] | None = None,
    bars: Sequence[tuple[str, float, str]] = (),
) -> str:
    """Generate HTML for a card display compatible with _repr_html_.
    
    Returns a pure HTML string that can be used with IPython's _repr_html_
    protocol.
    """
    metric_blocks = "".join(
        (
            "<div style='padding:0.85rem 0.9rem;"
            "border:1px solid rgba(127,127,127,0.14);"
            "border-radius:12px;"
            "background:rgba(255,255,255,0.65);"
            "min-width:0;'>"
            f"<div style='font-size:0.8rem;opacity:0.72;margin-bottom:0.25rem;'>{escape(str(key))}</div>"
            f"<div style='font-size:1rem;font-weight:700;overflow-wrap:anywhere;'>{escape(_format_value(value))}</div>"
            "</div>"
        )
        for key, value in metrics.items()
    )

    detail_rows = (
        ""
        if not details
        else "".join(
            (
                "<div style='display:flex;gap:0.75rem;align-items:flex-start;"
                "padding:0.35rem 0;border-bottom:1px solid rgba(127,127,127,0.12);'>"
                f"<div style='width:8rem;flex:0 0 auto;opacity:0.72;'>{escape(str(key))}</div>"
                f"<div style='font-family:ui-monospace,SFMono-Regular,monospace;overflow-wrap:anywhere;'>"
                f"{escape(_format_value(value))}</div>"
                "</div>"
            )
            for key, value in details.items()
        )
    )

    bar_blocks = "".join(
        (
            "<div style='margin-top:0.85rem;'>"
            f"<div style='display:flex;justify-content:space-between;gap:1rem;"
            "font-size:0.84rem;margin-bottom:0.25rem;'>"
            f"<span>{escape(label)}</span><span>{value:.1f}%</span></div>"
            "<div style='height:0.55rem;border-radius:999px;overflow:hidden;"
            "background:rgba(127,127,127,0.14);'>"
            f"<div style='height:100%;width:{max(0.0, min(100.0, value)):.1f}%;"
            f"background:{escape(color)};border-radius:999px;'></div>"
            "</div>"
            "</div>"
        )
        for label, value, color in bars
    )

    html = f"""
        <div style="padding: 0.05rem 0;">
          <div style="
            border: 1px solid rgba(127, 127, 127, 0.18);
            border-radius: 12px;
            padding: 1rem 1.1rem;
            background: linear-gradient(
              180deg,
              rgba(127, 127, 127, 0.06),
              rgba(127, 127, 127, 0.02)
            );
          ">
            <div style="font-size: 1.05rem; font-weight: 700; margin-bottom: 0.75rem;">
              {escape(title)}
            </div>
            <div style="
              display:grid;
              grid-template-columns:repeat(auto-fit, minmax(11rem, 1fr));
              gap:0.7rem;
            ">
              {metric_blocks}
            </div>
            {bar_blocks}
            <div style="margin-top:0.9rem;font-size:0.92rem;">
              {detail_rows}
            </div>
          </div>
        </div>
        """
    return html
