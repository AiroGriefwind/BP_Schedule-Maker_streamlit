import datetime as dt
from typing import Any, Dict

import pandas as pd
import streamlit as st


# --- Compatibility helpers ---
def _df_elementwise(df: pd.DataFrame, func):
    """
    Pandas deprecated DataFrame.applymap in favor of DataFrame.map (elementwise).
    Keep a small shim for compatibility across pandas versions.
    """
    if hasattr(df, "map"):
        return df.map(func)  # pandas >= 2.1
    return df.applymap(func)  # pragma: no cover


def _excel_rgb_to_hex6(rgb):
    """
    Convert openpyxl rgb/argb like 'FFD9D9D9' to 'D9D9D9' (hex6).
    Returns '' if invalid/empty.
    """
    if rgb is None:
        return ""
    s = str(rgb).strip()
    if not s or s.lower() in {"none", "nan"}:
        return ""
    if len(s) == 8:  # ARGB -> RGB
        s = s[2:]
    if len(s) != 6:
        return ""
    # Validate hex
    try:
        int(s, 16)
    except Exception:
        return ""
    return s.upper()


def _auto_text_hex_from_bg(bg_hex6):
    """
    Pick black/white text for readability based on background brightness.
    bg_hex6: 'RRGGBB'
    """
    if not bg_hex6 or len(bg_hex6) != 6:
        return ""
    r = int(bg_hex6[0:2], 16)
    g = int(bg_hex6[2:4], 16)
    b = int(bg_hex6[4:6], 16)
    # Perceived luminance (sRGB)
    luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
    return "FFFFFF" if luminance < 140 else "000000"


def _excel_rgb_to_css(rgb):
    """
    openpyxl fill.fgColor.rgb is often ARGB like 'FFD9D9D9'. Convert to CSS.
    Returns '' when no color should be applied.
    """
    hex6 = _excel_rgb_to_hex6(rgb)
    # Treat "no fill" / default as empty (openpyxl sometimes gives 00000000)
    if not hex6 or hex6 in {"000000"}:
        return ""
    return f"background-color: #{hex6};"


def _availability_cell_value(cell):
    """Extract display/edit value from availability cell."""
    v = cell
    if isinstance(cell, dict) and "value" in cell:
        v = cell.get("value")
    if v is None:
        return ""
    # If upstream stored an Excel-auto-parsed date for a shift like "10-19", show as "M-D".
    try:
        if isinstance(v, pd.Timestamp):
            if not pd.isna(v):
                dtv = v.to_pydatetime()
                if dtv.hour == 0 and dtv.minute == 0 and dtv.second == 0:
                    return f"{dtv.month}-{dtv.day}"
        if isinstance(v, dt.datetime):
            if v.hour == 0 and v.minute == 0 and v.second == 0:
                return f"{v.month}-{v.day}"
    except Exception:
        pass
    if isinstance(v, list):
        return ", ".join(map(str, v))
    return str(v)


def availability_cell_css(cell):
    """Extract CSS style string from availability cell (background + text color)."""
    if isinstance(cell, dict):
        bg_hex6 = _excel_rgb_to_hex6(cell.get("color"))
        font_hex6 = _excel_rgb_to_hex6(cell.get("font_color"))

        styles = []

        bg_css = _excel_rgb_to_css(cell.get("color"))
        if bg_css:
            styles.append(bg_css)

        # Prefer explicit font color; if missing, auto based on bg.
        chosen_font = font_hex6
        if bg_hex6 and (not chosen_font):
            chosen_font = _auto_text_hex_from_bg(bg_hex6)
        # If explicit font equals bg (unreadable), fall back to auto.
        if bg_hex6 and chosen_font and chosen_font == bg_hex6:
            chosen_font = _auto_text_hex_from_bg(bg_hex6)

        if chosen_font:
            styles.append(f"color: #{chosen_font};")

        return " ".join(styles)
    return ""


# Converts availability dict (with nested value/color) to a simple df for display
def flatten_availability_for_display(availability):
    records = []
    for date, emps in availability.items():
        row = {"Date": date}
        for emp, cell in emps.items():
            val = cell["value"]
            row[emp] = val
        records.append(row)
    return pd.DataFrame(records)


# If you want to visualize color in Streamlit, apply background formatting
def get_cell_styles(availability):
    # Returns a DataFrame of styles (you could use .style.apply in display, but Streamlit might need custom rendering)
    styles = []
    for date, emps in availability.items():
        row = {}
        for emp, cell in emps.items():
            color = cell.get("color")
            row[emp] = f"background-color: #{color[2:]}" if color else ""
        styles.append(row)
    return pd.DataFrame(styles)


def merge_edited_df_with_color(edited_df, orig_availability):
    """
    Given a DataFrame of "Date", Emp1, Emp2... (all cells are pure values, no color shown),
    and the original nested-availability dict with color+value,
    return a new nested dict with new values but colors preserved/retained.
    """
    if edited_df is None or edited_df.empty:
        return orig_availability or {}

    # Defensive: if "Date" column exists, use it as index
    df = edited_df.copy()
    if "Date" in df.columns:
        df = df.set_index("Date")

    new_avail = {}
    for date_key, row in df.iterrows():
        d = str(date_key)
        new_avail[d] = {}
        for emp in df.columns:
            new_val = row[emp]
            if isinstance(new_val, float) and pd.isna(new_val):
                new_val = ""
            if new_val is None:
                new_val = ""
            if not isinstance(new_val, str):
                new_val = str(new_val)
            new_val = new_val.strip()

            orig_cell = (orig_availability or {}).get(d, {}).get(emp)
            color = orig_cell.get("color") if isinstance(orig_cell, dict) else None
            font_color = orig_cell.get("font_color") if isinstance(orig_cell, dict) else None
            new_avail[d][emp] = {"value": new_val, "color": color, "font_color": font_color}

    return new_avail


def availability_to_dataframe():
    """Convert availability (dict) to DataFrame for display/edit."""
    avail = st.session_state.get("availability") or {}
    df = pd.DataFrame.from_dict(avail, orient="index")
    if not df.empty:
        df = _df_elementwise(df, _availability_cell_value)
    return df


def availability_to_color_css_dataframe():
    """Build a DataFrame of CSS strings (background colors) aligned to availability_to_dataframe()."""
    avail = st.session_state.get("availability") or {}
    df = pd.DataFrame.from_dict(avail, orient="index")
    if df.empty:
        return df
    return _df_elementwise(df, availability_cell_css)


def convert_availability_dates_to_str(availability):
    def conv(obj):
        if isinstance(obj, dict):
            return {conv_key(k): conv(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [conv(x) for x in obj]
        else:
            try:
                import numpy as np  # type: ignore
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                if isinstance(obj, np.datetime64):
                    # Convert to ISO string
                    return str(obj)
            except Exception:
                pass

            return obj

    def conv_key(k):
        # Most keys represent a date row; normalize to YYYY-MM-DD.
        try:
            if isinstance(k, pd.Timestamp):
                if pd.isna(k):
                    return "NaT"
                return k.to_pydatetime().strftime("%Y-%m-%d")
        except Exception:
            pass

        if isinstance(k, dt.datetime) or isinstance(k, dt.date):
            return k.strftime("%Y-%m-%d")
        return str(k)

    return conv(availability)


def dataframe_to_availability(edited_df):
    """Converts the edited DataFrame back to the availability dictionary format."""
    # Keep original structure (value+color) if present; update value only.
    orig = st.session_state.get("availability") or {}
    new_avail = {}
    for date_key, row in edited_df.iterrows():
        d = str(date_key)
        new_avail[d] = {}
        for emp in edited_df.columns:
            new_val = row[emp]
            if isinstance(new_val, float) and pd.isna(new_val):
                new_val = ""
            if new_val is None:
                new_val = ""
            if not isinstance(new_val, str):
                new_val = str(new_val)
            new_val = new_val.strip()

            orig_cell = orig.get(d, {}).get(emp)
            color = orig_cell.get("color") if isinstance(orig_cell, dict) else None
            font_color = orig_cell.get("font_color") if isinstance(orig_cell, dict) else None
            new_avail[d][emp] = {"value": new_val, "color": color, "font_color": font_color}

    st.session_state.availability = new_avail
