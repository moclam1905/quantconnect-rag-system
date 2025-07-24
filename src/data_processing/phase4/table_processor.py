"""Detect table queries & summarise Markdown tables into small DataFrame markdown."""
import re
from io import StringIO

import pandas as pd

_TABLE_REGEX = re.compile(r"\b(bảng|table|so sánh|compare|columns?)\b", re.I)


def detect_table_query(query: str, context_text: str) -> bool:
    if not _TABLE_REGEX.search(query):
        return False
    if context_text.count("|") >= 3:
        return True
    return False


def summarise_markdown_table(md: str, max_cols: int = 3, max_rows: int = 10) -> str:
    """Return a reduced Markdown table."""
    try:
        # convert md -> html -> df via pandas
        df_list = pd.read_html(StringIO(md), flavor="bs4")
        if not df_list:
            return ""
        df = df_list[0]
        # drop columns with low variance
        for col in df.columns.tolist():
            if df[col].nunique() <= 1:
                df.drop(columns=col, inplace=True)
        if len(df.columns) > max_cols:
            df = df.iloc[:, :max_cols]
        df = df.head(max_rows)
        return df.to_markdown(index=False)
    except Exception:
        return ""


