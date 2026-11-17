"""Readable names for columns and tables.

Column names in the data are written for computers (`trinidadian_dollar_to_usd_pct_change`,
`customers.segment`). Everything a person reads (sentences, chart labels, the app)
goes through these functions instead. They only change how a name is shown:
candidate ids and file names keep the raw names, so results stay traceable.

To override a label for one column, set `labels` in the config:
    {"labels": {"orders.amount": "order value", "segment": "customer segment"}}
Keys are `table.column` or just `column`.
"""

CHANGE_SUFFIX = "_pct_change"       # added by timeseries.py to a column's % change series
CHANGE_TABLE_SUFFIX = "__changes"   # added by timeseries.py to the table holding them
JOIN_SEPARATOR = "."                # `parent_table.column` for columns joined from a parent table

# Words that read better in capitals.
ACRONYMS = {"usd", "eur", "gbp", "jpy", "id", "ids", "url", "gdp", "cpi", "gpa", "kpi", "ip", "sku", "vat", "uk", "us", "uae", "eu"}


def words(name):
    """'exam_score' -> 'exam score', with known acronyms in capitals."""
    parts = str(name).replace("_", " ").replace("-", " ").split()
    return " ".join(part.upper() if part.lower() in ACRONYMS else part for part in parts)


def capitalize_first(text):
    """Capitalise the first letter of a sentence, leaving the rest as it is."""
    return text[:1].upper() + text[1:] if text else text


def pretty_table(table, labels=None):
    """'exchange_rate_to_usd__changes' -> 'exchange rate to USD (record-to-record changes)'."""
    labels = labels or {}
    if table in labels:
        return labels[table]
    if table.endswith(CHANGE_TABLE_SUFFIX):
        return f"{pretty_table(table.removesuffix(CHANGE_TABLE_SUFFIX), labels)} (record-to-record changes)"
    return words(table)


def pretty_column(name, labels=None, table=None):
    """A column name as a person would say it, lower case apart from acronyms.

    'study_hours_per_week'    -> 'study hours per week'
    'customers.segment'       -> 'segment (from customers)'
    'eur_to_usd_pct_change'   -> 'EUR to USD (% change)'
    """
    labels = labels or {}
    for key in (f"{table}.{name}" if table else None, name):
        if key in labels:
            return labels[key]
    name = str(name)
    if name.endswith(CHANGE_SUFFIX):
        return f"{pretty_column(name.removesuffix(CHANGE_SUFFIX), labels, table)} (% change)"
    if JOIN_SEPARATOR in name:
        parent, column = name.split(JOIN_SEPARATOR, 1)
        return f"{pretty_column(column, labels, parent)} (from {pretty_table(parent, labels)})"
    return words(name)


def title_column(name, labels=None, table=None):
    """Same as pretty_column, starting with a capital: for headings and table cells."""
    return capitalize_first(pretty_column(name, labels, table))
