"""Readable names for columns and tables, and the sentences built from them."""

from autoanalysis.labels import pretty_column, pretty_table, title_column
from autoanalysis.wording import statement


def test_plain_names_lose_underscores_and_keep_acronyms():
    assert pretty_column("study_hours_per_week") == "study hours per week"
    assert pretty_column("trinidadian_dollar_to_usd") == "trinidadian dollar to USD"
    assert title_column("exam_score") == "Exam score"


def test_change_series_and_joined_columns_are_explained():
    assert pretty_column("eur_to_usd_pct_change") == "EUR to USD (% change)"
    assert pretty_column("customers.segment") == "segment (from customers)"
    assert pretty_table("exchange_rate_to_usd__changes") == "exchange rate to USD (record-to-record changes)"


def test_config_labels_override_by_table_column_or_column():
    labels = {"orders.amount": "order value", "segment": "customer segment"}
    assert pretty_column("amount", labels, table="orders") == "order value"
    assert pretty_column("amount", labels, table="refunds") == "amount"
    assert pretty_column("customers.segment", labels) == "customer segment (from customers)"


def test_sentences_use_readable_names_but_ids_stay_raw(run):
    shown = [statement(f) for f in run["accepted"]]
    assert shown and all("_" not in text for text in shown)
    assert all("|" in f["id"] for f in run["accepted"])      # ids are the raw audit keys
