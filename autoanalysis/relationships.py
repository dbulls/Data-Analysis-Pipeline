"""Find shared keys between tables and build joined views for analysis."""

SEPARATOR = "."   # joined columns are named "<parent table>.<column>"


def detect_relationships(tables, types, config):
    """Find columns whose values are (almost) all present in another table's unique key.

    Returns a list of {child_table, child_column, parent_table, parent_column, containment}.
    """
    # Candidate parent keys: columns with no repeats and no gaps.
    parent_keys = []
    for table, df in tables.items():
        for column in df.columns:
            if types[table][column] in ("identifier", "categorical", "text", "numeric"):
                values = df[column].dropna()
                if len(values) >= 2 and values.is_unique and len(values) == len(df):
                    parent_keys.append((table, column, set(values.astype(str))))

    relationships = []
    for child_table, df in tables.items():
        for column in df.columns:
            if types[child_table][column] not in ("identifier", "categorical", "text"):
                continue
            child_values = df[column].dropna().astype(str)
            if child_values.empty:
                continue
            best = None
            for parent_table, parent_column, key_values in parent_keys:
                if parent_table == child_table:
                    continue
                containment = child_values.isin(key_values).mean()
                same_name = column.lower() == parent_column.lower()
                if containment >= config.relationship_min_containment and (same_name or containment == 1.0):
                    rank = (same_name, containment)
                    if best is None or rank > best[0]:
                        best = (rank, parent_table, parent_column, containment)
            if best:
                relationships.append({
                    "child_table": child_table, "child_column": column,
                    "parent_table": best[1], "parent_column": best[2],
                    "containment": float(best[3]),
                    "orphan_rows": int((~child_values.isin(set(tables[best[1]][best[2]].astype(str)))).sum()),
                })
    return relationships


def build_views(tables, types, relationships):
    """Left-join each parent's descriptive columns onto its child table.

    The parent key is unique, so the child's row count never changes.
    Returns (views, view_types, joined_columns) where joined_columns[table]
    maps each column that came from a parent table to the child key column
    it was joined through (e.g. {"customers.segment": "customer_id"}).
    """
    views, view_types, joined_columns = {}, {}, {}
    for table, df in tables.items():
        view = df.copy()
        column_types = dict(types[table])
        joined = {}
        for rel in relationships:
            if rel["child_table"] != table:
                continue
            parent = tables[rel["parent_table"]]
            parent_types = types[rel["parent_table"]]
            keep = [c for c in parent.columns
                    if c != rel["parent_column"]
                    and parent_types[c] in ("numeric", "boolean", "categorical")]
            if not keep:
                continue
            renamed = {c: f"{rel['parent_table']}{SEPARATOR}{c}" for c in keep}
            lookup = parent.set_index(parent[rel["parent_column"]].astype(str))[keep].rename(columns=renamed)
            view = view.join(lookup, on=view[rel["child_column"]].astype(str).rename("_key"))
            view = view.drop(columns=["_key"], errors="ignore")
            for original, new_name in renamed.items():
                column_types[new_name] = parent_types[original]
                joined[new_name] = rel["child_column"]
        views[table], view_types[table], joined_columns[table] = view, column_types, joined
    return views, view_types, joined_columns
