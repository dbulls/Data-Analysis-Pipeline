"""Automated exploratory analysis pipeline.

Stages (one module each):
    loading        read raw CSV files as text, never modify them
    infer          decide each column's type from its values
    profile        describe every table and column
    relationships  find shared keys between tables and build joined views
    clean          apply conservative, logged cleaning rules
    measures       one statistical measurement per kind of pattern
    search         enumerate candidate findings across all columns
    filters        multiple-comparison correction, robustness checks, ranking
    wording        plain-language statements for each finding
    charts         one chart per accepted finding
    report         Markdown and HTML findings report
    pipeline       runs the stages in order and saves every intermediate result
"""
