"""The app's loader: a finished analysis can be read back and any finding charted."""

import matplotlib.pyplot as plt

from autoanalysis.results import chart_figure, list_runs, load_config, load_run, load_views


def test_a_finished_run_can_be_read_back(output_dir, run):
    assert load_run(output_dir)["manifest"]["counts"]["accepted"] == len(run["accepted"])
    assert output_dir.name in list_runs(output_dir.parent)
    config = load_config(load_run(output_dir)["manifest"])
    assert config.effect_thresholds["trend_mean"] == run["config"].effect_thresholds["trend_mean"]


def test_every_accepted_finding_can_be_charted_from_saved_views(output_dir, run):
    saved = load_run(output_dir)
    views = load_views(output_dir)
    assert saved["accepted"], "the synthetic data should produce findings"
    for finding in saved["accepted"]:          # including the ones outside the top list
        fig = chart_figure(finding, views, saved["config"])
        assert fig.axes
        plt.close(fig)
