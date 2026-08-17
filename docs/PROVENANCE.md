# Release provenance

- Supplementary Data 1–3 originate from the author-designated manuscript package. Public filenames and non-scientific provenance-path metadata were normalized; numerical and statistical content was not changed.
- `figures/supplementary_figure_s3_brs_specification_landscape.jpg` is the author-finished four-panel S3 artwork.
- `scripts/generate_supplementary_figure_s3.py` is the data-driven S3 generator. Its values and panel semantics are audited against public Supplementary Data 3.
- `src/native_paired_beat_var.py` is the candidate-order 1–10 native paired-beat VAR implementation.
- `scripts/generate_summary_figures.py` is the provenance-matched S5 generator.
- `src/methods_text_matched_brs/core.py` preserves the output-generating correlation-only reference branch and the A0/A1 MAX/OVERLAP comparators.
- `src/nonlinear_sensitivity/` preserves the frozen six-primary-test targeted sensitivity pipeline. Public aggregate output confirms that no primary result survives BH correction.

Path-resolution changes are limited to `TAVNS_DATA_ROOT`, safe output roots, clean public filenames, and non-scientific metadata. Scientific estimators, thresholds, lags, sequence enumeration, coherence parameters, model-order range, and seeds are unchanged.
