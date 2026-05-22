# Figure Regeneration

Run all figure scripts with:

```bash
python figures/regenerate_all.py
```

Each script writes a PNG to `figures/outputs/`. The corrected v1.0.1 figure
scripts are the authoritative JNER submission versions, with their external
data dependencies copied into `figures/data/` and canonical summary tables read
from `data/reference/`.

| Figure | Script | Main dependency |
|---|---|---|
| Fig1 | `figures/main/generate_fig1.py` | bundled coupling/coherence subject tables + `data/reference/Additional_File_2.csv` |
| Fig2 | `figures/main/generate_fig2.py` | bundled GC3 subject/statistics tables + `data/reference/Additional_File_2.csv` |
| Fig3 | `figures/main/generate_fig3.py` | bundled fixed-lag, rhomax, PDI tables |
| Fig4 | `figures/main/generate_fig4.py` | bundled supplementary coupling/HRV tables and subject table |
| FigS1 | `figures/supplementary/generate_figS1.py` | `data/reference/Additional_File_2.csv` |
| FigS2 | `figures/supplementary/generate_figS2.py` | bundled HRV type classification + `data/reference/Additional_File_3.csv` |
| FigS3 | `figures/supplementary/generate_figS3.py` | bundled WTC group-average matrices |
| FigS4 | `figures/supplementary/generate_figS4.py` | bundled fixed-lag profiles |
| FigS5 | `figures/supplementary/generate_figS5.py` | bundled segmented rhomax time-series and coefficient tables |
| FigS6 | `figures/supplementary/generate_figS6.py` | bundled tracking group and individual time-series tables |
| FigS7 | `figures/supplementary/generate_figS7.py` | bundled coupling subject table and supplementary coupling table |
