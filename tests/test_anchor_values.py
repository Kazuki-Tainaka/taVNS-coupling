import pandas as pd


ANCHORS = [
    ("BRS_seq_all", 18, -0.74, 0.0005),
    ("BRS_seq_up", 18, -0.68, 0.0005),
    ("BRS_seq_down", 15, -0.52, 0.1547),
    ("BRS_TF_mean", 12, -0.63, 0.1547),
    ("rhomax_MATLAB", 18, 1.01, 0.0046),
    ("GC_F_BP_to_RRI", 18, 0.77, 0.0368),
    ("GC_F_RRI_to_SBP", 18, 0.69, 0.1395),
    ("GC3_F_RRI_to_PTT", 16, 0.85, 0.0368),
    ("GC3_F_SBP_to_RRI", 16, 0.44, 0.2265),
    ("GC3_F_RRI_to_SBP", 16, 0.53, 0.2833),
    ("PTT_mean", 16, 0.03, 1.0),
]


def test_anchor_values():
    df = pd.read_csv("results/Additional_File_2.csv")
    for metric, n, dz, q in ANCHORS:
        row = df[df["Metric"] == metric].iloc[0]
        assert int(row["n"]) == n
        assert abs(float(row["dz_Stim_Pre"]) - dz) < 0.02
        assert abs(float(row["p_FDR_Stim_Pre"]) - q) < max(0.002, abs(q) * 0.05)


def test_temporal_type_distribution():
    df = pd.read_csv("results/Additional_File_2.csv")
    assert df["Temporal_Type"].value_counts().to_dict() == {"D": 32, "B": 8, "C": 5, "A": 1}


def test_brs_tf_stim_pre_eligibility_flag_count():
    subjects = pd.read_csv("data/subjects.csv")
    assert subjects["brs_tf_eligible"].astype(bool).sum() == 12


def test_brs_seq_down_exclusion_flags():
    subjects = pd.read_csv("data/subjects.csv")
    assert "brs_seq_down_eligible" not in subjects.columns
    excluded = set(subjects.loc[~subjects["brsseq_down_eligible"].astype(bool), "subject_id"])
    assert excluded == {"S06", "S12", "S17"}
    assert subjects["brsseq_down_eligible"].astype(bool).sum() == 15
