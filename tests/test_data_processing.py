import os

import pandas as pd
import pytest

from utils import Milestone
from config import AppConfiguration
from src.data_processing import Timeblock, Data


@pytest.fixture
def app_test_configuration():
    dir_path = os.path.dirname(os.path.realpath(__file__))

    app_config = AppConfiguration()
    app_config.data_path = os.path.join(dir_path, "data/test-data-1-eu.csv")
    app_config.timeframe_start = pd.Timestamp(year=2016, month=1, day=1)
    app_config.timeframe_end = pd.Timestamp(year=2026, month=1, day=1)
    app_config.day_first_dates = True
    app_config.study_label = "Study"
    app_config.compound_label = "Compound"
    app_config.unique_identity_label = "DPN(Compound-Study)"
    app_config.milestone_definitions = {
        "Milestone 1": Milestone("Milestone 1", offset_before=14, offset_after=14, active=True),
        "Milestone 2": Milestone("Milestone 2", offset_before=14, offset_after=14, active=True),
        "Milestone 3": Milestone("Milestone 3", offset_before=14, offset_after=14, active=True),
        "Milestone 4": Milestone("Milestone 4", offset_before=14, offset_after=14, active=True),
        "Milestone 5": Milestone("Milestone 5", offset_before=7, offset_after=7, active=True),
    }
    return app_config


@pytest.fixture
def app_test_session_store(app_test_configuration):
    return {
        'timeframe_start': app_test_configuration.timeframe_start.isoformat(),
        'timeframe_end': app_test_configuration.timeframe_end.isoformat(),
        'milestones': {
            "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 4": {"label": "Milestone 4", "offset_before": 14, "offset_after": 14, "active": True},
            "Milestone 5": {"label": "Milestone 5", "offset_before": 7, "offset_after": 7, "active": True},
        },
        'first_start': False
    }


@pytest.fixture
def app_test_dataclass(app_test_configuration):
    return Data(app_test_configuration)


def test_date_parsing(app_test_configuration):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    app_test_configuration.data_path = os.path.join(dir_path, "data/test-data-2-us.csv")
    app_test_configuration.day_first_dates = False

    us_data = Data(app_test_configuration)
    ambiguous_study = us_data.df.loc[us_data.df[app_test_configuration.unique_identity_label] == "gt396-204"]

    assert all(ambiguous_study['Milestone 1'] == [pd.Timestamp(year=1999, month=7, day=8)])
    assert all(ambiguous_study['Milestone 2'] == [pd.Timestamp(year=1999, month=2, day=3)])
    assert all(ambiguous_study['Milestone 3'] == [pd.Timestamp(year=2002, month=2, day=11)])
    assert all(ambiguous_study['Milestone 4'] == [pd.Timestamp(year=2002, month=3, day=11)])
    assert all(ambiguous_study['Milestone 5'] == [pd.Timestamp(year=2003, month=4, day=12)])


def test_create_timeblock_apply(app_test_dataclass, app_test_session_store):
    test_milestone_1 = Milestone('Milestone 1', offset_before=14, offset_after=14, active=True)
    test_milestone_2 = Milestone('Milestone 2', offset_before=14, offset_after=14, active=True)
    test_milestone_1_date = pd.Timestamp(year=2020, month=3, day=20)
    test_milestone_2_date = pd.Timestamp(year=2020, month=5, day=20)

    test_series = pd.Series(
        data=[test_milestone_1_date, test_milestone_2_date],
        index=[test_milestone_1.label, test_milestone_2.label]
    )

    milestones = {
        label: Milestone(**milestone_dict) for label, milestone_dict in app_test_session_store['milestones'].items()
    }

    res = app_test_dataclass.create_timeblock_apply(test_series, milestones)
    assert res, "create_timeblock_apply expected to return data!"
    assert len(res) == 2, "Exactly 2 timeblocks expected in res!"

    expected_start_1, expected_end_1 = test_milestone_1.apply_offsets(test_milestone_1_date)
    assert res[0] == Timeblock(start=expected_start_1, end=expected_end_1, milestone=test_milestone_1)

    expected_start_2, expected_end_2 = test_milestone_2.apply_offsets(test_milestone_2_date)
    assert res[1] == Timeblock(start=expected_start_2, end=expected_end_2, milestone=test_milestone_2)


def test_create_timeblock_with_active_milestones(app_test_dataclass, app_test_session_store):
    app_test_dataclass.create_timeblock(app_test_session_store)

    assert "Time Block" in app_test_dataclass.df.columns
    assert app_test_dataclass.df["Time Block"][0] == []

    for timeblock_list in app_test_dataclass.df["Time Block"][1:]:
        for timeblock in timeblock_list:
            assert isinstance(timeblock, Timeblock)


def test_create_timeblock_without_active_milestones(app_test_dataclass, app_test_session_store):
    app_test_session_store['milestones'] = {
        "Milestone 1": {"label": "Milestone 1", "offset_before": 14, "offset_after": 14, "active": False},
        "Milestone 2": {"label": "Milestone 2", "offset_before": 14, "offset_after": 14, "active": False},
        "Milestone 3": {"label": "Milestone 3", "offset_before": 14, "offset_after": 14, "active": False},
        "Milestone 4": {"label": "Milestone 4", "offset_before": 14, "offset_after": 14, "active": False},
        "Milestone 5": {"label": "Milestone 5", "offset_before": 7, "offset_after": 7, "active": False},
    }
    app_test_dataclass.create_timeblock(app_test_session_store)
    assert "Time Block" not in app_test_dataclass.df.columns


def test_create_plotting_df_apply(app_test_dataclass, app_test_session_store):
    test_df = pd.DataFrame(data={
        app_test_dataclass.study_label: "study",
        app_test_dataclass.compound_label: "compound",
        app_test_dataclass.unique_identity_label: "unique id",
        "Time Block": [
            [Timeblock(start=pd.Timestamp(year=2015, month=1, day=1), end=pd.Timestamp(year=2017, month=1, day=1),
                       milestone=Milestone("Milestone 1", offset_before=14, offset_after=14, active=True)),
             Timeblock(start=pd.Timestamp(year=2018, month=1, day=1), end=pd.Timestamp(year=2019, month=1, day=1),
                       milestone=Milestone("Milestone 2", offset_before=14, offset_after=14, active=True)),
             Timeblock(start=pd.Timestamp(year=2025, month=1, day=1), end=pd.Timestamp(year=2027, month=1, day=1),
                       milestone=Milestone("Milestone 3", offset_before=14, offset_after=14, active=True))]
        ]
    })

    timeframe_start = pd.Timestamp(app_test_session_store['timeframe_start'])
    timeframe_end = pd.Timestamp(app_test_session_store['timeframe_end'])
    plot_df = app_test_dataclass.create_plotting_df_apply(test_df, timeframe_start, timeframe_end)

    expected_df = pd.DataFrame(data={
        app_test_dataclass.unique_identity_label: ["unique id"] * 3,
        app_test_dataclass.study_label: ["study"] * 3,
        app_test_dataclass.compound_label: ["compound"] * 3,
        "start": [
            pd.Timestamp(year=2015, month=1, day=1), pd.Timestamp(year=2018, month=1, day=1),
            pd.Timestamp(year=2025, month=1, day=1)
        ],
        "end": [
            pd.Timestamp(year=2017, month=1, day=1), pd.Timestamp(year=2019, month=1, day=1),
            pd.Timestamp(year=2027, month=1, day=1)
        ],
        "type": [
            "Milestone 1", "Milestone 2", "Milestone 3"
        ],
        "inside timeframe": ["Yes", "Yes", "Yes"]
    })

    assert plot_df.equals(expected_df)


def test_create_plotting_df(app_test_dataclass, app_test_session_store):

    # Ensure plot df has been initialised as expected
    assert len(app_test_dataclass.plot_df.columns) == 7
    assert app_test_dataclass.plot_df.empty

    # No timeblocks created
    app_test_dataclass.create_plotting_df(app_test_session_store)
    assert all(app_test_dataclass.plot_df.columns == [
        app_test_dataclass.unique_identity_label, app_test_dataclass.study_label, app_test_dataclass.compound_label,
        "start", "end", "type", "inside timeframe"
    ])
    assert app_test_dataclass.plot_df.empty

    # With timeblocks created
    app_test_dataclass.create_timeblock(app_test_session_store)
    app_test_dataclass.create_plotting_df(app_test_session_store)
    assert all(app_test_dataclass.plot_df.columns == [
        app_test_dataclass.unique_identity_label, app_test_dataclass.study_label, app_test_dataclass.compound_label,
        "start", "end", "type", "inside timeframe"
    ])
    assert app_test_dataclass.plot_df.shape == (2982, 7)

    # Study without milestones is not present
    loc_1 = app_test_dataclass.plot_df.loc[app_test_dataclass.plot_df["DPN(Compound-Study)"] == "jN165-661"]
    assert loc_1.shape == (0, 7)

    # Study with all milestones is fully present, regardless of timeframe
    loc_2 = app_test_dataclass.plot_df.loc[app_test_dataclass.plot_df["DPN(Compound-Study)"] == "DN053-007"]
    assert loc_2.shape == (5, 7)
    assert all(loc_2['type'] == ["Milestone 1", "Milestone 2", "Milestone 3", "Milestone 4", "Milestone 5"])
    assert all(loc_2['inside timeframe'] == ["Yes", "Yes", "No", "No", "No"])

    # Study with partial milestones is partially present, regardless of timeframe
    loc_3 = app_test_dataclass.plot_df.loc[app_test_dataclass.plot_df["DPN(Compound-Study)"] == "jN165-061"]
    assert loc_3.shape == (1, 7)
    assert all(loc_3['type'] == ["Milestone 2"])
    assert all(loc_3['inside timeframe'] == ["No"])
