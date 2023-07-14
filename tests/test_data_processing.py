import os

import pandas as pd
import pytest

from config import AppConfiguration
from src.data_processing import Timeblock, MergedTimeblock, Data, ItemInformation, ItemStatus, Gap
from test_app import load_json_test_dataframe
from utils import Milestone
from datetime import date


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
    app_config.day_weight_coefficient = 5
    app_config.gap_weight_coefficient = 0.1
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


@pytest.mark.parametrize(
    "file_path, day_first_dates",
    [
        (
            "data/test-data-1-eu.csv",
            True
        ),
        (
            "data/test-data-2-us.csv",
            False
        ),
        (
            "data/test-data-3-iso.csv",
            True
        ),
        (
            "data/test-data-1-eu.xlsx",
            True
        ),
        (
            "data/test-data-2-us.xlsx",
            False
        ),
        (
            "data/test-data-3-iso.xlsx",
            True
        )
    ]
)
def test_date_parsing(file_path, day_first_dates, app_test_configuration):


    dir_path = os.path.dirname(os.path.realpath(__file__))
    app_test_configuration.data_path = os.path.join(dir_path, file_path)
    app_test_configuration.day_first_dates = day_first_dates

    data_ = Data(app_test_configuration)
    ambiguous_study = data_.df.loc[data_.df[app_test_configuration.unique_identity_label] == "gt396-204"]

    assert ambiguous_study.iloc[0]['Milestone 1'] == pd.Timestamp(year=1999, month=8, day=17)
    assert ambiguous_study.iloc[0]['Milestone 2'] == pd.Timestamp(year=1999, month=3, day=7)
    assert ambiguous_study.iloc[0]['Milestone 3'] == pd.Timestamp(year=2002, month=3, day=13)
    assert ambiguous_study.iloc[0]['Milestone 4'] == pd.Timestamp(year=2002, month=12, day=13)
    assert ambiguous_study.iloc[0]['Milestone 5'] == pd.Timestamp(year=2003, month=6, day=10)


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

    res = app_test_dataclass._create_timeblock_apply(test_series, milestones)
    assert res, "create_timeblock_apply expected to return data!"
    assert len(res) == 2, "Exactly 2 timeblocks expected in res!"

    expected_start_1, expected_end_1 = test_milestone_1.apply_offsets(test_milestone_1_date)
    assert res[0] == Timeblock(start=expected_start_1, end=expected_end_1, milestone=test_milestone_1,
                               timestamp=test_milestone_1_date)

    expected_start_2, expected_end_2 = test_milestone_2.apply_offsets(test_milestone_2_date)
    assert res[1] == Timeblock(start=expected_start_2, end=expected_end_2, milestone=test_milestone_2,
                               timestamp=test_milestone_2_date)


def test_create_timeblock_with_active_milestones(app_test_dataclass, app_test_session_store):

    timeblock_df = app_test_dataclass.create_timeblock(app_test_session_store)

    assert "Time Block" in timeblock_df.columns

    # Ensure study with no milestones is represented as an empty list
    assert timeblock_df["Time Block"][0] == []

    for timeblock_list in timeblock_df["Time Block"][1:]:
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
            [
                Timeblock(start=pd.Timestamp(year=2015, month=1, day=1), end=pd.Timestamp(year=2017, month=1, day=1),
                          milestone=Milestone("Milestone 1", offset_before=366, offset_after=366, active=True),
                          timestamp=pd.Timestamp(2016, month=1, day=1)),
                Timeblock(start=pd.Timestamp(year=2018, month=1, day=1), end=pd.Timestamp(year=2019, month=1, day=1),
                          milestone=Milestone("Milestone 2", offset_before=183, offset_after=183, active=True),
                          timestamp=pd.Timestamp(2018, month=7, day=3)),
                Timeblock(start=pd.Timestamp(year=2025, month=1, day=1), end=pd.Timestamp(year=2027, month=1, day=1),
                          milestone=Milestone("Milestone 3", offset_before=366, offset_after=366, active=True),
                          timestamp=pd.Timestamp(2026, month=1, day=1))
            ]
        ]
    })

    timeframe_start = pd.Timestamp(app_test_session_store['timeframe_start'])
    timeframe_end = pd.Timestamp(app_test_session_store['timeframe_end'])
    plot_df = app_test_dataclass._create_plotting_df_apply(test_df, timeframe_start, timeframe_end)

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

    # No timeblocks created
    plot_df_1 = app_test_dataclass.create_plotting_df(app_test_dataclass.df, app_test_session_store)
    assert not list(plot_df_1.columns)
    assert plot_df_1.empty

    # With timeblocks created
    tb_df = app_test_dataclass.create_timeblock(app_test_session_store)
    plot_df_2 = app_test_dataclass.create_plotting_df(tb_df, app_test_session_store)
    assert all(plot_df_2.columns == [
        app_test_dataclass.unique_identity_label, app_test_dataclass.study_label, app_test_dataclass.compound_label,
        "start", "end", "type", "inside timeframe"
    ])
    assert plot_df_2.shape == (2982, 7)

    # Study without milestones is not present
    loc_1 = plot_df_2.loc[plot_df_2["DPN(Compound-Study)"] == "jN165-661"]
    assert loc_1.shape == (0, 7)

    # Study with all milestones is fully present, regardless of timeframe
    loc_2 = plot_df_2.loc[plot_df_2["DPN(Compound-Study)"] == "DN053-007"]
    assert loc_2.shape == (5, 7)
    assert all(loc_2['type'] == ["Milestone 1", "Milestone 2", "Milestone 3", "Milestone 4", "Milestone 5"])
    assert all(loc_2['inside timeframe'] == ["Yes", "Yes", "No", "No", "No"])

    # Study with partial milestones is partially present, regardless of timeframe
    loc_3 = plot_df_2.loc[plot_df_2["DPN(Compound-Study)"] == "jN165-061"]
    assert loc_3.shape == (1, 7)
    assert all(loc_3['type'] == ["Milestone 2"])
    assert all(loc_3['inside timeframe'] == ["No"])


@pytest.mark.parametrize(
    "timeblock_lst, expected_output",
    [
        # No timeblocks
        (
                [],
                []
        ),
        # Single timeblock
        (
                [
                    Timeblock(start=pd.Timestamp(year=2022, month=5, day=20),
                              end=pd.Timestamp(year=2022, month=6, day=17),
                              milestone=Milestone("Milestone 1", offset_before=14, offset_after=14, active=True),
                              timestamp=pd.Timestamp(year=2022, month=6, day=3))
                ],
                [
                    Timeblock(start=pd.Timestamp(year=2022, month=5, day=20),
                              end=pd.Timestamp(year=2022, month=6, day=17),
                              milestone=Milestone("Milestone 1", offset_before=14, offset_after=14, active=True),
                              timestamp=pd.Timestamp(year=2022, month=6, day=3))
                ]
        ),
        # Two timeblocks with slight overlap
        (
                [
                    Timeblock(start=pd.Timestamp(year=2022, month=5, day=20),
                              end=pd.Timestamp(year=2022, month=6, day=17),
                              milestone=Milestone("Milestone 1", offset_before=14, offset_after=14, active=True),
                              timestamp=pd.Timestamp(year=2022, month=6, day=3)),
                    Timeblock(start=pd.Timestamp(year=2022, month=5, day=30),
                              end=pd.Timestamp(year=2022, month=6, day=23),
                              milestone=Milestone("Milestone 2", offset_before=14, offset_after=10, active=True),
                              timestamp=pd.Timestamp(year=2022, month=6, day=13))
                ],
                [
                    MergedTimeblock(
                        start=pd.Timestamp(year=2022, month=5, day=20),
                        end=pd.Timestamp(year=2022, month=6, day=23),
                        milestones=[
                            Milestone("Milestone 1", offset_before=14, offset_after=14, active=True),
                            Milestone("Milestone 2", offset_before=14, offset_after=10, active=True)
                        ],
                        timestamps=[
                            pd.Timestamp(year=2022, month=6, day=3),
                            pd.Timestamp(year=2022, month=6, day=13)
                        ]
                    )
                ]
        ),
        # Two timeblocks with no overlap
        (
                [
                    Timeblock(start=pd.Timestamp(year=2022, month=5, day=20),
                              end=pd.Timestamp(year=2022, month=6, day=17),
                              milestone=Milestone("Milestone 1", offset_before=14, offset_after=14, active=True),
                              timestamp=pd.Timestamp(year=2022, month=6, day=3)),
                    Timeblock(start=pd.Timestamp(year=2022, month=6, day=30),
                              end=pd.Timestamp(year=2022, month=7, day=24),
                              milestone=Milestone("Milestone 2", offset_before=14, offset_after=10, active=True),
                              timestamp=pd.Timestamp(year=2022, month=7, day=14))
                ],
                [
                    Timeblock(start=pd.Timestamp(year=2022, month=5, day=20),
                              end=pd.Timestamp(year=2022, month=6, day=17),
                              milestone=Milestone("Milestone 1", offset_before=14, offset_after=14, active=True),
                              timestamp=pd.Timestamp(year=2022, month=6, day=3)),
                    Timeblock(start=pd.Timestamp(year=2022, month=6, day=30),
                              end=pd.Timestamp(year=2022, month=7, day=24),
                              milestone=Milestone("Milestone 2", offset_before=14, offset_after=10, active=True),
                              timestamp=pd.Timestamp(year=2022, month=7, day=14))
                ]
        ),
        # Three timeblocks where the middle overlaps the first and last
        (
                [
                    Timeblock(start=pd.Timestamp(year=2022, month=5, day=20),
                              end=pd.Timestamp(year=2022, month=6, day=17),
                              milestone=Milestone("Milestone 1", offset_before=14, offset_after=14, active=True),
                              timestamp=pd.Timestamp(year=2022, month=6, day=3)),
                    Timeblock(start=pd.Timestamp(year=2022, month=5, day=30),
                              end=pd.Timestamp(year=2022, month=6, day=23),
                              milestone=Milestone("Milestone 2", offset_before=14, offset_after=10, active=True),
                              timestamp=pd.Timestamp(year=2022, month=6, day=13)),
                    Timeblock(start=pd.Timestamp(year=2022, month=6, day=20),
                              end=pd.Timestamp(year=2022, month=7, day=14),
                              milestone=Milestone("Milestone 3", offset_before=10, offset_after=14, active=True),
                              timestamp=pd.Timestamp(year=2022, month=6, day=30))
                ],
                [
                    MergedTimeblock(
                        start=pd.Timestamp(year=2022, month=5, day=20),
                        end=pd.Timestamp(year=2022, month=7, day=14),
                        milestones=[
                            Milestone("Milestone 1", offset_before=14, offset_after=14, active=True),
                            Milestone("Milestone 2", offset_before=14, offset_after=10, active=True),
                            Milestone("Milestone 3", offset_before=10, offset_after=14, active=True)
                        ],
                        timestamps=[
                            pd.Timestamp(year=2022, month=6, day=3),
                            pd.Timestamp(year=2022, month=6, day=13),
                            pd.Timestamp(year=2022, month=6, day=30)
                        ]
                    )
                ]
        ),
        # Three timeblocks where the first two overlap last
        (
                [
                    Timeblock(start=pd.Timestamp(year=2022, month=5, day=20),
                              end=pd.Timestamp(year=2022, month=6, day=17),
                              milestone=Milestone("Milestone 1", offset_before=14, offset_after=14, active=True),
                              timestamp=pd.Timestamp(year=2022, month=6, day=3)),
                    Timeblock(start=pd.Timestamp(year=2022, month=5, day=30),
                              end=pd.Timestamp(year=2022, month=6, day=23),
                              timestamp=pd.Timestamp(year=2022, month=6, day=13),
                              milestone=Milestone("Milestone 2", offset_before=14, offset_after=10, active=True)),
                    Timeblock(start=pd.Timestamp(year=2022, month=6, day=5),
                              end=pd.Timestamp(year=2022, month=7, day=29),
                              milestone=Milestone("Milestone 3", offset_before=10, offset_after=14, active=True),
                              timestamp=pd.Timestamp(year=2022, month=7, day=15))
                ],
                [
                    MergedTimeblock(
                        start=pd.Timestamp(year=2022, month=5, day=20),
                        end=pd.Timestamp(year=2022, month=7, day=29),
                        milestones=[
                            Milestone("Milestone 1", offset_before=14, offset_after=14, active=True),
                            Milestone("Milestone 2", offset_before=14, offset_after=10, active=True),
                            Milestone("Milestone 3", offset_before=10, offset_after=14, active=True)
                        ],
                        timestamps=[
                            pd.Timestamp(year=2022, month=6, day=3),
                            pd.Timestamp(year=2022, month=6, day=13),
                            pd.Timestamp(year=2022, month=7, day=15)
                        ]
                    )
                ]
        ),
        # Timeblock Fully Encompassed By Another
        (
                [
                    Timeblock(start=pd.Timestamp(year=2022, month=6, day=20),
                              end=pd.Timestamp(year=2022, month=8, day=30),
                              milestone=Milestone("Milestone 1", offset_before=20, offset_after=20, active=True),
                              timestamp=pd.Timestamp(year=2022, month=8, day=10)),
                    Timeblock(start=pd.Timestamp(year=2022, month=6, day=25),
                              end=pd.Timestamp(year=2022, month=7, day=15),
                              milestone=Milestone("Milestone 2", offset_before=10, offset_after=10, active=True),
                              timestamp=pd.Timestamp(year=2022, month=7, day=5))
                ],
                [
                    MergedTimeblock(
                        start=pd.Timestamp(year=2022, month=6, day=20),
                        end=pd.Timestamp(year=2022, month=8, day=30),
                        milestones=[
                            Milestone("Milestone 1", offset_before=20, offset_after=20, active=True),
                            Milestone("Milestone 2", offset_before=10, offset_after=10, active=True)
                        ],
                        timestamps=[
                            pd.Timestamp(year=2022, month=8, day=10),
                            pd.Timestamp(year=2022, month=7, day=5)
                        ]
                    )
                ]
        ),
    ]
)
def test_merge_timeblock_apply(timeblock_lst, expected_output, app_test_dataclass):

    assert expected_output == app_test_dataclass._merge_timeblock_apply(timeblock_lst)


def test_merge_timeblocks(app_test_dataclass):

    test_df_no_timeblocks = pd.DataFrame({
        'Study_id': [43, 123]
    })

    test_df_with_timeblocks = pd.DataFrame({
        'Study_id': [43, 123],
        'Time Block': [
            [],
            [
                Timeblock(start=pd.Timestamp(year=2022, month=6, day=20), end=pd.Timestamp(year=2022, month=7, day=10),
                          milestone=Milestone("Milestone 1", offset_before=10, offset_after=10, active=True),
                          timestamp=pd.Timestamp(year=2022, month=6, day=30)),
                Timeblock(start=pd.Timestamp(year=2022, month=6, day=25), end=pd.Timestamp(year=2022, month=7, day=15),
                          milestone=Milestone("Milestone 2", offset_before=10, offset_after=10, active=True),
                          timestamp=pd.Timestamp(year=2022, month=7, day=5))
            ]
        ]
    })

    assert test_df_no_timeblocks.equals(app_test_dataclass.merge_timeblocks(test_df_no_timeblocks))
    assert 'Merged Time Block' in app_test_dataclass.merge_timeblocks(test_df_with_timeblocks).columns


@pytest.mark.parametrize(
    "merged_timeblock_list, expected_item_information",
    [
        # Empty merged timeblock list
        (
                [],
                ItemInformation(
                    study_id="test_study",
                    gap_number=1,
                    gap_day_total=3654,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=1),
                            end=pd.Timestamp(year=2026, month=1, day=1),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=1),
                                    end=pd.Timestamp(year=2026, month=1, day=1)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=False,
                    status=ItemStatus.NO_STATUS_GIVEN
                )
        ),
        # Fully after timeframe
        (
                [
                    Timeblock(
                        start=pd.Timestamp(year=2028, month=2, day=13),
                        end=pd.Timestamp(year=2028, month=2, day=23),
                        milestone=Milestone("Milestone 1", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2028, month=2, day=18)
                    )
                ],
                ItemInformation(
                    study_id="test_study",
                    gap_number=1,
                    gap_day_total=3654,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=1),
                            end=pd.Timestamp(year=2026, month=1, day=1),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=1),
                                    end=pd.Timestamp(year=2026, month=1, day=1)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=False,
                    status=ItemStatus.STARTING_AFTER_TIMEFRAME
                )
        ),
        # Fully before timeframe
        (
                [
                    Timeblock(
                        start=pd.Timestamp(year=2008, month=2, day=13),
                        end=pd.Timestamp(year=2008, month=2, day=23),
                        milestone=Milestone("Milestone 1", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2008, month=2, day=18)
                    )
                ],
                ItemInformation(
                    study_id="test_study",
                    gap_number=1,
                    gap_day_total=3654,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=1),
                            end=pd.Timestamp(year=2026, month=1, day=1),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=1),
                                    end=pd.Timestamp(year=2026, month=1, day=1)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=False,
                    status=ItemStatus.CLOSING_BEFORE_TIMEFRAME
                )
        ),
        # Inside And Before Timeframe
        (
                [
                    Timeblock(
                        start=pd.Timestamp(year=2008, month=2, day=13),
                        end=pd.Timestamp(year=2008, month=2, day=23),
                        milestone=Milestone("Milestone 1", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2008, month=2, day=18)
                    ),
                    Timeblock(
                        start=pd.Timestamp(year=2018, month=2, day=13),
                        end=pd.Timestamp(year=2018, month=2, day=23),
                        milestone=Milestone("Milestone 2", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2018, month=2, day=18)
                    )
                ],
                ItemInformation(
                    study_id="test_study",
                    gap_number=2,
                    gap_day_total=3643,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=1),
                            end=pd.Timestamp(year=2018, month=2, day=12),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=1),
                                    end=pd.Timestamp(year=2018, month=2, day=12)
                                ).date
                            )
                        ),
                        Gap(
                            start=pd.Timestamp(year=2018, month=2, day=24),
                            end=pd.Timestamp(year=2026, month=1, day=1),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2018, month=2, day=24),
                                    end=pd.Timestamp(year=2026, month=1, day=1)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_DURING_TIMEFRAME
                )
        ),
        # Inside And After Timeframe
        (
                [
                    Timeblock(
                        start=pd.Timestamp(year=2018, month=2, day=13),
                        end=pd.Timestamp(year=2018, month=2, day=23),
                        milestone=Milestone("Milestone 1", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2018, month=2, day=18)
                    ),
                    Timeblock(
                        start=pd.Timestamp(year=2028, month=2, day=13),
                        end=pd.Timestamp(year=2028, month=2, day=23),
                        milestone=Milestone("Milestone 2", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2028, month=2, day=18)
                    )
                ],
                ItemInformation(
                    study_id="test_study",
                    gap_number=2,
                    gap_day_total=3643,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=1),
                            end=pd.Timestamp(year=2018, month=2, day=12),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=1),
                                    end=pd.Timestamp(year=2018, month=2, day=12)
                                ).date
                            )
                        ),
                        Gap(
                            start=pd.Timestamp(year=2018, month=2, day=24),
                            end=pd.Timestamp(year=2026, month=1, day=1),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2018, month=2, day=24),
                                    end=pd.Timestamp(year=2026, month=1, day=1)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_AFTER_TIMEFRAME
                )
        ),
        # Inside, Before And After Timeframe
        (
                [
                    Timeblock(
                        start=pd.Timestamp(year=2008, month=2, day=13),
                        end=pd.Timestamp(year=2008, month=2, day=23),
                        milestone=Milestone("Milestone 1", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2008, month=2, day=18)
                    ),
                    Timeblock(
                        start=pd.Timestamp(year=2018, month=2, day=13),
                        end=pd.Timestamp(year=2018, month=2, day=23),
                        milestone=Milestone("Milestone 2", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2018, month=2, day=18)
                    ),
                    MergedTimeblock(
                        start=pd.Timestamp(year=2028, month=2, day=13),
                        end=pd.Timestamp(year=2028, month=2, day=23),
                        milestones=[
                            Milestone("Milestone 3", offset_before=2, offset_after=3, active=True),
                            Milestone("Milestone 4", offset_before=3, offset_after=2, active=True)
                        ],
                        timestamps=[
                            pd.Timestamp(year=2028, month=2, day=15),
                            pd.Timestamp(year=2028, month=2, day=21)
                        ]
                    )
                ],
                ItemInformation(
                    study_id="test_study",
                    gap_number=2,
                    gap_day_total=3643,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=1),
                            end=pd.Timestamp(year=2018, month=2, day=12),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=1),
                                    end=pd.Timestamp(year=2018, month=2, day=12)
                                ).date
                            )
                        ),
                        Gap(
                            start=pd.Timestamp(year=2018, month=2, day=24),
                            end=pd.Timestamp(year=2026, month=1, day=1),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2018, month=2, day=24),
                                    end=pd.Timestamp(year=2026, month=1, day=1)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_AFTER_TIMEFRAME
                )
        ),
        # Before And After Timeframe
        (
                [
                    Timeblock(
                        start=pd.Timestamp(year=2008, month=2, day=13),
                        end=pd.Timestamp(year=2008, month=2, day=23),
                        milestone=Milestone("Milestone 1", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2008, month=2, day=18)
                    ),
                    Timeblock(
                        start=pd.Timestamp(year=2028, month=2, day=13),
                        end=pd.Timestamp(year=2028, month=2, day=23),
                        milestone=Milestone("Milestone 3", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2028, month=2, day=18)
                    )
                ],
                ItemInformation(
                    study_id="test_study",
                    gap_number=1,
                    gap_day_total=3654,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=1),
                            end=pd.Timestamp(year=2026, month=1, day=1),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=1),
                                    end=pd.Timestamp(year=2026, month=1, day=1)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.NO_ACTIVITY_OVER_TIMEFRAME
                )
        ),
        # Timeblock is exactly timeframe
        (
                [
                    Timeblock(
                        start=pd.Timestamp(year=2016, month=1, day=1),
                        end=pd.Timestamp(year=2026, month=1, day=1),
                        milestone=Milestone("Milestone 1", offset_before=1827, offset_after=1827, active=True),
                        timestamp=pd.Timestamp(year=2021, month=1, day=1)
                    )
                ],
                ItemInformation(
                    study_id="test_study",
                    gap_number=1,
                    gap_day_total=3654,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=1),
                            end=pd.Timestamp(year=2026, month=1, day=1),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=1),
                                    end=pd.Timestamp(year=2026, month=1, day=1)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_AFTER_TIMEFRAME
                )
        ),
        # Partially after timeframe
        (
                [
                    Timeblock(
                        start=pd.Timestamp(year=2025, month=12, day=25),
                        end=pd.Timestamp(year=2026, month=1, day=4),
                        milestone=Milestone("Milestone 1", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2025, month=12, day=30)
                    )
                ],
                ItemInformation(
                    study_id="test_study",
                    gap_number=1,
                    gap_day_total=3646,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=1),
                            end=pd.Timestamp(year=2025, month=12, day=24),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=1),
                                    end=pd.Timestamp(year=2025, month=12, day=24)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_AFTER_TIMEFRAME
                )
        ),
        # Partially before timeframe
        (
                [
                    Timeblock(
                        start=pd.Timestamp(year=2015, month=12, day=25),
                        end=pd.Timestamp(year=2016, month=1, day=4),
                        milestone=Milestone("Milestone 1", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2015, month=12, day=30)
                    )
                ],
                ItemInformation(
                    study_id="test_study",
                    gap_number=1,
                    gap_day_total=3650,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=5),
                            end=pd.Timestamp(year=2026, month=1, day=1),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=5),
                                    end=pd.Timestamp(year=2026, month=1, day=1)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_DURING_TIMEFRAME
                )
        ),
        # Inside and partially after
        (
                [
                    Timeblock(
                        start=pd.Timestamp(year=2018, month=2, day=13),
                        end=pd.Timestamp(year=2018, month=2, day=23),
                        milestone=Milestone("Milestone 1", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2018, month=2, day=18)
                    ),
                    Timeblock(
                        start=pd.Timestamp(year=2025, month=12, day=25),
                        end=pd.Timestamp(year=2026, month=1, day=4),
                        milestone=Milestone("Milestone 2", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2025, month=12, day=30)
                    )
                ],
                ItemInformation(
                    study_id="test_study",
                    gap_number=2,
                    gap_day_total=774 + 2861,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=1),
                            end=pd.Timestamp(year=2018, month=2, day=12),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=1),
                                    end=pd.Timestamp(year=2018, month=2, day=12)
                                ).date
                            )
                        ),
                        Gap(
                            start=pd.Timestamp(year=2018, month=2, day=24),
                            end=pd.Timestamp(year=2025, month=12, day=24),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2018, month=2, day=24),
                                    end=pd.Timestamp(year=2025, month=12, day=24)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_AFTER_TIMEFRAME
                )
        ),
        # Inside and partially before
        (
                [
                    Timeblock(
                        start=pd.Timestamp(year=2015, month=12, day=25),
                        end=pd.Timestamp(year=2016, month=1, day=4),
                        milestone=Milestone("Milestone 1", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2025, month=12, day=30)
                    ),
                    Timeblock(
                        start=pd.Timestamp(year=2018, month=2, day=13),
                        end=pd.Timestamp(year=2018, month=2, day=23),
                        milestone=Milestone("Milestone 2", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2018, month=2, day=18)
                    )
                ],
                ItemInformation(
                    study_id="test_study",
                    gap_number=2,
                    gap_day_total=770 + 2869,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=5),
                            end=pd.Timestamp(year=2018, month=2, day=12),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=5),
                                    end=pd.Timestamp(year=2018, month=2, day=12)
                                ).date
                            )
                        ),
                        Gap(
                            start=pd.Timestamp(year=2018, month=2, day=24),
                            end=pd.Timestamp(year=2026, month=1, day=1),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2018, month=2, day=24),
                                    end=pd.Timestamp(year=2026, month=1, day=1)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_DURING_TIMEFRAME
                )
        ),
        # Two Inside and One Partially After
        (
                [
                    Timeblock(
                        start=pd.Timestamp(year=2018, month=2, day=13),
                        end=pd.Timestamp(year=2018, month=2, day=23),
                        milestone=Milestone("Milestone 1", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2018, month=2, day=18)
                    ),
                    Timeblock(
                        start=pd.Timestamp(year=2021, month=4, day=11),
                        end=pd.Timestamp(year=2021, month=4, day=21),
                        milestone=Milestone("Milestone 2", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2021, month=4, day=16)
                    ),
                    Timeblock(
                        start=pd.Timestamp(year=2025, month=12, day=25),
                        end=pd.Timestamp(year=2026, month=1, day=4),
                        milestone=Milestone("Milestone 3", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2025, month=12, day=30)
                    )
                ],
                ItemInformation(
                    study_id="test_study",
                    gap_number=3,
                    gap_day_total=774 + 1142 + 1708,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=1),
                            end=pd.Timestamp(year=2018, month=2, day=12),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=1),
                                    end=pd.Timestamp(year=2018, month=2, day=12)
                                ).date
                            )
                        ),
                        Gap(
                            start=pd.Timestamp(year=2018, month=2, day=24),
                            end=pd.Timestamp(year=2021, month=4, day=10),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2018, month=2, day=24),
                                    end=pd.Timestamp(year=2021, month=4, day=10)
                                ).date
                            )
                        ),
                        Gap(
                            start=pd.Timestamp(year=2021, month=4, day=22),
                            end=pd.Timestamp(year=2025, month=12, day=24),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2021, month=4, day=22),
                                    end=pd.Timestamp(year=2025, month=12, day=24)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_AFTER_TIMEFRAME
                )
        ),
        # Two Inside and One Partially Before
        (
                [
                    Timeblock(
                        start=pd.Timestamp(year=2015, month=12, day=25),
                        end=pd.Timestamp(year=2016, month=1, day=4),
                        milestone=Milestone("Milestone 1", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2015, month=12, day=30)
                    ),
                    MergedTimeblock(
                        start=pd.Timestamp(year=2018, month=2, day=13),
                        end=pd.Timestamp(year=2018, month=2, day=23),
                        milestones=[
                            Milestone("Milestone 1", offset_before=2, offset_after=3, active=True),
                            Milestone("Milestone 1", offset_before=3, offset_after=2, active=True)
                        ],
                        timestamps=[
                            pd.Timestamp(year=2018, month=2, day=15),
                            pd.Timestamp(year=2018, month=2, day=21)
                        ]
                    ),
                    Timeblock(
                        start=pd.Timestamp(year=2021, month=4, day=11),
                        end=pd.Timestamp(year=2021, month=4, day=21),
                        milestone=Milestone("Milestone 2", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2021, month=4, day=16)
                    )
                ],
                ItemInformation(
                    study_id="test_study",
                    gap_number=3,
                    gap_day_total=770 + 1142 + 1716,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=5),
                            end=pd.Timestamp(year=2018, month=2, day=12),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=5),
                                    end=pd.Timestamp(year=2018, month=2, day=12)
                                ).date
                            )
                        ),
                        Gap(
                            start=pd.Timestamp(year=2018, month=2, day=24),
                            end=pd.Timestamp(year=2021, month=4, day=10),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2018, month=2, day=24),
                                    end=pd.Timestamp(year=2021, month=4, day=10)
                                ).date
                            )
                        ),
                        Gap(
                            start=pd.Timestamp(year=2021, month=4, day=22),
                            end=pd.Timestamp(year=2026, month=1, day=1),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2021, month=4, day=22),
                                    end=pd.Timestamp(year=2026, month=1, day=1)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_DURING_TIMEFRAME
                )
        ),
        # Two Inside, One Partially Before and One Partially After
        (
                [
                    Timeblock(
                        start=pd.Timestamp(year=2015, month=12, day=25),
                        end=pd.Timestamp(year=2016, month=1, day=4),
                        milestone=Milestone("Milestone 1", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2015, month=12, day=30)
                    ),
                    MergedTimeblock(
                        start=pd.Timestamp(year=2018, month=2, day=13),
                        end=pd.Timestamp(year=2018, month=2, day=23),
                        milestones=[
                            Milestone("Milestone 1", offset_before=2, offset_after=3, active=True),
                            Milestone("Milestone 1", offset_before=3, offset_after=2, active=True)
                        ],
                        timestamps=[
                            pd.Timestamp(year=2018, month=2, day=15),
                            pd.Timestamp(year=2018, month=2, day=21)
                        ]
                    ),
                    Timeblock(
                        start=pd.Timestamp(year=2021, month=4, day=11),
                        end=pd.Timestamp(year=2021, month=4, day=21),
                        milestone=Milestone("Milestone 2", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2021, month=4, day=16)
                    ),
                    Timeblock(
                        start=pd.Timestamp(year=2025, month=12, day=25),
                        end=pd.Timestamp(year=2026, month=1, day=4),
                        milestone=Milestone("Milestone 3", offset_before=5, offset_after=5, active=True),
                        timestamp=pd.Timestamp(year=2025, month=12, day=30)
                    )
                ],
                ItemInformation(
                    study_id="test_study",
                    gap_number=3,
                    gap_day_total=770 + 1142 + 1708,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2016, month=1, day=5),
                            end=pd.Timestamp(year=2018, month=2, day=12),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2016, month=1, day=5),
                                    end=pd.Timestamp(year=2018, month=2, day=12)
                                ).date
                            )
                        ),
                        Gap(
                            start=pd.Timestamp(year=2018, month=2, day=24),
                            end=pd.Timestamp(year=2021, month=4, day=10),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2018, month=2, day=24),
                                    end=pd.Timestamp(year=2021, month=4, day=10)
                                ).date
                            )
                        ),
                        Gap(
                            start=pd.Timestamp(year=2021, month=4, day=22),
                            end=pd.Timestamp(year=2025, month=12, day=24),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2021, month=4, day=22),
                                    end=pd.Timestamp(year=2025, month=12, day=24)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_AFTER_TIMEFRAME
                )
        ),
    ]
)
def test_generate_gap_information_apply(merged_timeblock_list, expected_item_information, app_test_session_store):

    study_id = 'test_study'
    timeframe_start = pd.Timestamp(app_test_session_store["timeframe_start"])
    timeframe_end = pd.Timestamp(app_test_session_store["timeframe_end"])

    output = Data._generate_gap_information_apply(study_id, merged_timeblock_list, timeframe_start, timeframe_end)
    assert output == expected_item_information


def test_generate_gap_information(app_test_dataclass, app_test_session_store):

    test_df_no_timeblocks = pd.DataFrame({
        app_test_dataclass.unique_identity_label: [43, 123]
    })

    test_df_with_timeblocks = pd.DataFrame({
        app_test_dataclass.unique_identity_label: [43, 123, 342],
        'Merged Time Block': [
            [],
            [
                Timeblock(start=pd.Timestamp(year=2022, month=6, day=20), end=pd.Timestamp(year=2022, month=7, day=10),
                          milestone=Milestone("Milestone 1", offset_before=10, offset_after=10, active=True),
                          timestamp=pd.Timestamp(year=2022, month=6, day=30)),
                Timeblock(start=pd.Timestamp(year=2022, month=6, day=25), end=pd.Timestamp(year=2022, month=7, day=15),
                          milestone=Milestone("Milestone 2", offset_before=10, offset_after=10, active=True),
                          timestamp=pd.Timestamp(year=2022, month=7, day=5))
            ],
            [
                Timeblock(start=pd.Timestamp(year=2022, month=6, day=20), end=pd.Timestamp(year=2022, month=7, day=10),
                          milestone=Milestone("Milestone 1", offset_before=10, offset_after=10, active=True),
                          timestamp=pd.Timestamp(year=2022, month=6, day=30)),
                MergedTimeblock(
                    start=pd.Timestamp(year=2022, month=6, day=25),
                    end=pd.Timestamp(year=2022, month=7, day=25),
                    milestones=[
                        Milestone("Milestone 2", offset_before=10, offset_after=10, active=True),
                        Milestone("Milestone 3", offset_before=5, offset_after=5, active=True)
                    ],
                    timestamps=[
                        pd.Timestamp(year=2022, month=7, day=5),
                        pd.Timestamp(year=2022, month=7, day=20)
                    ]
                )
            ]
        ]
    })

    assert test_df_no_timeblocks.equals(app_test_dataclass.generate_gap_information(
        test_df_no_timeblocks, app_test_session_store))
    assert 'Gap Information' in app_test_dataclass.generate_gap_information(
        test_df_with_timeblocks, app_test_session_store).columns


@pytest.mark.parametrize(
    "item_information, expected_weight",
    [
        (
                ItemInformation(
                    study_id="test_study",
                    gap_number=0,
                    gap_day_total=0,
                    gap_lst=[],
                    active_during_timeframe=False,
                    status=ItemStatus.NO_STATUS_GIVEN
                ),
                pd.Series([0, 0])
        ),
        (
                ItemInformation(
                    study_id="test_study",
                    gap_number=0,
                    gap_day_total=0,
                    gap_lst=[],
                    active_during_timeframe=False,
                    status=ItemStatus.CLOSING_BEFORE_TIMEFRAME
                ),
                pd.Series([0, 0])
        ),
        (
                ItemInformation(
                    study_id="test_study",
                    gap_number=0,
                    gap_day_total=0,
                    gap_lst=[],
                    active_during_timeframe=False,
                    status=ItemStatus.STARTING_AFTER_TIMEFRAME
                ),
                pd.Series([0, 0])
        ),
        (
                ItemInformation(
                    study_id="test_study",
                    gap_number=1,
                    gap_day_total=1000,
                    gap_lst=[],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_AFTER_TIMEFRAME
                ),
                pd.Series([1 / 1000, 1])
        ),
        (
                ItemInformation(
                    study_id="test_study",
                    gap_number=4,
                    gap_day_total=190,
                    gap_lst=[],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_DURING_TIMEFRAME
                ),
                pd.Series([1 / 190, 1 / 4])
        ),
        # Closing after timeframe with no gaps in timeframe
        (
                ItemInformation(
                    study_id="test_study",
                    gap_number=0,
                    gap_day_total=0,
                    gap_lst=[],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_AFTER_TIMEFRAME
                ),
                pd.Series([1, 1])
        )
    ]
)
def test_compute_weights_apply(item_information: ItemInformation, expected_weight, app_test_dataclass):

    assert Data._compute_weights_apply(item_information).equals(expected_weight)


def test_compute_weights(app_test_dataclass):

    test_df = pd.DataFrame({
        app_test_dataclass.unique_identity_label: ["Study 1", "Study 2", "Study 3", "Study 4", "Study 5"],
        'Gap Information': [
            ItemInformation(
                study_id="Study 1",
                gap_number=0,
                gap_day_total=0,
                gap_lst=[],
                active_during_timeframe=False,
                status=ItemStatus.NO_STATUS_GIVEN
            ),
            ItemInformation(
                study_id="Study 2",
                gap_number=1,
                gap_day_total=1000,
                gap_lst=[],
                active_during_timeframe=True,
                status=ItemStatus.CLOSING_AFTER_TIMEFRAME
            ),
            ItemInformation(
                study_id="Study 3",
                gap_number=4,
                gap_day_total=190,
                gap_lst=[],
                active_during_timeframe=True,
                status=ItemStatus.CLOSING_DURING_TIMEFRAME
            ),
            ItemInformation(
                study_id="Study 4",
                gap_number=2,
                gap_day_total=20,
                gap_lst=[],
                active_during_timeframe=True,
                status=ItemStatus.CLOSING_DURING_TIMEFRAME
            ),
            ItemInformation(
                study_id="Study 5",
                gap_number=2,
                gap_day_total=500,
                gap_lst=[],
                active_during_timeframe=True,
                status=ItemStatus.CLOSING_DURING_TIMEFRAME
            ),
        ]
    })

    app_test_dataclass.compute_weights(test_df)

    assert 'Day Weights' not in test_df.columns
    assert 'Gap Weights' not in test_df.columns
    assert 'Weights' in test_df.columns

    assert list(test_df[app_test_dataclass.unique_identity_label]) == \
           ['Study 4', 'Study 3', 'Study 5', 'Study 2', 'Study 1']


@pytest.mark.parametrize(
    "kwargs, expected_date_list",
    [
        (
                {
                    "timeframe_start": pd.Timestamp(year=2022, month=5, day=10),
                    "timeframe_end": pd.Timestamp(year=2022, month=5, day=25),
                    "transfer_window_type": 'D'
                },
                pd.date_range(
                    start=pd.Timestamp(year=2022, month=5, day=10),
                    end=pd.Timestamp(year=2022, month=5, day=25)
                ).date.tolist()
        ),
        (
                {
                    "timeframe_start": pd.Timestamp(year=2022, month=5, day=10),  # Tuesday
                    "timeframe_end": pd.Timestamp(year=2022, month=5, day=25),  # Wednesday
                    "transfer_window_type": 'W'
                },
                [pd.Timestamp(year=2022, month=5, day=15).date()]  # only one week fully fits in the timeframe
        ),
        (
                {
                    "timeframe_start": pd.Timestamp(year=2022, month=5, day=8),  # Sunday
                    "timeframe_end": pd.Timestamp(year=2022, month=5, day=22),  # Sunday
                    "transfer_window_type": 'W'
                },
                # sunday is the starting day of the week in pandas
                [pd.Timestamp(year=2022, month=5, day=8).date(), pd.Timestamp(year=2022, month=5, day=15).date()]
        )
    ]
)
def test_timeframe_date_range(kwargs, expected_date_list):

    assert expected_date_list == Data._timeframe_date_range(**kwargs)


@pytest.mark.parametrize(
    "item_info, moving_dates, transfer_window, expected_result",
    [
        # No moving dates - day
        (
                ItemInformation(
                    study_id="test_study",
                    gap_number=0,
                    gap_day_total=0,
                    gap_lst=[],
                    active_during_timeframe=False,
                    status=ItemStatus.NO_STATUS_GIVEN
                ),
                [
                    {
                        k: 2 for k in pd.date_range(
                        start=pd.Timestamp(year=2022, month=5, day=10),
                        end=pd.Timestamp(year=2022, month=5, day=25)
                        ).date.tolist()
                    }
                ],
                "D",
                pd.Series([False, 'Not Moved', pd.NaT, ''])
        ),
        # No moving dates - Week
        (
                ItemInformation(
                    study_id="test_study",
                    gap_number=0,
                    gap_day_total=0,
                    gap_lst=[],
                    active_during_timeframe=False,
                    status=ItemStatus.NO_STATUS_GIVEN
                ),
                [
                    {
                        k: 2 for k in pd.date_range(
                        start=pd.Timestamp(year=2022, month=5, day=8),
                        end=pd.Timestamp(year=2022, month=5, day=15),
                        freq="W"
                        ).date.tolist()
                    }
                ],
                "W",
                pd.Series([False, 'Not Moved', pd.NaT, ''])
        ),
        # Can move on first gap and first period - days
        (
                ItemInformation(
                    study_id="test_study",
                    gap_number=1,
                    gap_day_total=10,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2023, month=1, day=1),
                            end=pd.Timestamp(year=2023, month=1, day=10),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2023, month=1, day=1),
                                    end=pd.Timestamp(year=2023, month=1, day=10)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_DURING_TIMEFRAME
                ),
                [
                    {
                        k: 2 for k in pd.date_range(
                        start=pd.Timestamp(year=2023, month=1, day=5),
                        end=pd.Timestamp(year=2023, month=1, day=8)
                        ).date.tolist()
                    }
                ],
                "D",
                pd.Series([True, 'Period 1', pd.Timestamp(year=2023, month=1, day=5).date(), ''])
        ),
        # Can move on first gap and first period - weeks
        (
                ItemInformation(
                    study_id="test_study",
                    gap_number=1,
                    gap_day_total=10,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2023, month=6, day=9),
                            end=pd.Timestamp(year=2023, month=6, day=19),
                            timestamp_lst=set(
                                # Should be overwritten
                                pd.date_range(
                                    start=pd.Timestamp(year=2023, month=6, day=9),
                                    end=pd.Timestamp(year=2023, month=6, day=19),
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_DURING_TIMEFRAME
                ),
                [
                    {
                        k: 2 for k in pd.date_range(
                        start=pd.Timestamp(year=2023, month=6, day=11),
                        end=pd.Timestamp(year=2023, month=6, day=25),
                        freq="W"
                        ).date.tolist()
                    }
                ],
                "D",
                pd.Series([True, 'Period 1', pd.Timestamp(year=2023, month=6, day=11).date(), ''])
        ),
        # Can move on second gap, third period - days
        (
                ItemInformation(
                    study_id="test_study",
                    gap_number=2,
                    gap_day_total=10,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2023, month=1, day=1),
                            end=pd.Timestamp(year=2023, month=1, day=5),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2023, month=1, day=1),
                                    end=pd.Timestamp(year=2023, month=1, day=5)
                                ).date
                            )
                        ),
                        Gap(
                            start=pd.Timestamp(year=2023, month=6, day=15),
                            end=pd.Timestamp(year=2023, month=6, day=20),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2023, month=6, day=15),
                                    end=pd.Timestamp(year=2023, month=6, day=20)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_DURING_TIMEFRAME
                ),
                [
                    {
                        k: 2 for k in pd.date_range(
                        start=pd.Timestamp(year=2022, month=11, day=1),
                        end=pd.Timestamp(year=2022, month=12, day=31)
                        ).date.tolist()
                    },
                    {},
                    {
                        k: 2 for k in pd.date_range(
                        start=pd.Timestamp(year=2023, month=6, day=15),
                        end=pd.Timestamp(year=2023, month=6, day=17)
                        ).date.tolist()
                    },
                ],
                "D",
                pd.Series([True, 'Period 3', pd.Timestamp(year=2023, month=6, day=15).date(), ''])
        ),
        # Can move on second gap, third period - weeks
        (
                ItemInformation(
                    study_id="test_study",
                    gap_number=2,
                    gap_day_total=10,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2022, month=3, day=3),
                            end=pd.Timestamp(year=2022, month=3, day=20),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2022, month=3, day=3),
                                    end=pd.Timestamp(year=2022, month=3, day=20)
                                ).date
                            )
                        ),
                        Gap(
                            start=pd.Timestamp(year=2022, month=10, day=7),
                            end=pd.Timestamp(year=2022, month=10, day=26),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2022, month=10, day=9),
                                    end=pd.Timestamp(year=2022, month=10, day=23)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_DURING_TIMEFRAME
                ),
                [
                    {
                        k: 2 for k in pd.date_range(
                        start=pd.Timestamp(year=2022, month=4, day=1),
                        end=pd.Timestamp(year=2022, month=4, day=21),
                        freq="W"
                        ).date.tolist()
                    },
                    {},
                    {
                        k: 2 for k in pd.date_range(
                        start=pd.Timestamp(year=2022, month=10, day=16),
                        end=pd.Timestamp(year=2022, month=10, day=23),
                        freq="W"
                        ).date.tolist()
                    },
                ],
                "D",
                pd.Series([True, 'Period 3', pd.Timestamp(year=2022, month=10, day=16).date(), ''])
        ),
    ]
)
def test_migrate_study_apply_day_frequency(item_info, moving_dates, transfer_window, expected_result):

    assert expected_result.equals(Data._migrate_study_apply(item_info, moving_dates, transfer_window))

    should_move_date = expected_result[2]
    should_move_period = expected_result[1].split('Period ')

    if pd.notna(should_move_date):
        assert len(should_move_period) == 2
        period_index = int(should_move_period[1]) - 1
        assert moving_dates[period_index][should_move_date] == 1


@pytest.mark.parametrize(
    "item_info, moving_dates, should_remove, transfer_window_type",
    [
        # Day Frequency Filled
        (
                ItemInformation(
                    study_id="test_study",
                    gap_number=1,
                    gap_day_total=10,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2023, month=6, day=13),
                            end=pd.Timestamp(year=2023, month=6, day=23),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2023, month=6, day=13),
                                    end=pd.Timestamp(year=2023, month=6, day=23)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_DURING_TIMEFRAME
                ),
                [
                    {
                        pd.Timestamp(year=2023, month=6, day=15).date(): 1,
                        pd.Timestamp(year=2023, month=6, day=16).date(): 2
                    },
                ],
                pd.Timestamp(year=2023, month=6, day=15).date(),
                'D'
        ),
        # Week Frequency Filled
        (
                ItemInformation(
                    study_id="test_study",
                    gap_number=1,
                    gap_day_total=10,
                    gap_lst=[
                        Gap(
                            start=pd.Timestamp(year=2022, month=5, day=8),  # Sunday
                            end=pd.Timestamp(year=2022, month=5, day=15),
                            timestamp_lst=set(
                                pd.date_range(
                                    start=pd.Timestamp(year=2022, month=5, day=8),
                                    end=pd.Timestamp(year=2022, month=5, day=15)
                                ).date
                            )
                        )
                    ],
                    active_during_timeframe=True,
                    status=ItemStatus.CLOSING_DURING_TIMEFRAME
                ),
                [
                    {
                        pd.Timestamp(year=2022, month=5, day=8).date(): 1,
                        pd.Timestamp(year=2023, month=5, day=15).date(): 2,
                        pd.Timestamp(year=2023, month=5, day=22).date(): 2
                    }
                ],
                pd.Timestamp(year=2022, month=5, day=8).date(),
                'W'
        )
    ]
)
def test_migrate_study_apply_date_capacity(item_info, moving_dates, should_remove, transfer_window_type):

    assert should_remove in moving_dates[0]

    Data._migrate_study_apply(item_info, moving_dates, transfer_window_type)

    assert should_remove not in moving_dates[0]
    for key, value in moving_dates[0].items():
        assert value == 2


def test_migration_table_processing(app_test_dataclass):

    timeframe_start = pd.Timestamp(year=2022, month=1, day=2)
    timeframe_end = pd.Timestamp(year=2022, month=12, day=31)

    mock_df = pd.DataFrame({
        "study id": ["12", "34", "56", "78"],
        "Gap Information": [
            # Study with no gaps
            ItemInformation(
                study_id="12",
                gap_number=0,
                gap_day_total=0,
                gap_lst=[],
                active_during_timeframe=False,
                status=ItemStatus.NO_STATUS_GIVEN
            ),
            # Study with Gap outside timeframe
            ItemInformation(
                study_id="34",
                gap_number=1,
                gap_day_total=20,
                gap_lst=[
                    Gap(
                        start=pd.Timestamp(year=2021, month=9, day=1),
                        end=pd.Timestamp(year=2021, month=9, day=20),
                        timestamp_lst=pd.date_range(
                            start=pd.Timestamp(year=2021, month=9, day=1),
                            end=pd.Timestamp(year=2021, month=9, day=20)
                        ).date
                    )
                ],
                active_during_timeframe=False,
                status=ItemStatus.CLOSING_BEFORE_TIMEFRAME
            ),
            # Study that moves as normal
            ItemInformation(
                study_id="56",
                gap_number=1,
                gap_day_total=17,
                gap_lst=[
                    Gap(
                        start=pd.Timestamp(year=2021, month=12, day=24),
                        end=pd.Timestamp(year=2022, month=1, day=9),
                        timestamp_lst=pd.date_range(
                            start=pd.Timestamp(year=2021, month=12, day=24),
                            end=pd.Timestamp(year=2022, month=1, day=9),
                        ).date
                    )
                ],
                active_during_timeframe=True,
                status=ItemStatus.CLOSING_DURING_TIMEFRAME
            ),
            # Study that cant move because its transfer date has been filled
            ItemInformation(
                study_id="78",
                gap_number=1,
                gap_day_total=17,
                gap_lst=[
                    Gap(
                        start=pd.Timestamp(year=2021, month=12, day=24),
                        end=pd.Timestamp(year=2022, month=1, day=9),
                        timestamp_lst=pd.date_range(
                            start=pd.Timestamp(year=2021, month=12, day=24),
                            end=pd.Timestamp(year=2022, month=1, day=9),
                        ).date
                    )
                ],
                active_during_timeframe=True,
                status=ItemStatus.CLOSING_DURING_TIMEFRAME
            ),
        ],
        "Weights": [1, 0.8, 0.6, 0.4]
    })

    day_freq_df, day_period_start_end = app_test_dataclass.migration_table_processing(
        mock_df, {"timeframe_start": timeframe_start, "timeframe_end": timeframe_end}, 1, 100, "D"
    )

    expected_day_cols = ['study id', 'Gap Information', 'Weights', 'Transfer Flag', 'Transfer Start Date',
                         'Period Moved']

    assert len(day_freq_df.columns) == len(expected_day_cols)
    assert all(col in day_freq_df.columns for col in expected_day_cols)
    assert day_freq_df['Transfer Flag'].equals(pd.Series([False, False, True, True]))
    assert day_freq_df['Transfer Start Date'].equals(pd.Series([pd.NaT, pd.NaT, date(year=2022, month=1, day=2),
                                                                date(year=2022, month=1, day=3)]))
    assert day_freq_df['Period Moved'].equals(pd.Series(['Not Moved', 'Not Moved', 'Period 1', 'Period 1']))
    assert day_period_start_end == [
        (date(2022, 1, 2), date(2022, 4, 11)),
        (date(2022, 4, 12), date(2022, 7, 20)),
        (date(2022, 7, 21), date(2022, 10, 28)),
        (date(2022, 10, 29), date(2022, 12, 31))
    ]

    expected_week_cols = ['study id', 'Gap Information', 'Weights', 'Transfer Flag', 'Transfer Start Date',
                          'Transfer Range', 'Period Moved']

    week_freq_df, week_period_start_end = app_test_dataclass.migration_table_processing(
        mock_df, {"timeframe_start": timeframe_start, "timeframe_end": timeframe_end}, 1, 14, "W"
    )
    assert len(week_freq_df.columns) == len(expected_week_cols)
    assert all(col in week_freq_df.columns for col in expected_week_cols)
    assert week_freq_df['Transfer Flag'].equals(pd.Series([False, False, True, False]))
    assert week_freq_df['Transfer Start Date'].equals(
        pd.Series([pd.NaT, pd.NaT, date(year=2022, month=1, day=2), pd.NaT]))
    assert week_freq_df['Transfer Range'].equals(pd.Series(['', '', '2022-01-02 to 2022-01-08', '']))
    assert week_freq_df['Period Moved'].equals(pd.Series(['Not Moved', 'Not Moved', 'Period 1', 'Not Moved']))


@pytest.mark.parametrize(
    "input_df, group_col, expected_df",
    [
        (
            pd.DataFrame({
                'study_id': [12, 23, 34, 45, 56],
                'Transfer Flag': [False, True, True, False, True],
                'Period Moved': ['Not Moved', 'Period 1', 'Period 1', 'Not Moved', 'Period 2'],
                'Transfer Range': ['', '', '', '', '']
            }),
            'Overall',
            pd.DataFrame({
                'Total': [5],
                'Transfer Dates Found': [3],
                'Transfer Dates Not Found': [2],
                'Period 1': [2],
                'Period 2': [1]
            })
        ),
        (
            pd.DataFrame({
                'study_id': [12, 23, 34, 45, 56],
                'grouping_col': ["type 1", "type 1", "type 1", "type 2", "type 2"],
                'Transfer Flag': [False, True, True, False, True],
                'Period Moved': ['Not Moved', 'Period 1', 'Period 1', 'Not Moved', 'Period 2'],
                'Transfer Range': ['', '', '', '', '']
            }),
            'grouping_col',
            pd.DataFrame({
                'grouping_col': ["type 1", "type 2"],
                'Total': [3, 2],
                'Transfer Dates Found': [2, 1],
                'Transfer Dates Not Found': [1, 1],
                'Period 1': [2, 0],
                'Period 2': [0, 1]
            })
        ),
    ]
)
def test_table_formatting(input_df, group_col, expected_df):

    result_df = Data.table_formatting(input_df, group_col)
    pd.testing.assert_frame_equal(result_df, expected_df, check_names=False)


def test_format_for_export():

    input_df = pd.DataFrame({
        "Study ID": ["123", "567", "890"],
        "Transfer Flag": [False, True, True],
        "Transfer Start Date": [
            "",
            pd.Timestamp(year=2021, month=12, day=24).date(),
            pd.Timestamp(year=2016, month=1, day=1).date()
        ],
        "Period Moved": ["Not Moved", "Period 2", "Period 1"],
        "Gap Information": [
            ItemInformation(
                study_id="123",
                gap_number=0,
                gap_day_total=0,
                gap_lst=[],
                active_during_timeframe=False,
                status=ItemStatus.NO_STATUS_GIVEN
            ),
            ItemInformation(
                study_id="567",
                gap_number=1,
                gap_day_total=17,
                gap_lst=[
                    Gap(
                        start=pd.Timestamp(year=2021, month=12, day=24),
                        end=pd.Timestamp(year=2022, month=1, day=9),
                        timestamp_lst=pd.date_range(
                            start=pd.Timestamp(year=2021, month=12, day=24),
                            end=pd.Timestamp(year=2022, month=1, day=9),
                        ).date
                    )
                ],
                active_during_timeframe=True,
                status=ItemStatus.CLOSING_DURING_TIMEFRAME
            ),
            ItemInformation(
                study_id="890",
                gap_number=3,
                gap_day_total=774 + 1142 + 1708,
                gap_lst=[
                    Gap(
                        start=pd.Timestamp(year=2016, month=1, day=1),
                        end=pd.Timestamp(year=2018, month=2, day=12),
                        timestamp_lst=set(
                            pd.date_range(
                                start=pd.Timestamp(year=2016, month=1, day=1),
                                end=pd.Timestamp(year=2018, month=2, day=12)
                            ).date
                        )
                    ),
                    Gap(
                        start=pd.Timestamp(year=2018, month=2, day=24),
                        end=pd.Timestamp(year=2021, month=4, day=10),
                        timestamp_lst=set(
                            pd.date_range(
                                start=pd.Timestamp(year=2018, month=2, day=24),
                                end=pd.Timestamp(year=2021, month=4, day=10)
                            ).date
                        )
                    ),
                    Gap(
                        start=pd.Timestamp(year=2021, month=4, day=22),
                        end=pd.Timestamp(year=2025, month=12, day=24),
                        timestamp_lst=set(
                            pd.date_range(
                                start=pd.Timestamp(year=2021, month=4, day=22),
                                end=pd.Timestamp(year=2025, month=12, day=24)
                            ).date
                        )
                    )
                ],
                active_during_timeframe=True,
                status=ItemStatus.CLOSING_AFTER_TIMEFRAME
            ),
        ],
        "Weights": [0, 1, 0.5],
        "Time Block": [None, None, None],
        "Merged Time Block": [None, None, None],
        "Persistent Column": ["A", "B", "C"]
    })

    expected_df = pd.DataFrame({
        "Study ID": ["567", "890", "123"],
        "Transfer Status": [True, True, False],
        "Transfer Start Date": [
            pd.Timestamp(year=2021, month=12, day=24).date(),
            pd.Timestamp(year=2016, month=1, day=1).date(),
            ""
        ],
        "Period Moved": ["Period 2", "Period 1", "Not Moved"],
        "Weights": [1, 0.5, 0],
        "Persistent Column": ["B", "C", "A"],
        "Gap Day Total": [17, 3624, 0],
        "Closing Status": ["Closing During Timeframe", "Closing After Timeframe", "No Status Given"],
        "Gap 1 Start": ["2021-12-24", "2016-01-01", None],
        "Gap 1 End": ["2022-01-09", "2018-02-12", None],
        "Gap 1 Length": pd.Series([17, 774, None], dtype=object),
        "Gap 2 Start": [None, "2018-02-24", None],
        "Gap 2 End": [None, "2021-04-10", None],
        "Gap 2 Length": pd.Series([None, 1142, None], dtype=object),
        "Gap 3 Start": [None, "2021-04-22", None],
        "Gap 3 End": [None, "2025-12-24", None],
        "Gap 3 Length": pd.Series([None, 1708, None], dtype=object),
    })

    result_df = Data.format_for_export(input_df).reset_index(drop=True)

    pd.testing.assert_frame_equal(left=expected_df, right=result_df)


@pytest.mark.parametrize(
    "app_session_store, period_start_end, period_length, transfer_window_type, active_studies_per, test_id",
    [
        (
            {
                "timeframe_start": pd.Timestamp(year=2022, month=6, day=10),
                "timeframe_end": pd.Timestamp(year=2022, month=6, day=19),
                "milestones": {
                    "milestone 1": Milestone("milestone 1", offset_after=10, offset_before=10).__dict__,
                    "milestone 2": Milestone("milestone 2", offset_after=5, offset_before=5).__dict__
                },
                "active_filters": []
            },
            [(pd.Timestamp(year=2022, month=6, day=10).date(), pd.Timestamp(year=2022, month=6, day=14).date()),
             (pd.Timestamp(year=2022, month=6, day=15).date(), pd.Timestamp(year=2022, month=6, day=19).date())],
            10,
            'D',
            2,
            "PYT39[Daily Without Filters]"
        ),
        (
            {
                "timeframe_start": pd.Timestamp(year=2022, month=6, day=10),
                "timeframe_end": pd.Timestamp(year=2022, month=6, day=19),
                "milestones": {
                    "milestone 1": Milestone("milestone 1", offset_after=10, offset_before=10).__dict__,
                    "milestone 2": Milestone("milestone 2", offset_after=5, offset_before=5).__dict__
                },
                "active_filters": []
            },
            [(pd.Timestamp(year=2022, month=6, day=12).date(), pd.Timestamp(year=2022, month=6, day=18).date())],
            1,
            'W',
            2,
            "PYT39[Weekly Without Filters]"
        ),
        (
            {
                "timeframe_start": pd.Timestamp(year=2022, month=6, day=10),
                "timeframe_end": pd.Timestamp(year=2022, month=6, day=19),
                "milestones": {
                    "milestone 1": Milestone("milestone 1", offset_after=10, offset_before=10).__dict__,
                    "milestone 2": Milestone("milestone 2", offset_after=5, offset_before=5, active=False).__dict__
                },
                "active_filters": [["Compound", [212, 323, 434]], ["Domain", ["VS", "DM"]]]
            },
            [(pd.Timestamp(year=2022, month=6, day=10).date(), pd.Timestamp(year=2022, month=6, day=14).date()),
             (pd.Timestamp(year=2022, month=6, day=15).date(), pd.Timestamp(year=2022, month=6, day=19).date())],
            10,
            'D',
            2,
            "PYT39[Daily With Filters]"
        ),
        (
            {
                "timeframe_start": pd.Timestamp(year=2022, month=6, day=10),
                "timeframe_end": pd.Timestamp(year=2022, month=6, day=19),
                "milestones": {
                    "milestone 1": Milestone("milestone 1", offset_after=10, offset_before=10).__dict__,
                    "milestone 2": Milestone("milestone 2", offset_after=5, offset_before=5, active=False).__dict__
                },
                "active_filters": [["Compound", [212, 323, 434]], ["Domain", ["VS", "DM"]]]
            },
            [(pd.Timestamp(year=2022, month=6, day=12).date(), pd.Timestamp(year=2022, month=6, day=18).date())],
            1,
            'W',
            2,
            "PYT39[Weekly With Filters]"
        )
    ]
)
def test_format_config_sheet(app_session_store, period_start_end, period_length, transfer_window_type,
                             active_studies_per, test_id):

    result_df = Data.format_config_sheet(app_session_store, period_start_end, period_length, transfer_window_type,
                                         active_studies_per)

    expected_df = load_json_test_dataframe(test_id)

    # Dataframe used for presentation only so dtype comparison is not required
    pd.testing.assert_frame_equal(left=expected_df, right=result_df, check_dtype=False)
