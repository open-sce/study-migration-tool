import pandas as pd
import pytest

from utils import Gap, Timeblock, MergedTimeblock, Milestone, ItemInformation, ItemStatus


class NullContext:
    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def test_apply_offsets():

    milestone_1 = Milestone('Test Milestone')
    test_timestamp_1 = pd.Timestamp(year=2020, month=3, day=20)

    res_1 = milestone_1.apply_offsets(test_timestamp_1)
    assert res_1[0] == pd.Timestamp(year=2020, month=3, day=6), \
        "milestone_1 apply offset [0] expected to be 14 days (default) BEFORE test_timestamp_1!"

    assert res_1[1] == pd.Timestamp(year=2020, month=4, day=3), \
        "milestone_1 apply offset [1] expected to be 14 days (default) AFTER test_timestamp_1!"

    milestone_2 = Milestone('Test Milestone 2', offset_before=20, offset_after=42)

    res_2 = milestone_2.apply_offsets(test_timestamp_1)
    assert res_2[0] == pd.Timestamp(year=2020, month=2, day=29), \
        "milestone_2 apply offset [0] expected to be offset_before days BEFORE test_timestamp_1!"

    assert res_2[1] == pd.Timestamp(year=2020, month=5, day=1), \
        "milestone_2 apply offset [1] expected to be offset_after AFTER test_timestamp_1!"


def test_milestone_equality():

    milestone_1 = Milestone('Test Milestone')
    milestone_2 = Milestone('Test Milestone')

    assert milestone_1 == milestone_2, \
        "milestone_1 and milestone_2 are expected to be equal!"

    milestone_3 = Milestone('Test Milestone', offset_before=1, offset_after=1)
    milestone_4 = Milestone('Test Milestone', offset_before=1, offset_after=2)

    assert milestone_3 != milestone_4, \
        "milestone_3 and milestone_4 are expected to NOT be equal!"

    milestone_5 = Milestone('Test Milestone', offset_before=1, offset_after=2)
    milestone_6 = Milestone('Test Milestone', offset_before=1, offset_after=2)

    assert milestone_5 == milestone_6, \
        "milestone_4 and milestone_5 are expected to be equal!"

    milestone_7 = Milestone('Test Milestone', offset_before=1, offset_after=2, active=False)

    assert milestone_6 != milestone_7, \
        "milestone_6 and milestone_7 are expected to NOT be equal!"


def test_milestone_repr():

    milestone_1 = Milestone('Test Milestone', offset_before=1, offset_after=2, active=False)
    assert milestone_1.__repr__() == "Milestone(label='Test Milestone',offset_before=1,offset_after=2,active=False)", \
        "milestone_1 __repr__ did not follow expected format!"


def test_gap_repr():

    gap_1 = Gap(
        start=pd.Timestamp(year=2020, month=10, day=20),
        end=pd.Timestamp(year=2021, month=10, day=20),
        timestamp_lst=set(
            pd.date_range(
                start=pd.Timestamp(year=2020, month=10, day=20),
                end=pd.Timestamp(year=2021, month=10, day=20)
            ).date
        )
    )
    assert gap_1.__repr__() == "Gap(start=Timestamp('2020-10-20 00:00:00'),end=Timestamp('2021-10-20 00:00:00'))", \
        "gap_1 __repr__ did not follow expected format!"


def test_item_information_repr():

    item_info = ItemInformation(
        study_id='test_study',
        gap_number=1,
        gap_day_total=10,
        gap_lst=[
            Gap(
                start=pd.Timestamp(year=2023, month=2, day=1),
                end=pd.Timestamp(year=2023, month=2, day=10),
                timestamp_lst=set(
                    pd.date_range(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2023, month=2, day=10)
                    ).date
                )
            )
        ],
        active_during_timeframe=False,
        status=ItemStatus.CLOSING_BEFORE_TIMEFRAME
    )
    assert item_info.__repr__() == "ItemInformation(study_id='test_study', gap_number=1, gap_day_total=10, " \
      "gap_lst=[Gap(start=Timestamp('2023-02-01 00:00:00'),end=Timestamp('2023-02-10 00:00:00'))], " \
      "active_during_timeframe=False, status=<ItemStatus.CLOSING_BEFORE_TIMEFRAME: 'Closing Before Timeframe'>)"


def test_timeblock_repr():

    timeblock_1 = Timeblock(
        start=pd.Timestamp(year=2020, month=3, day=20), end=pd.Timestamp(year=2020, month=3, day=30),
        milestone=Milestone(
            'Milestone 1', offset_before=5, offset_after=5, active=True
        ),
        timestamp=pd.Timestamp(year=2020, month=3, day=20)
    )
    assert timeblock_1.__repr__() == "Timeblock(start=Timestamp('2020-03-20 00:00:00'),end=Timestamp('2020-03-30 00:00:00'),timestamp=Timestamp('2020-03-20 00:00:00'),milestone=Milestone(label='Milestone 1',offset_before=5,offset_after=5,active=True))"


def test_merged_timeblock_repr():

    merged_timeblock_1 = MergedTimeblock(
        start=pd.Timestamp(year=2020, month=3, day=5), end=pd.Timestamp(year=2020, month=3, day=20),
        milestones=[
            Milestone('Milestone 1', offset_before=5, offset_after=5, active=True),
            Milestone('Milestone 2', offset_before=5, offset_after=5, active=True)
        ],
        timestamps=[
            pd.Timestamp(year=2020, month=3, day=10),
            pd.Timestamp(year=2020, month=3, day=15)
        ]
    )
    assert merged_timeblock_1.__repr__() == "Timeblock(start=Timestamp('2020-03-05 00:00:00'),end=Timestamp('2020-03-20 00:00:00'),timestamp=[Timestamp('2020-03-10 00:00:00'), Timestamp('2020-03-15 00:00:00')],milestone=[Milestone(label='Milestone 1',offset_before=5,offset_after=5,active=True), Milestone(label='Milestone 2',offset_before=5,offset_after=5,active=True)])"


@pytest.mark.parametrize(
    "input_gap, comparison_gap, expected_equality",
    [
        (
            Gap(
                start=pd.Timestamp(year=2023, month=2, day=1),
                end=pd.Timestamp(year=2023, month=2, day=10),
                timestamp_lst=set(
                    pd.date_range(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2023, month=2, day=10)
                    ).date
                )
            ),
            Gap(
                start=pd.Timestamp(year=2023, month=2, day=1),
                end=pd.Timestamp(year=2023, month=2, day=10),
                timestamp_lst=set(
                    pd.date_range(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2023, month=2, day=10)
                    ).date
                )
            ),
            True
        ),
        (
            Gap(
                start=pd.Timestamp(year=2023, month=2, day=1),
                end=pd.Timestamp(year=2021, month=2, day=10),
                timestamp_lst=set(
                    pd.date_range(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2021, month=2, day=10)
                    ).date
                )
            ),
            Gap(
                start=pd.Timestamp(year=2023, month=2, day=1),
                end=pd.Timestamp(year=2023, month=2, day=10),
                timestamp_lst=set(
                    pd.date_range(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2023, month=2, day=10)
                    ).date
                )
            ),
            False
        ),
        (
            Gap(
                start=pd.Timestamp(year=2023, month=2, day=1),
                end=pd.Timestamp(year=2023, month=2, day=10),
                timestamp_lst=set(
                    pd.date_range(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2021, month=2, day=10)
                    ).date
                )
            ),
            Gap(
                start=pd.Timestamp(year=2023, month=2, day=1),
                end=pd.Timestamp(year=2023, month=2, day=10),
                timestamp_lst=set(
                    pd.date_range(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2023, month=2, day=10)
                    ).date
                )
            ),
            True
        )
    ]
)
def test_gap_equality(input_gap, comparison_gap, expected_equality):

    equality = input_gap == comparison_gap
    assert equality == expected_equality


@pytest.mark.parametrize(
    "input_timeblock, comparison_timeblock, expected_equality",
    [
        (
            Timeblock(
                start=pd.Timestamp(year=2020, month=3, day=20), end=pd.Timestamp(year=2020, month=3, day=30),
                milestone=Milestone(
                    'Milestone 1', offset_before=5, offset_after=5, active=True
                ),
                timestamp=pd.Timestamp(year=2020, month=3, day=20)
            ),
            Timeblock(
                start=pd.Timestamp(year=2020, month=3, day=20), end=pd.Timestamp(year=2020, month=3, day=30),
                milestone=Milestone(
                    'Milestone 1', offset_before=5, offset_after=5, active=True
                ),
                timestamp=pd.Timestamp(year=2020, month=3, day=20)
            ),
            True
        ),
        (
            Timeblock(
                start=pd.Timestamp(year=2020, month=3, day=20), end=pd.Timestamp(year=2020, month=3, day=30),
                milestone=Milestone(
                    'Milestone 1', offset_before=5, offset_after=5, active=True
                ),
                timestamp=pd.Timestamp(year=2020, month=3, day=20)
            ),
            Timeblock(
                start=pd.Timestamp(year=2020, month=3, day=25), end=pd.Timestamp(year=2020, month=3, day=30),
                milestone=Milestone(
                    'Milestone 1', offset_before=5, offset_after=5, active=True
                ),
                timestamp=pd.Timestamp(year=2020, month=3, day=20)
            ),
            False
        ),
        (
            Timeblock(
                start=pd.Timestamp(year=2020, month=3, day=20), end=pd.Timestamp(year=2020, month=3, day=30),
                milestone=Milestone(
                    'Milestone 1', offset_before=5, offset_after=5, active=True
                ),
                timestamp=pd.Timestamp(year=2020, month=3, day=20)
            ),
            Timeblock(
                start=pd.Timestamp(year=2020, month=3, day=25), end=pd.Timestamp(year=2020, month=3, day=30),
                milestone=Milestone(
                    'Milestone 4', offset_before=5, offset_after=5, active=True
                ),
                timestamp=pd.Timestamp(year=2020, month=3, day=20)
            ),
            False
        ),
        (
            MergedTimeblock(
                start=pd.Timestamp(year=2020, month=3, day=5), end=pd.Timestamp(year=2020, month=3, day=20),
                milestones=[
                    Milestone('Milestone 1', offset_before=5, offset_after=5, active=True),
                    Milestone('Milestone 2', offset_before=5, offset_after=5, active=True)
                ],
                timestamps=[
                    pd.Timestamp(year=2020, month=3, day=10),
                    pd.Timestamp(year=2020, month=3, day=15)
                ]
            ),
            MergedTimeblock(
                start=pd.Timestamp(year=2020, month=3, day=5), end=pd.Timestamp(year=2020, month=3, day=20),
                milestones=[
                    Milestone('Milestone 1', offset_before=5, offset_after=5, active=True),
                    Milestone('Milestone 2', offset_before=5, offset_after=5, active=True)
                ],
                timestamps=[
                    pd.Timestamp(year=2020, month=3, day=10),
                    pd.Timestamp(year=2020, month=3, day=15)
                ]
            ),
            True
        ),
        (
            MergedTimeblock(
                start=pd.Timestamp(year=2020, month=3, day=5), end=pd.Timestamp(year=2020, month=3, day=20),
                milestones=[
                    Milestone('Milestone 1', offset_before=5, offset_after=5, active=True),
                    Milestone('Milestone 2', offset_before=5, offset_after=5, active=True)
                ],
                timestamps=[
                    pd.Timestamp(year=2020, month=3, day=10),
                    pd.Timestamp(year=2020, month=3, day=15)
                ]
            ),
            MergedTimeblock(
                start=pd.Timestamp(year=2020, month=3, day=3), end=pd.Timestamp(year=2020, month=3, day=20),
                milestones=[
                    Milestone('Milestone 1', offset_before=5, offset_after=5, active=True),
                    Milestone('Milestone 2', offset_before=5, offset_after=5, active=True)
                ],
                timestamps=[
                    pd.Timestamp(year=2020, month=3, day=10),
                    pd.Timestamp(year=2020, month=3, day=15)
                ]
            ),
            False
        ),
        (
            MergedTimeblock(
                start=pd.Timestamp(year=2020, month=3, day=5), end=pd.Timestamp(year=2020, month=3, day=20),
                milestones=[
                    Milestone('Milestone 1', offset_before=5, offset_after=5, active=True),
                    Milestone('Milestone 2', offset_before=5, offset_after=5, active=True)
                ],
                timestamps=[
                    pd.Timestamp(year=2020, month=3, day=10),
                    pd.Timestamp(year=2020, month=3, day=15)
                ]
            ),
            MergedTimeblock(
                start=pd.Timestamp(year=2020, month=3, day=3), end=pd.Timestamp(year=2020, month=3, day=20),
                milestones=[
                    Milestone('Milestone 1', offset_before=5, offset_after=5, active=True),
                    Milestone('Milestone 3', offset_before=5, offset_after=5, active=True)
                ],
                timestamps=[
                    pd.Timestamp(year=2020, month=3, day=10),
                    pd.Timestamp(year=2020, month=3, day=15)
                ]
            ),
            False
        )
    ]
)
def tst_timeblock_equality(input_timeblock, comparison_timeblock, expected_equality):

    equality = input_timeblock == comparison_timeblock
    assert equality == expected_equality


@pytest.mark.parametrize(
    "input_information, comparison_information, expected_equality",
    [
        (
            ItemInformation(
                study_id='test_study',
                gap_number=1,
                gap_day_total=10,
                gap_lst=[
                    Gap(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2023, month=2, day=10),
                        timestamp_lst=set(
                            pd.date_range(
                                start=pd.Timestamp(year=2023, month=2, day=1),
                                end=pd.Timestamp(year=2023, month=2, day=10)
                            ).date
                        )
                    )
                ],
                active_during_timeframe=False,
                status=ItemStatus.CLOSING_BEFORE_TIMEFRAME
            ),
            ItemInformation(
                study_id='test_study',
                gap_number=1,
                gap_day_total=10,
                gap_lst=[
                    Gap(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2023, month=2, day=10),
                        timestamp_lst=set(
                            pd.date_range(
                                start=pd.Timestamp(year=2023, month=2, day=1),
                                end=pd.Timestamp(year=2023, month=2, day=10)
                            ).date
                        )
                    )
                ],
                active_during_timeframe=False,
                status=ItemStatus.CLOSING_BEFORE_TIMEFRAME
            ),
            True
        ),
        (
            ItemInformation(
                study_id='test_study_1',
                gap_number=1,
                gap_day_total=10,
                gap_lst=[
                    Gap(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2023, month=2, day=10),
                        timestamp_lst=set(
                            pd.date_range(
                                start=pd.Timestamp(year=2023, month=2, day=1),
                                end=pd.Timestamp(year=2023, month=2, day=10)
                            ).date
                        )
                    )
                ],
                active_during_timeframe=False,
                status=ItemStatus.CLOSING_BEFORE_TIMEFRAME
            ),
            ItemInformation(
                study_id='test_study_2',
                gap_number=1,
                gap_day_total=10,
                gap_lst=[
                    Gap(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2023, month=2, day=10),
                        timestamp_lst=set(
                            pd.date_range(
                                start=pd.Timestamp(year=2023, month=2, day=1),
                                end=pd.Timestamp(year=2023, month=2, day=10)
                            ).date
                        )
                    )
                ],
                active_during_timeframe=False,
                status=ItemStatus.CLOSING_BEFORE_TIMEFRAME
            ),
            False
        ),
        (
            ItemInformation(
                study_id='test_study',
                gap_number=2,
                gap_day_total=15,
                gap_lst=[
                    Gap(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2023, month=2, day=10),
                        timestamp_lst=set(
                            pd.date_range(
                                start=pd.Timestamp(year=2023, month=2, day=1),
                                end=pd.Timestamp(year=2023, month=2, day=10)
                            ).date
                        )
                    ),
                    Gap(
                        start=pd.Timestamp(year=2023, month=2, day=6),
                        end=pd.Timestamp(year=2023, month=2, day=11),
                        timestamp_lst=set(
                            pd.date_range(
                                start=pd.Timestamp(year=2023, month=2, day=11),
                                end=pd.Timestamp(year=2023, month=2, day=16)
                            ).date
                        )
                    )
                ],
                active_during_timeframe=False,
                status=ItemStatus.CLOSING_BEFORE_TIMEFRAME
            ),
            ItemInformation(
                study_id='test_study',
                gap_number=1,
                gap_day_total=10,
                gap_lst=[
                    Gap(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2023, month=2, day=10),
                        timestamp_lst=set(
                            pd.date_range(
                                start=pd.Timestamp(year=2023, month=2, day=1),
                                end=pd.Timestamp(year=2023, month=2, day=10)
                            ).date
                        )
                    )
                ],
                active_during_timeframe=False,
                status=ItemStatus.CLOSING_BEFORE_TIMEFRAME
            ),
            False
        ),
        (
            ItemInformation(
                study_id='test_study',
                gap_number=1,
                gap_day_total=10,
                gap_lst=[
                    Gap(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2023, month=2, day=10),
                        timestamp_lst=set(
                            pd.date_range(
                                start=pd.Timestamp(year=2023, month=2, day=1),
                                end=pd.Timestamp(year=2023, month=2, day=10)
                            ).date
                        )
                    )
                ],
                active_during_timeframe=True,
                status=ItemStatus.CLOSING_DURING_TIMEFRAME
            ),
            ItemInformation(
                study_id='test_study',
                gap_number=1,
                gap_day_total=10,
                gap_lst=[
                    Gap(
                        start=pd.Timestamp(year=2023, month=2, day=1),
                        end=pd.Timestamp(year=2023, month=2, day=10),
                        timestamp_lst=set(
                            pd.date_range(
                                start=pd.Timestamp(year=2023, month=2, day=1),
                                end=pd.Timestamp(year=2023, month=2, day=10)
                            ).date
                        )
                    )
                ],
                active_during_timeframe=False,
                status=ItemStatus.CLOSING_BEFORE_TIMEFRAME
            ),
            False
        ),
    ]
)
def test_item_information_equality(input_information, comparison_information, expected_equality):

    equality = input_information == comparison_information
    assert equality == expected_equality
