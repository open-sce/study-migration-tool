import pandas as pd

import utils


def test_apply_offsets():
    milestone_1 = utils.Milestone('Test Milestone')
    test_timestamp_1 = pd.Timestamp(year=2020, month=3, day=20)

    res_1 = milestone_1.apply_offsets(test_timestamp_1)
    assert res_1[0] == pd.Timestamp(year=2020, month=3, day=6), \
        "milestone_1 apply offset [0] expected to be 14 days (default) BEFORE test_timestamp_1!"

    assert res_1[1] == pd.Timestamp(year=2020, month=4, day=3), \
        "milestone_1 apply offset [1] expected to be 14 days (default) AFTER test_timestamp_1!"

    milestone_2 = utils.Milestone('Test Milestone 2', offset_before=20, offset_after=42)

    res_2 = milestone_2.apply_offsets(test_timestamp_1)
    assert res_2[0] == pd.Timestamp(year=2020, month=2, day=29), \
        "milestone_2 apply offset [0] expected to be offset_before days BEFORE test_timestamp_1!"

    assert res_2[1] == pd.Timestamp(year=2020, month=5, day=1), \
        "milestone_2 apply offset [1] expected to be offset_after AFTER test_timestamp_1!"


def test_milestone_equality():

    milestone_1 = utils.Milestone('Test Milestone')
    milestone_2 = utils.Milestone('Test Milestone')

    assert milestone_1 == milestone_2, \
        "milestone_1 and milestone_2 are expected to be equal!"

    milestone_3 = utils.Milestone('Test Milestone', offset_before=1, offset_after=1)
    milestone_4 = utils.Milestone('Test Milestone', offset_before=1, offset_after=2)

    assert milestone_3 != milestone_4, \
        "milestone_3 and milestone_4 are expected to NOT be equal!"

    milestone_5 = utils.Milestone('Test Milestone', offset_before=1, offset_after=2)
    milestone_6 = utils.Milestone('Test Milestone', offset_before=1, offset_after=2)

    assert milestone_5 == milestone_6, \
        "milestone_4 and milestone_5 are expected to be equal!"

    milestone_7 = utils.Milestone('Test Milestone', offset_before=1, offset_after=2, active=False)

    assert milestone_6 != milestone_7, \
        "milestone_6 and milestone_7 are expected to NOT be equal!"


def test_milestone_repr():
    milestone_1 = utils.Milestone('Test Milestone', offset_before=1, offset_after=2, active=False)
    assert milestone_1.__repr__() == f"Milestone(label='Test Milestone',offset_before=1,offset_after=2,active=False)", \
        "milestone_1 __repr__ did not follow expected format!"
