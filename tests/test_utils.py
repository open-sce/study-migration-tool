import pandas as pd

import utils


class NullContext:
    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


def test_apply_offsets(record_xml_attribute):
    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", "PYT18")
    record_xml_attribute("srs_requirement", "SRSREQ4")
    record_xml_attribute("frs_requirement", "FRSREQ6")
    record_xml_attribute("scenario", "OQSCE1")

    record_xml_attribute("purpose", "Test milestone offset calculation.")
    record_xml_attribute("description",
                         "Create a milestone and run method apply_offsets to apply that milestone to a timestamp.")
    record_xml_attribute("acceptance_criteria",
                         "Assert timestamps for offset_before and offset_after are equal to expected values.")

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


def test_milestone_equality(record_xml_attribute):
    record_xml_attribute("qualification", "oq")
    record_xml_attribute("test_id", "PYT19")
    record_xml_attribute("srs_requirement", "SRSREQ4")
    record_xml_attribute("frs_requirement", "FRSREQ6")
    record_xml_attribute("scenario", "OQSCE1")

    record_xml_attribute("purpose", "Test milestone offset equality.")
    record_xml_attribute("description",
                         "Create a set of milestones and compare against each other for equality.")
    record_xml_attribute("acceptance_criteria",
                         "Assert that expected milestone equality is true or false.")

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


def test_milestone_repr(record_xml_attribute):
    record_xml_attribute("qualification", "dq")
    record_xml_attribute("test_id", "PYT20")
    record_xml_attribute("srs_requirement", "SRSREQ4")
    record_xml_attribute("frs_requirement", "FRSREQ6")
    record_xml_attribute("scenario", "OQSCE1")

    record_xml_attribute("purpose", "Test milestone offset log formatting")
    record_xml_attribute("description",
                         "Creates a test milestone and invokes the __repr__ dunder method, comparing the "
                         "result against an expected value.")
    record_xml_attribute("acceptance_criteria",
                         "Assert that milestone_1.__repr__() matches the expected string representation.")

    milestone_1 = utils.Milestone('Test Milestone', offset_before=1, offset_after=2, active=False)
    assert milestone_1.__repr__() == "Milestone(label='Test Milestone',offset_before=1,offset_after=2,active=False)", \
        "milestone_1 __repr__ did not follow expected format!"
