import pandas as pd
import utils
from dataclasses import dataclass
import os


@dataclass
class AppConfiguration:

    # Source data configuration
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(dir_path, "data/clean_input_data_reduced - v2.csv")
    day_first_dates = True  # Toggle between international and European date formats

    # Required source data column labels
    study_label = "Study"
    compound_label = "Compound"
    unique_identity_label = "DPN(Compound-Study)"

    # Default displayed timeframe
    timeframe_start = pd.Timestamp(year=2016, month=1, day=1)
    timeframe_end = pd.Timestamp(year=2026, month=1, day=1)

    # Source data milestone labels and default offsets
    milestone_type_1 = "Protocol Approval"
    milestone_type_2 = "FPFV"
    milestone_type_3 = "LPLV Final"
    milestone_type_4 = "Last DBL"
    milestone_type_5 = "Final CSR"

    milestone_definitions = {
        milestone_type_1: utils.Milestone(milestone_type_1),  # set custom default values here for milestones
        milestone_type_2: utils.Milestone(milestone_type_2),  # e.g. utils.Milestone("...", offset_before=2)
        milestone_type_3: utils.Milestone(milestone_type_3),
        milestone_type_4: utils.Milestone(milestone_type_4),
        milestone_type_5: utils.Milestone(milestone_type_5, offset_before=7, offset_after=7),
    }
