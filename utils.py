import pandas as pd
from itertools import tee, zip_longest
from dataclasses import dataclass
from typing import List, Set
from enum import Enum, unique


@dataclass
class Milestone:
    label: str
    offset_before: int = 14
    offset_after: int = 14
    active: bool = True

    def apply_offsets(self, timestamp: pd.Timestamp):
        """Returns a 2 item list with calculated before and after offsets from given timestamp."""
        return [
            timestamp - pd.to_timedelta(self.offset_before, unit="day"),
            timestamp + pd.to_timedelta(self.offset_after, unit="day")
        ]

    def __eq__(self, other):
        if isinstance(other, Milestone):
            return self.label == other.label and \
                   self.offset_before == other.offset_before and \
                   self.offset_after == other.offset_after and \
                   self.active is other.active

    def __repr__(self):
        return f"Milestone(" \
               f"label={repr(self.label)}," \
               f"offset_before={repr(self.offset_before)}," \
               f"offset_after={repr(self.offset_after)}," \
               f"active={repr(self.active)}" \
               f")"


class TimeblockBase:
    is_timeframe_start = False
    is_timeframe_end = False

    def __init__(self, start: pd.Timestamp, end: pd.Timestamp):
        self.start: pd.Timestamp = start
        self.end: pd.Timestamp = end
        self.timestamp = NotImplemented
        self.milestone = NotImplemented

    def __eq__(self, other):
        if isinstance(other, TimeblockBase):
            return self.start == other.start and \
                self.end == other.end and \
                self.milestone == other.milestone

    def __repr__(self):
        return f"Timeblock(" \
               f"start={repr(self.start)}," \
               f"end={repr(self.end)}," \
               f"timestamp={repr(self.timestamp)}," \
               f"milestone={repr(self.milestone)}" \
               f")"


class Timeblock(TimeblockBase):

    def __init__(self, start: pd.Timestamp, end: pd.Timestamp, timestamp: pd.Timestamp,
                 milestone: Milestone):
        super().__init__(start, end)
        self.timestamp: pd.Timestamp = timestamp
        self.milestone: Milestone = milestone


class MergedTimeblock(TimeblockBase):

    def __init__(self, start: pd.Timestamp, end: pd.Timestamp, timestamps: List[pd.Timestamp],
                 milestones: List[Milestone]):
        super().__init__(start, end)
        self.timestamp: List[pd.Timestamp] = timestamps
        self.milestone: List[Milestone] = milestones


@unique
class ItemStatus(str, Enum):
    
    CLOSING_BEFORE_TIMEFRAME = 'Closing Before Timeframe'
    STARTING_AFTER_TIMEFRAME = 'Starting After Timeframe'
    NO_ACTIVITY_OVER_TIMEFRAME = 'No Activity Over Timeframe'

    CLOSING_AFTER_TIMEFRAME = 'Closing After Timeframe'
    CLOSING_DURING_TIMEFRAME = 'Closing During Timeframe'

    NO_STATUS_GIVEN = 'No Status Given'


@dataclass
class Gap:
    start: pd.Timestamp
    end: pd.Timestamp
    timestamp_lst: Set[pd.Timestamp]

    def __eq__(self, other):
        if isinstance(other, Gap):
            return self.start == other.start and \
                self.end == other.end

    def __repr__(self):
        return f"Gap(" \
               f"start={repr(self.start)}," \
               f"end={repr(self.end)}"  \
               f")"


@dataclass
class ItemInformation:
    """
    Used to store useful information about a specific item ("study usually") in the data.
    """
    study_id: str
    gap_number: int
    gap_day_total: int
    gap_lst: List[Gap]

    active_during_timeframe: bool
    status: ItemStatus

    def __eq__(self, other):
        if isinstance(other, ItemInformation):
            return self.gap_number == other.gap_number and \
                self.gap_day_total == other.gap_day_total and \
                self.gap_lst == other.gap_lst and \
                self.active_during_timeframe == other.active_during_timeframe and \
                self.status == other.status and \
                self.study_id == other.study_id

    def __repr__(self):
        return f"ItemInformation(" \
               f"study_id={repr(self.study_id)}, " \
               f"gap_number={repr(self.gap_number)}, " \
               f"gap_day_total={repr(self.gap_day_total)}, " \
               f"gap_lst={repr(self.gap_lst)}, " \
               f"active_during_timeframe={repr(self.active_during_timeframe)}, " \
               f"status={repr(self.status)}" \
               f")"


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip_longest(a, b)
