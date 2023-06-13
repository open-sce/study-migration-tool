from itertools import tee, zip_longest
from dataclasses import dataclass
import pandas as pd


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


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip_longest(a, b)
