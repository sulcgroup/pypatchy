from __future__ import annotations

import datetime
from typing import Union, Optional

from pypatchy.slurm_log_entry import SlurmLogEntry, LogEntryObject


class SlurmLog:
    """
    Wrapper class for a list of SlurmLogEntry objects, which provides a lot of helpful accessor methods
    """
    # list of Slurm Log Entries, sorted by date
    log_list: list[SlurmLogEntry]
    id_map: dict[int, SlurmLogEntry]

    def __init__(self, *args: SlurmLogEntry):
        """
        Parameters:
            args (optional) list of log entry objects
        """

        self.log_list = sorted(args, key=lambda x: x.job_submit_date)
        self.id_map = {
            x.job_id: x for x in self.log_list
        }

    def __getitem__(self, key: Union[int, datetime.date, slice]) -> Union[list[SlurmLogEntry], SlurmLogEntry]:
        """
        Accessor function. Provided an int index, a date, or a slice (of ints or dates),
        returns the log entry or entries specified

        Returns:
            a SlurmLogEntry or list of SlurmLogEntry objects

        """
        if isinstance(key, int):
            assert -1 < key < len(self), f"Index {key} out of bounds for list length {len(self)}" 
            return self.log_list[key]
        if isinstance(key, datetime.date):
            start = self.idx_begin(key)
            stop = self.idx_end(key)
            if start is not None and stop is not None:
                return self[start:stop]
        if isinstance(key, slice):
            if isinstance(key.start, int):
                return self.log_list[key]
            else:
                # key is a date
                return self.log_list[self.idx_begin(key.start):self.idx_end(key.stop)]

    def idx_begin(self,
                  key: datetime.date,
                  begin: int = 0,
                  end: int = -1) -> Union[int, None]:
        """
        Uses a modified version of a binary search to find the first entry for the provided date

        Args:
            key: the date to search for
            begin: the index in the log to start searching
            end: the index in the log to stop searching

        Returns:
            the index of the first position in the log where the date matches the provided date,
            or None if the index provided was out of bounds
        """
        # if no end is provided, use last entry
        if end == -1:
            end = len(self.log_list) - 1
        # handle out of bounds
        if self.log_list[begin].job_submit_date > key:
            return None
        if self.log_list[end].job_submit_date < key:
            return None
        # if end and begin are the same, we've found our startpoint
        if end == begin:
            return begin
        else:
            # find midpoint
            midpoint = int((end + begin) / 2)
            # if the midpoint is before or equal to our key
            if self.log_list[midpoint].job_submit_date <= key:
                # compute slice from begin to midpoint
                return self.idx_begin(key, begin, midpoint)
            else:
                # slice from midpoint to end
                return self.idx_begin(key, midpoint, end)

    def idx_end(self,
                key: datetime.date,
                begin: int = 0,
                end: int = -1) -> Union[int, None]:
        """
        Uses a modified version of a binary search to find the last entry for the provided date

        Args:
            key: the date to search for
            begin: the index in the log to start searching
            end: the index in the log to stop searching

        Returns:
            the index of the last position in the log where the date matches the provided date,
            or None if the index provided was out of bounds
        """
        # if no end is provided, use last entry
        if end == -1:
            end = len(self.log_list) - 1
        # handle out of bounds
        if self.log_list[begin].job_submit_date > key:
            return None
        if self.log_list[end].job_submit_date < key:
            return None
        # if end and begin are the same, we've found our startpoint
        if end == begin:
            return begin
        else:
            # find midpoint
            midpoint = int((end + begin) / 2)
            # if the midpoint is after or at our key
            if self.log_list[midpoint].job_submit_date >= key:
                # slice from midpoint to end
                return self.idx_begin(key, midpoint, end)
            else:
                # compute slice from begin to midpoint
                return self.idx_begin(key, begin, midpoint)

    def find_idx(self, key: datetime.datetime):
        """
        Finds the index in self.log_list where the next entry has a start
        date which is after the key, and the previous entry has a start date before the key
        normal binary search
        this time I was lazy and had chatGPT write this code
        """

        def binary_search(l: int, r: int) -> Optional[int]:
            if l <= r:
                mid = l + (r - l) // 2

                if self.log_list[mid].job_submit_date == key:
                    return mid

                # If the current entry is less than the key and the next entry is greater than the key, return the next index.
                if (mid + 1 < len(self.log_list) and
                        self.log_list[mid].job_submit_date < key < self.log_list[mid + 1].job_submit_date):
                    return mid + 1

                # If the current entry is greater than the key and the previous entry is less than the key, return the current index.
                elif (mid - 1 >= 0 and
                      self.log_list[mid - 1].job_submit_date < key < self.log_list[mid].job_submit_date):
                    return mid

                # If the current entry is greater than the key, search the left half.
                elif self.log_list[mid].job_submit_date > key:
                    return binary_search(l, mid - 1)

                # If the current entry is less than the key, search the right half.
                else:
                    return binary_search(mid + 1, r)

            # If the element is not present in the array.
            else:
                return None

        return binary_search(0, len(self.log_list) - 1)

    def by_id(self, pid: int) -> SlurmLogEntry:
        """
        Finds the log entry with the provided id
        Returns:
            a slurm log entry
        """
        return self.id_map[pid]

    def by_type(self, entry_type: Union[str, list[str]]) -> SlurmLog:
        """
        Filters the log by type, returning a log containing only the entries that match
        the provided type.
        Args:
            entry_type: a string or list of strings that are entry type names, e.g. "oxdna".

        Returns:
            a slurm log containing all entries matching the specified entry type
        """
        if isinstance(entry_type, str):
            return SlurmLog(*[x for x in self.log_list if x.job_type == entry_type])
        else:
            return SlurmLog(*[x for x in self.log_list if x.job_type in entry_type])

    def by_entry_subject(self, subject: LogEntryObject) -> SlurmLog:
        """
        Filters the slurm log by simulation

        Args:
            subject: the subject to filter by

        Return:
            a SlurmLog object containing only the entries where the log entry subject matches the provided subject
        """
        return SlurmLog(*[x for x in self.log_list if x.simulation == subject])

    def by_other(self, key: str, value) -> SlurmLog:
        """
        Filters the slurm log by some other key, as included in SlurmLogEntry.additional_metadata

        Args:
            key: a string, the key to filter by
            value: the value to filter with. can be any data type.

        Returns:
            a new SlurmLog object containing only the slurm log entries where entry.additonal_metadata[key] == value
        """
        return SlurmLog(*[x for x in self.log_list
                          if key in x.additional_metadata and x.additional_metadata[key] == value])

    def append(self, e: SlurmLogEntry):
        """
        Appends a new entry to the slurm log

        Args:
            e: the new entry
        """
        # if our log is empty or this new entry is after the last entry, this is easy
        if len(self.log_list) == 0 or e.job_submit_date > self.log_list[-1].job_submit_date:
            self.log_list.append(e)
        else: # *sigh*
            self.log_list.insert(self.find_idx(e.job_submit_date) + 1, e)

        # if we assume linear tim

    def __len__(self):
        """
        Returns:
            the number of entries in the slurm log
        """
        return len(self.log_list)

    def to_list(self):
        """
        Returns:
            a list containing all log entries, where each log entry is converted to a dict

        """
        return [e.to_dict() for e in self.log_list]

    def __iter__(self):
        return iter(self.log_list)

    def __repr__(self):
        return "\n\n".join(str(entry for entry in self))