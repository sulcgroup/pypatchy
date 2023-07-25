from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Union


class LogEntryObject(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Union[str, int, float]]:
        pass


class SlurmLogEntry:
    """
    I've written this class to be deliberately flexable while also
    specifically for Patchy Simulations
    """
    # the date on which this slurm job was submitted
    job_submit_date: datetime
    # slurm job ID
    job_id: int
    # an object for this simulaion.
    simulation: LogEntryObject
    # path to the shell script that runs this job
    script_path: Path
    # this variable is provided to store extra text at user's disgression
    notes: str
    # this attribute is included to make sorting log entries easier. possible values depend on use case
    job_type: str
    # path to the file
    log_path: Path
    additional_metadata: dict

    def __init__(self,
                 job_type: str,
                 pid: int,
                 simulation: LogEntryObject,
                 script_path: Union[Path, str],
                 log_path: Union[Path, str],
                 notes: str = "",
                 additional_metadata: dict = {},
                 start_date: Union[str, datetime] = datetime.now()):
        if isinstance(start_date, datetime):
            self.job_submit_date = start_date
        else:
            self.job_submit_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.job_type = job_type
        self.job_id = pid
        self.simulation = simulation
        if isinstance(script_path, Path):
            self.script_path = script_path
        else:
            self.script_path = Path(script_path)
        if isinstance(log_path, Path):
            self.log_path = log_path
        else:
            self.log_path = Path(log_path)
        self.notes = notes
        self.additional_metadata = additional_metadata

    def get_log_txt(self) -> str:
        try:
            with self.log_path.open("r") as f:
                return f.read()
        except FileNotFoundError:
            return f"No file {str(self.log_path)}"

    def to_dict(self):
        return {
            "job_type": self.job_type,
            "pid": self.job_id,
            "simulation": self.simulation.to_dict(),
            "script_path": str(self.script_path),
            "log_path": str(self.log_path),
            "notes": self.notes,
            "additional_metadata": self.additional_metadata,
            "start_date": self.job_submit_date.strftime("%Y-%m-%d")
        }

    def __str__(self):
        metadata_str = '\n'.join([f"\t\t{key}: {value}" for key, value in self.additional_metadata.items()])
        return "Slurm log entry:\n" \
               f"\tjob type: {self.job_type}\n" \
               f"\tslurm job ID: {self.job_id}\n" \
               f"\tsimulation: {str(self.simulation)}\n" \
               f"\tscript path: {str(self.script_path)}\n" \
               f"\tlog path: {str(self.log_path)}\n" \
               f"\tnotes: {self.notes}" \
               f"\tsubmit date: {self.job_submit_date.strftime('%Y-%m-%d')}\n" \
               f"\tadditional metadata: {metadata_str}"
