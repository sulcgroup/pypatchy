from pathlib import Path
from datetime import datetime

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
    simulation: object
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
                 simulation: object,
                 script_path: Path,
                 log_path: Path,
                 notes: str = "",
                 additional_metadata: dict = {},
                 start_date: datetime = datetime.now()):
        self.job_submit_date = start_date
        self.job_type = job_type
        self.job_id = pid
        self.simulation = simulation
        self.script_path = script_path
        self.log_path = log_path
        self.notes = notes
        self.additional_metadata = additional_metadata

    def get_log_txt(self) -> str:
        try:
            with self.log_path.open("r") as f:
                return f.read()
        except FileNotFoundError:
            return f"No file {str(self.log_path)}"
