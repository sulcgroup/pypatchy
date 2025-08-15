# TODO: ANY ALTERNATERIVE TO THIS GOD PLEASE
import os
import stat


from pypatchy.patchy.simulation_ensemble import PatchySimulationEnsemble

try:
    from paramiko.sftp_client import SFTPClient
    from paramiko.client import SSHClient

except ImportError as e:
    print("`paramiko` python package not installed! `paramiko` is not required for pypatchy as a whole but is for the test scrits")
    raise e

SERVER_IP = "10.126.22.12"
TEST_USER_NAME = "testdata_user"
# extaordinarily bad practive to store password here but this user acct has like
# no privelages
TEST_USER_PWD = "test_user"


# for some incomprehensible reason paramiko doesn't already have this functionality????
# maybe it's some kind of strange security issue
# code by chatGPT
def download_directory(sftp, remote_dir, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    for filename in sftp.listdir(remote_dir):
        remote_path = f"{remote_dir}/{filename}"
        local_path = os.path.join(local_dir, filename)

        if is_file(sftp, remote_path):
            sftp.get(remote_path, local_path)
        else:
            download_directory(sftp, remote_path, local_path)


def is_file(sftp: SFTPClient, path: str):
    try:
        return stat.S_ISREG(sftp.stat(path).st_mode)
    except IOError:
        return False

def get_data(e: PatchySimulationEnsemble, copy_to: str):
    ssh = SSHClient()
    ssh.load_system_host_keys()
    ssh.connect(SERVER_IP, username=TEST_USER_NAME, password=TEST_USER_PWD)

    with ssh.open_sftp() as sftp:
        sftp.chdir("./pypatchy_test_data")
        download_directory(sftp, f"{e.long_name()}", copy_to)
