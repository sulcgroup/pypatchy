# importing module should run tests
# not true of any others
import test_prerequisites

# test server configs
from pypatchy.server_config import get_server_config
from test_server_configs import test_server_configs, test_config
test_config(get_server_config())

# test patchy IO
from test_patchy_io import *
test_fwriter()
test_lwriter()
test_swriter()

# test ensembles
from test_ensembles import test_ensembles
test_ensembles(get_server_config())

# todo: test analytics

# todo: test polycubes lib stuff

# todo: test patchy -> origami convert

# todo: test solver stuff
