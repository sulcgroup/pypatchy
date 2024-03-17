from setuptools import setup, find_packages

setup(
    name="pypatchy",
    version="0.2.2",
    packages=find_packages(),
    install_require=[
        'ipy_oxdna @ git+https://github.com/mlsample/ipy_oxDNA.git'
    ],
    author="Joshua Evans",
    author_email="jrevan21@asu.edu",
    package_data={
        "pypatchy": ["spec_files/*/*.json"]
    },
    entrypoints={
        # TODO
    }
    # TODO: more
)
# TODO: compile TLM? i guess
