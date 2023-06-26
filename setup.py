from setuptools import setup, find_packages

setup(
    name="pypatchy",
    version="0.1",
    packages=["pypatchy"],
    author="Joshua Evans",
    author_email="jrevan21@asu.edu",
    package_data={
        "pypatchy": ["spec_files/*/*.json"]
    }
    # TODO: more
)
# TODO: compile TLM? i guess
