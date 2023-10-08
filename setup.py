from setuptools import setup, find_packages

setup(
    name="pypatchy",
    version="0.1.2",
    packages=find_packages(),
    author="Joshua Evans",
    author_email="jrevan21@asu.edu",
    package_data={
        "pypatchy": ["spec_files/*/*.json"]
    }
    # TODO: more
)
# TODO: compile TLM? i guess
