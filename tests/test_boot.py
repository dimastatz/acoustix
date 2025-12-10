""" tests for boot module """

from sonix import __version__


def test_version():
    """test version import"""
    assert __version__ == "0.1.2"
