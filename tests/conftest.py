import torch


def pytest_sessionstart(session):
    # Small CPU fixtures otherwise spend more time launching worker threads.
    torch.set_num_threads(2)
