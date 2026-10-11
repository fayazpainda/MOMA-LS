"""Environment smoke test: the scientific stack imports cleanly."""


def test_scientific_stack_imports():
    import numpy
    import pandas
    import matplotlib
    import scipy
    import tqdm

    for mod in (numpy, pandas, matplotlib, scipy, tqdm):
        assert mod is not None
