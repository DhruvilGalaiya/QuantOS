# QuantOS MLOps

## Automated validation

The GitHub Actions workflow runs input validation tests and compiles the HMM experiment script.

## Local test command

Run the following command from the repository root:

    python -m unittest discover -s mlops/tests -p 'test_*.py' -v
