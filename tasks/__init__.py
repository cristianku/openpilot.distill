"""The jobs of openpilot.distill as Python modules: what the web site starts (python -m tasks.<name>).

Every task validates its own arguments, prints its progress in Italian on stdout (the site keeps it
as the job log) and ends with exit code 0 only when the work is done. A GPU task holds the GPU lock
shared with the other jobs of the server for its whole life (tasks/common.py).
"""
