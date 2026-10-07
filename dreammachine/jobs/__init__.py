"""Job system: one pipeline from baseline to comparison (PLAN.md §5, §7).

    requests.create_job_from_request   validate + resolve + store a job and its steps (queued)
    execute.execute_step                run one step (the worker runs each in a subprocess)
    execute.run_job                     run a whole job in the foreground (development)
"""

from .errors import JobError
from .execute import execute_step, finish_job, run_job
from .requests import create_job_from_request

__all__ = ["JobError", "create_job_from_request", "execute_step", "finish_job", "run_job"]
