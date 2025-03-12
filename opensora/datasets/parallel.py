import multiprocessing
from itertools import count
from multiprocessing.managers import SyncManager
from typing import Any, Callable, Dict, Tuple, Type, cast

import dill
import pandarallel
import pandas as pd
from pandarallel.data_types import DataType
from pandarallel.progress_bars import ProgressBarsType, get_progress_bars, progress_wrapper
from pandarallel.utils import WorkerStatus

CONTEXT = multiprocessing.get_context("fork")
TMP = []


class WrapWorkFunctionForPipe:
    def __init__(
        self,
        work_function: Callable[
            [
                Any,
                Callable,
                tuple,
                Dict[str, Any],
                Dict[str, Any],
            ],
            Any,
        ],
    ) -> None:
        self.work_function = work_function

    def __call__(
        self,
        progress_bars_type: ProgressBarsType,
        worker_index: int,
        master_workers_queue: multiprocessing.Queue,
        dilled_user_defined_function: bytes,
        user_defined_function_args: tuple,
        user_defined_function_kwargs: Dict[str, Any],
        extra: Dict[str, Any],
    ) -> Any:
        try:
            data = TMP[worker_index]
            data_size = len(data)
            user_defined_function: Callable = dill.loads(dilled_user_defined_function)

            progress_wrapped_user_defined_function = progress_wrapper(
                user_defined_function, master_workers_queue, worker_index, data_size
            )

            used_user_defined_function = (
                progress_wrapped_user_defined_function
                if progress_bars_type
                in (
                    ProgressBarsType.InUserDefinedFunction,
                    ProgressBarsType.InUserDefinedFunctionMultiplyByNumberOfColumns,
                )
                else user_defined_function
            )

            results = self.work_function(
                data,
                used_user_defined_function,
                user_defined_function_args,
                user_defined_function_kwargs,
                extra,
            )

            master_workers_queue.put((worker_index, WorkerStatus.Success, None))

            return results

        except:
            master_workers_queue.put((worker_index, WorkerStatus.Error, None))
            raise


def parallelize_with_pipe(
    nb_requested_workers: int,
    data_type: Type[DataType],
    progress_bars_type: ProgressBarsType,
):
    def closure(
        data: Any,
        user_defined_function: Callable,
        *user_defined_function_args: tuple,
        **user_defined_function_kwargs: Dict[str, Any],
    ):
        wrapped_work_function = WrapWorkFunctionForPipe(data_type.work)
        dilled_user_defined_function = dill.dumps(user_defined_function)
        manager: SyncManager = CONTEXT.Manager()
        master_workers_queue = manager.Queue()

        chunks = list(
            data_type.get_chunks(
                nb_requested_workers,
                data,
                user_defined_function_kwargs=user_defined_function_kwargs,
            )
        )
        TMP.extend(chunks)

        nb_workers = len(chunks)

        multiplicator_factor = (
            len(cast(pd.DataFrame, data).columns)
            if progress_bars_type == ProgressBarsType.InUserDefinedFunctionMultiplyByNumberOfColumns
            else 1
        )

        progresses_length = [len(chunk_) * multiplicator_factor for chunk_ in chunks]

        work_extra = data_type.get_work_extra(data)
        reduce_extra = data_type.get_reduce_extra(data, user_defined_function_kwargs)

        show_progress_bars = progress_bars_type != ProgressBarsType.No

        progress_bars = get_progress_bars(progresses_length, show_progress_bars)
        progresses = [0] * nb_workers
        workers_status = [WorkerStatus.Running] * nb_workers

        work_args_list = [
            (
                progress_bars_type,
                worker_index,
                master_workers_queue,
                dilled_user_defined_function,
                user_defined_function_args,
                user_defined_function_kwargs,
                {
                    **work_extra,
                    **{
                        "master_workers_queue": master_workers_queue,
                        "show_progress_bars": show_progress_bars,
                        "worker_index": worker_index,
                    },
                },
            )
            for worker_index in range(nb_workers)
        ]

        pool = CONTEXT.Pool(nb_workers)
        results_promise = pool.starmap_async(wrapped_work_function, work_args_list)
        pool.close()

        generation = count()

        while any((worker_status == WorkerStatus.Running for worker_status in workers_status)):
            message: Tuple[int, WorkerStatus, Any] = master_workers_queue.get()
            worker_index, worker_status, payload = message
            workers_status[worker_index] = worker_status

            if worker_status == WorkerStatus.Success:
                progresses[worker_index] = progresses_length[worker_index]
                progress_bars.update(progresses)
            elif worker_status == WorkerStatus.Running:
                progress = cast(int, payload)
                progresses[worker_index] = progress

                if next(generation) % nb_workers == 0:
                    progress_bars.update(progresses)
            elif worker_status == WorkerStatus.Error:
                progress_bars.set_error(worker_index)

        results = results_promise.get()
        TMP.clear()

        return data_type.reduce(results, reduce_extra)

    return closure


pandarallel.core.WrapWorkFunctionForPipe = WrapWorkFunctionForPipe
pandarallel.core.parallelize_with_pipe = parallelize_with_pipe
pandarallel = pandarallel.pandarallel
