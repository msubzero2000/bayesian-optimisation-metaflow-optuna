import os
import uuid


class DistributedOptunaInfo(object):

    def __init__(self, study_name, storage, workers_info):
        self.study_name = study_name
        self.storage = storage
        self.workers_info = workers_info


class DistributedOptunaWorkerInfo(object):

    def __init__(self, worker_id, study_name, num_trials, timeout, storage):
        self.worker_id = worker_id
        self.study_name = study_name
        self.storage = storage
        self.num_trials = num_trials
        self.timeout = timeout


class DistributedOptuna(object):
    #TODO: This should be read from the metaflow config
    #One shared Postgresql database should be created during metaflow setup
    DISTRIBUTED_STORAGE_URL = "postgresql://username:password@postgreshost:5432"

    @staticmethod
    def create(num_workers=4, num_trials=20, timeout=600, study_name_prefix=None):
        workers_info = []
        if study_name_prefix is None:
            study_name_prefix = "distributed"

        study_name = study_name_prefix + f"-{str(uuid.uuid4())}"

        for i in range(num_workers):
            workers_info.append(DistributedOptunaWorkerInfo(worker_id=i,
                                                            study_name=study_name,
                                                            num_trials=num_trials,
                                                            timeout=timeout,
                                                            storage=DistributedOptuna.DISTRIBUTED_STORAGE_URL))

        distributed_optuna_info = DistributedOptunaInfo(study_name=study_name,
                                                  storage=DistributedOptuna.DISTRIBUTED_STORAGE_URL,
                                                  workers_info=workers_info)

        return distributed_optuna_info
