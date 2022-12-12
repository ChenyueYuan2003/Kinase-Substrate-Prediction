import nni

search_space = {

    "batch_size": {"_type": "choice", "_value": [32, 64, 128, 256, 512]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    "activation": {"_type": "choice", "_value": ['relu', 'tanh', 'sigmoid']}

}

from nni.experiment import Experiment


experiment = Experiment('local')
experiment.config.experiment_name = 'ksp_trial'
experiment.config.trial_command = 'python model_trial.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.max_trial_number = 800
experiment.config.max_experiment_duration = '5h'
experiment.config.trial_concurrency = 2
experiment.config.trial_gpu_number = 1
experiment.config.training_service.use_active_gpu = True
experiment.config.training_service.max_trial_number_per_gpu = 2

experiment.run(8080)
input()
