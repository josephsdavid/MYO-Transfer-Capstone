# get the best result from keras tuner in a new file
import os
import json
import kerastuner.engine.hyperparameters as hp_module
import kerastuner.engine.trial as trial_module
import kerastuner.engine.metrics_tracking as metrics_tracking
from kerastuner.abstractions.tensorflow import TENSORFLOW_UTILS as tf_utils
import tensorflow as tf
from tensorflow.keras.models import model_from_json

class SearchResults(object):
    def __init__(self, directory, project_name, objective):
        self.directory = directory
        self.project_name = project_name
        self.objective = objective

    def reload(self):
        """Populate `self.trials` and `self.oracle` state."""
        fname = os.path.join(self.directory, self.project_name, 'tuner.json')
        state_data = tf_utils.read_file(fname)
        state = json.loads(state_data)

        self.hyperparameters = hp_module.HyperParameters.from_config(
            state['hyperparameters'])
        self.best_metrics = metrics_tracking.MetricsTracker.from_config(
            state['best_metrics'])
        self.trials = [trial_module.Trial.load(f) for f in state['trials']]
        self.start_time = state['start_time']

    def _get_best_trials(self, num_trials=1):
        if not self.best_metrics.exists(self.objective):
            return []
        trials = []
        for x in self.trials:
            if x.score is not None:
                trials.append(x)
        if not trials:
            return []
        direction = self.best_metrics.directions[self.objective]
        sorted_trials = sorted(trials,
                               key=lambda x: x.score,
                               reverse=direction == 'max')
        return sorted_trials[:num_trials]

    def get_best_models(self, num_models = 1):
        best_trials = self._get_best_trials(num_models)
        models = []
        for trial in best_trials:
            hp = trial.hyperparameters.copy()
            # Get best execution.
            direction = self.best_metrics.directions[self.objective]
            executions = sorted(
                trial.executions,
                key=lambda x: x.per_epoch_metrics.get_best_value(
                    self.objective),
                reverse=direction == 'max')

            # Reload best checkpoint.
            ckpt = executions[0].best_checkpoint
            model_graph = ckpt + '-config.json'
            model_wts = ckpt + '-weights.h5'
            with open(model_graph, 'r') as f:
                model = model_from_json(f.read())
            model.load_weights(model_wts)
            models.append(model)
        return models
