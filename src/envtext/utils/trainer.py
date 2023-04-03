import torch
from typing import Dict, Union , List ,Optional
from transformers import Trainer,TrainingArguments, TrainerState, TrainerControl, PrinterCallback

from transformers.utils.notebook import NotebookProgressCallback
from transformers.file_utils import is_in_notebook
from transformers.trainer_utils import IntervalStrategy, has_length

import re
# from copy import deepcopy

# class EnvBERTTrainer(Trainer):
#     def log(self, logs: Dict[str,Union[float , List[float] , List[List[float]]]]) -> None:
#         """
#         Log :obj:`logs` on the various objects watching training.

#         Subclass and override this method to inject custom behavior.

#         Args:
#             logs (:obj:`Dict[str,  List[float] , List[List[float]]]]`):
#                 The values to log.
#                 float: metrics for hold dataset
#                 List[float]: metrics for each texts
#                 List[List[float]] : matrix of metrics for whole dataset
#         """
#         if self.state.epoch is not None:
#             logs["epoch"] = round(self.state.epoch, 2)

#         output = {**logs, **{"step": self.state.global_step}}
#         self.state.log_history.append(output)
#         self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)



#     def evaluate(
#         self,
#         eval_dataset: Optional[Dataset] = None,
#         ignore_keys: Optional[List[str]] = None,
#         metric_key_prefix: str = "eval",
#     ) -> Dict[str, float]:
#         """
#         Run evaluation and returns metrics.

#         The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
#         (pass it to the init :obj:`compute_metrics` argument).

#         You can also subclass and override this method to inject custom behavior.

#         Args:
#             eval_dataset (:obj:`Dataset`, `optional`):
#                 Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
#                 columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
#                 :obj:`__len__` method.
#             ignore_keys (:obj:`Lst[str]`, `optional`):
#                 A list of keys in the output of your model (if it is a dictionary) that should be ignored when
#                 gathering predictions.
#             metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
#                 An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
#                 "eval_bleu" if the prefix is "eval" (default)

#         Returns:
#             A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
#             dictionary also contains the epoch number which comes from the training state.
#         """
#         # memory metrics - must set up as early as possible
#         self._memory_tracker.start()

#         if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
#             raise ValueError("eval_dataset must implement __len__")

#         eval_dataloader = self.get_eval_dataloader(eval_dataset)
#         start_time = time.time()

#         output = self.prediction_loop(
#             eval_dataloader,
#             description="Evaluation",
#             # No point gathering the predictions if there are no metrics, otherwise we defer to
#             # self.args.prediction_loss_only
#             prediction_loss_only=True if self.compute_metrics is None else None,
#             ignore_keys=ignore_keys,
#             metric_key_prefix=metric_key_prefix,
#         )

#         n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
#         output.metrics.update(speed_metrics(metric_key_prefix, start_time, n_samples))
#         self.log(output.metrics)

#         if hasattr(self, "weakly_supervise") and self.weakly_supervise and "weakly_supervise" in output.metrics:
#             weakly_dataset = deepcopy(eval_dataset)
#             weakly_dataset["score"] = torch.tensor(metrics )

#             pass

#         if self.args.tpu_metrics_debug or self.args.debug:
#             # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
#             xm.master_print(met.metrics_report())

#         self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

#         self._memory_tracker.stop_and_update_metrics(output.metrics)

#         return output.metrics


class NotebookCallback(NotebookProgressCallback):
    """
    A bare [`TrainerCallback`] that just prints the logs.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Only for when there is no evaluation
        if args.evaluation_strategy == IntervalStrategy.NO and "loss" in logs:
            values = {"Training Loss": logs["loss"]}
            # First column is necessarily Step sine we're not in epoch eval strategy
            values["Step"] = state.global_step
            valid_values = {v for k,v in values.items() if isinstance(v, (str,int,float))}
            self.training_tracker.write_line(valid_values)


    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.training_tracker is not None:
            values = {"Training Loss": "No log", "Validation Loss": "No log"}
            for log in reversed(state.log_history):
                if "loss" in log:
                    values["Training Loss"] = log["loss"]
                    break

            if self.first_column == "Epoch":
                values["Epoch"] = int(state.epoch)
            else:
                values["Step"] = state.global_step
            metric_key_prefix = "eval"
            for k in metrics:
                if k.endswith("_loss"):
                    metric_key_prefix = re.sub(r"\_loss$", "", k)

            # _ = metrics.pop("total_flos", None)
            # _ = metrics.pop("total_flos", None)
            # _ = metrics.pop("epoch", None)
            # _ = metrics.pop(f"{metric_key_prefix}_runtime", None)
            # _ = metrics.pop(f"{metric_key_prefix}_samples_per_second", None)
            # _ = metrics.pop(f"{metric_key_prefix}_steps_per_second", None)
            # _ = metrics.pop(f"{metric_key_prefix}_jit_compilation_time", None)

            for k, v in metrics.items():
                if k == f"{metric_key_prefix}_loss":
                    values["Validation Loss"] = v
                else:
                    splits = k.split("_")
                    name = " ".join([part.capitalize() for part in splits[1:]])
                    values[name] = v
            
            valid_values = {k:v for k,v in values.items() if isinstance(v, (str,int,float))}
            self.training_tracker.write_line(valid_values)
            self.training_tracker.remove_child()
            self.prediction_bar = None
            # Evaluation takes a long time so we should force the next update.
            self._force_next_update = True
# class PrinterCallback():

