import time
import torch
import logging
import datetime
from collections import OrderedDict
from contextlib import contextmanager
from detectron2.utils.comm import is_main_process
from .calibration_layer import PrototypicalCalibrationBlock
from .novel_methods import build_novel_method_pcb


class DatasetEvaluator:
    """
    Base class for a dataset evaluator.

    The function :func:`inference_on_dataset` runs the model over
    all samples in the dataset, and have a DatasetEvaluator to process the inputs/outputs.

    This class will accumulate information of the inputs/outputs (by :meth:`process`),
    and produce evaluation results in the end (by :meth:`evaluate`).
    """

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        pass

    def process(self, input, output):
        """
        Process an input/output pair.

        Args:
            input: the input that's used to call the model.
            output: the return value of `model(output)`
        """
        pass

    def evaluate(self):
        """
        Evaluate/summarize the performance, after processing all input/output pairs.

        Returns:
            dict:
                A new evaluator class can return a dict of arbitrary format
                as long as the user can process the results.
                In our train_net.py, we expect the following format:

                * key: the name of the task (e.g., bbox)
                * value: a dict of {metric name: score}, e.g.: {"AP50": 80}
        """
        pass


class DatasetEvaluators(DatasetEvaluator):
    def __init__(self, evaluators):
        assert len(evaluators)
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, input, output):
        for evaluator in self._evaluators:
            evaluator.process(input, output)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process():
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results


def inference_on_dataset(model, data_loader, evaluator, cfg=None):

    num_devices = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    logger = logging.getLogger(__name__)

    pcb = None
    if cfg.TEST.PCB_ENABLE:
        logger.info("Start initializing PCB module, please wait a seconds...")
        pcb = PrototypicalCalibrationBlock(cfg)

        # Wrap PCB with novel method if enabled
        if cfg.NOVEL_METHODS.ENABLE and cfg.NOVEL_METHODS.METHOD:
            method_name = cfg.NOVEL_METHODS.METHOD
            logger.info(f"Applying novel method: {method_name}")
            pcb = build_novel_method_pcb(pcb, cfg, method_name)

    # UPR-TTA two-pass: uncertainty-guided pseudo-label collection (replaces standard transductive)
    if pcb is not None and hasattr(pcb, 'run_pass1') and cfg.NOVEL_METHODS.ENABLE:
        method_name = cfg.NOVEL_METHODS.METHOD.lower()
        if method_name in ("upr_tta", "upr", "uncertainty_refinement"):
            with inference_context(model):
                pcb.run_pass1(model, data_loader)

    # Two-pass transductive: collect pseudo-labels in pass 1, rebuild, then evaluate in pass 2.
    elif pcb is not None and cfg.TEST.PCB_TRANSDUCTIVE and not cfg.TEST.PCB_TRANS_ONLINE:
        logger.info(
            "Transductive inference pass 1: collecting pseudo-labels over %d images...",
            len(data_loader),
        )
        pseudo_dict = {}
        with inference_context(model), torch.no_grad():
            for inputs in data_loader:
                cur_idx = sum(len(v.get("features", [])) for v in pseudo_dict.values())
                outputs = model(inputs)
                if cfg.TEST.PCB_TRANS_PSEUDO_CALIBRATED:
                    outputs = pcb.execute_calibration(inputs, outputs, allow_reassign=False)
                pcb.collect_pseudo(inputs, outputs, pseudo_dict)
                new_idx = sum(len(v.get("features", [])) for v in pseudo_dict.values())
                if new_idx > 0 and (new_idx // max(cfg.TEST.PCB_TRANS_MAX_PER_CLASS, 1)) != (cur_idx // max(cfg.TEST.PCB_TRANS_MAX_PER_CLASS, 1)):
                    logger.info(
                        "Transductive pass 1 progress: pseudo_classes=%d pseudo_samples=%d",
                        len(pseudo_dict),
                        new_idx,
                    )
            logger.info(
                "Transductive pass 1 complete: pseudo_classes=%d pseudo_samples=%d",
                len(pseudo_dict),
                sum(len(v.get('features', [])) for v in pseudo_dict.values()),
            )
        pcb.rebuild_with_pseudo(pseudo_dict)
        logger.info("Transductive inference pass 2: final evaluation...")

    # Online transductive: accumulate pseudo across images, rebuild after each.
    online_pseudo = {} if (pcb is not None and cfg.TEST.PCB_TRANSDUCTIVE and cfg.TEST.PCB_TRANS_ONLINE) else None

    logger.info("Start inference on {} images".format(len(data_loader)))
    total = len(data_loader)  # inference data loader must have a fixed length
    evaluator.reset()

    logging_interval = 50
    num_warmup = min(5, logging_interval - 1, total - 1)
    start_time = time.time()
    total_compute_time = 0
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.time()
                total_compute_time = 0

            start_compute_time = time.time()
            outputs = model(inputs)
            if cfg.TEST.PCB_ENABLE:
                outputs = pcb.execute_calibration(inputs, outputs)
                if online_pseudo is not None:
                    pcb.collect_pseudo(inputs, outputs, online_pseudo)
                    pcb.rebuild_with_pseudo(online_pseudo)
            # CPU-only runs should skip CUDA synchronization.
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.time() - start_compute_time
            evaluator.process(inputs, outputs)

            if (idx + 1) % logging_interval == 0:
                duration = time.time() - start_time
                seconds_per_img = duration / (idx + 1 - num_warmup)
                eta = datetime.timedelta(
                    seconds=int(seconds_per_img * (total - num_warmup) - duration)
                )
                logger.info(
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    )
                )

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = int(time.time() - start_time)
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)
