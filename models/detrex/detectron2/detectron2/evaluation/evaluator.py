# Copyright (c) Facebook, Inc. and its affiliates.
import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn

import numpy as np
import cv2
import os
from skimage.metrics import structural_similarity as ssim

from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds


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

    def process(self, inputs, outputs):
        """
        Process the pair of inputs and outputs.
        If they contain batches, the pairs can be consumed one-by-one using `zip`:

        .. code-block:: python

            for input_, output in zip(inputs, outputs):
                # do evaluation on single input/output pair
                ...

        Args:
            inputs (list): the inputs that's used to call the model.
            outputs (list): the return value of `model(inputs)`
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
    """
    Wrapper class to combine multiple :class:`DatasetEvaluator` instances.

    This class dispatches every evaluation call to
    all of its :class:`DatasetEvaluator`.
    """

    def __init__(self, evaluators):
        """
        Args:
            evaluators (list): the evaluators to combine.
        """
        super().__init__()
        self._evaluators = evaluators

    def reset(self):
        for evaluator in self._evaluators:
            evaluator.reset()

    def process(self, inputs, outputs):
        for evaluator in self._evaluators:
            evaluator.process(inputs, outputs)

    def evaluate(self):
        results = OrderedDict()
        for evaluator in self._evaluators:
            result = evaluator.evaluate()
            if is_main_process() and result is not None:
                for k, v in result.items():
                    assert (
                        k not in results
                    ), "Different evaluators produce results with the same key {}".format(k)
                    results[k] = v
        return results

def calc_ssim(im1, im2):
    im1 = im1.detach().cpu().numpy()
    im2 = im2.detach().cpu().numpy()
    the_ssim = ssim(im1, im2, data_range=(np.max(im1) - np.min(im1)), channel_axis=0)
    return the_ssim

def max_diff(im1, im2):
    im1 = im1.detach().cpu()
    im2 = im2.detach().cpu()
    #if torch.max(im1) - torch.min(im1) != 0:
    #    im1 = (im1 - torch.max(im1)) / (torch.max(im1) - torch.min(im1))
    #if torch.max(im1) - torch.min(im1) != 0:
    #    im2 = (im2 - torch.max(im2)) / (torch.max(im2) - torch.min(im2))
    diff = torch.max(torch.abs(im1 - im2))
    return diff

def l2(im1, im2):
    im1 = im1.detach().cpu()
    im2 = im2.detach().cpu()
    #if torch.max(im1) - torch.min(im1) != 0:
    #    im1 = (im1 - torch.max(im1)) / (torch.max(im1) - torch.min(im1))
    #if torch.max(im1) - torch.min(im1) != 0:
    #    im2 = (im2 - torch.max(im2)) / (torch.max(im2) - torch.min(im2))
    l2_norm = torch.linalg.vector_norm(im1 - im2, 2) / (10e-3 * im1.shape[0] * im1.shape[1] * im1.shape[2])
    return l2_norm

def l0(im1, im2):
    im1 = im1.detach().cpu()
    im2 = im2.detach().cpu()
    l0_norm = (torch.count_nonzero(im1 - im2)) / (float(im1.shape[0]) * im1.shape[1] * im1.shape[2])
    return l0_norm

def mean_pert(im1, im2):
    im1 = im1.detach().cpu()
    im2 = im2.detach().cpu()
    mean_pert = torch.mean(torch.abs(im1 - im2))
    return mean_pert

def median_pert(im1, im2):
    im1 = im1.detach().cpu()
    im2 = im2.detach().cpu()
    median_pert = torch.median(torch.abs(im1 - im2))
    return median_pert

# Modified for attacking
def inference_on_dataset(
    model,
    data_loader,
    evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
    callbacks=None,
    attack=None,
    attack_mode=None,
    save_attack=None,
    save_dir=None,
    load_attack=None,
    load_dir=None,
    sample=1.0
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.
        callbacks (dict of callables): a dictionary of callback functions which can be
            called at each stage of inference.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    # Count the parameters in the model
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    diff = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        dict.get(callbacks or {}, "on_start", lambda: None)()
        print("Doing attack:", attack)
        
        # Initialize logging
        total_l2 = 0.0
        total_l0 = 0.0
        total_mean_pert = 0.0
        total_median_pert = 0.0
        total_ssim = 0.0
        total_time = 0.0
        i = 0.0
        for idx, inputs in enumerate(data_loader):
            
            # COCO downsampling to make attacking large models possible on limited compute
            if torch.rand(1)[0] >= sample:
                continue
                
            i += 1.0
            
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            dict.get(callbacks or {}, "before_inference", lambda: None)()
           
            if float(load_attack) == 1.0:
                name = os.path.basename(inputs[0].get("file_name"))
                path = load_dir + name[:-4] + ".npz"
                pert = torch.tensor(np.resize(np.load(path)["arr_0"], inputs[0]["image"].shape))
                inputs[0]["image"] = inputs[0]["image"].float() 
                inputs[0]["image"] += pert

            # Attack the outputs
            assert torch.isfinite(inputs[0]["image"]).all(), "input image contains infinite or NaN!"
            if attack != None:
                orig_image = inputs[0]["image"].detach().clone()
                start = time.perf_counter()
                output_image = attack(model, x_query = inputs[0]["image"], mode=attack_mode, meta=inputs[0])
                end = time.perf_counter()
                total_time += (end - start)
                output_image = output_image.detach().float()
                
                # If we want to save the images for use later
                if float(save_attack) == 1.0:
                    name = os.path.basename(inputs[0].get("file_name"))
                    arr = output_image.clone().cpu().numpy() - orig_image.clone().cpu().numpy()
                    arr = arr.astype(np.float16)
                    path = save_dir + name[:-4] + ".npz"
                    np.savez_compressed(path, arr)
                
                # Calculate the metrics of interest  
                out_copy = output_image.detach().clone()
                orig_image = (orig_image - torch.min(orig_image)) / (torch.max(orig_image) - torch.min(orig_image))
                out_copy = (out_copy - torch.min(out_copy)) / (torch.max(out_copy) - torch.min(out_copy))
                total_ssim += calc_ssim(orig_image, out_copy)
                total_l2 += l2(orig_image, out_copy)
                total_l0 += l0(orig_image, out_copy)
                total_mean_pert += mean_pert(orig_image, out_copy)
                total_median_pert += median_pert(orig_image, out_copy)
                this_diff = max_diff(orig_image, out_copy)
                diff = max(this_diff, diff)
                assert torch.isfinite(output_image).all(), "output_image after metrics contains infinite or NaN!"
                inputs[0]["image"] = output_image
            outputs = model(inputs)
            
            dict.get(callbacks or {}, "after_inference", lambda: None)()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()
        dict.get(callbacks or {}, "on_end", lambda: None)()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
        
    # Print the logging results
    print("\n\n------- Similarity and Timing Metrics -------")
    print("Total Parameters:", total_params)
    print(f"Average Attack Time: %0.4f" % (total_time / i))
    print(f"Average L2 Norm: %0.4f" % (total_l2 / i))
    print(f"Average L0 Norm: %0.4f" % (total_l0 / i))
    print(f"Average SSIM %0.4f" % (total_ssim / i))
    print(f"Mean Difference in Images: %0.4f" % (total_mean_pert / i))
    print(f"Median Difference in Images: %0.4f" % (total_median_pert / i))
    print(f"Max Difference in Images: %0.4f" % diff)
    print("------------------------------------------------\n\n")
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
