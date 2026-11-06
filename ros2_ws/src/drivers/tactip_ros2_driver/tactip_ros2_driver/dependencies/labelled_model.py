import time

import numpy as np
import torch
from torch.autograd import Variable

from .image_transforms import process_image


class LabelledModel:
    def __init__(self,
                 model,
                 image_processing_params,
                 label_encoder,
                 device='cpu'
                 ):
        self.model = model
        self.image_processing_params = image_processing_params
        self.label_encoder = label_encoder
        self.label_names = label_encoder.label_names
        self.target_label_names = label_encoder.target_label_names
        self.device = device
        # (prep, forward, decode) seconds of the last predict() call, for timing logs
        self.last_timing = (0.0, 0.0, 0.0)

    def predict(self, tactile_image):
        t0 = time.perf_counter()
        processed_image = process_image(
            tactile_image,
            gray=False,
            **self.image_processing_params
        )

        # channel first for pytorch; add batch dim
        processed_image = np.rollaxis(processed_image, 2, 0)
        processed_image = processed_image[np.newaxis, ...]

        # perform inference with the trained model
        model_input = Variable(torch.from_numpy(processed_image)).float().to(self.device)
        t1 = time.perf_counter()
        outputs = self.model(model_input)
        t2 = time.perf_counter()

        # decode the prediction
        predictions_dict = self.label_encoder.decode_label(outputs)

        # pack into array and report
        #print("\nPredictions: ", end="")
        predictions_arr = np.zeros(len(self.label_names))
        for label_name in self.target_label_names:
            predicted_val = predictions_dict[label_name].detach().cpu().numpy().squeeze()
            predictions_arr[self.label_names.index(label_name)] = predicted_val
            with np.printoptions(precision=2, suppress=True):
                #print(label_name, predicted_val, end=" ")
                pass

        self.last_timing = (t1 - t0, t2 - t1, time.perf_counter() - t2)
        return predictions_arr


class RKNNLabelledModel:
    """Drop-in for LabelledModel that runs a quantized .rknn model on the RK3588 NPU.

    The .rknn is converted from the same weights with tools/npu_bench (export_onnx.py +
    convert_rknn.py). The /255 normalization is folded into the .rknn, so the model takes the
    resized uint8 image directly as NHWC (1, H, W, 1). Output decoding is the same LabelEncoder
    as the PyTorch path.

    Only NPU core 0 is used: on the current Orange Pi kernel (rknpu driver 0.9.3) cores 1 and 2
    hang. Several instances (one per sensor) share core 0 fine as long as their inferences do
    not overlap, which the driver's single inference worker guarantees.
    """

    def __init__(self, rknn_path, image_processing_params, label_encoder):
        from rknnlite.api import RKNNLite  # only needed (and only installed) on the board

        if image_processing_params.get('stdiz'):
            raise ValueError("RKNN backend does not support per-image standardisation (stdiz)")
        # Keep everything except the /255: the image stays uint8 for the NPU.
        self.image_processing_params = dict(image_processing_params, normlz=False)
        self.label_encoder = label_encoder
        self.label_names = label_encoder.label_names
        self.target_label_names = label_encoder.target_label_names
        self.last_timing = (0.0, 0.0, 0.0)

        self.rknn = RKNNLite(verbose=False)
        if self.rknn.load_rknn(rknn_path) != 0:
            raise RuntimeError(f"load_rknn failed: {rknn_path}")
        if self.rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0) != 0:
            raise RuntimeError("RKNN init_runtime failed (NPU driver or librknnrt missing?)")

    def predict(self, tactile_image):
        t0 = time.perf_counter()
        img = process_image(tactile_image, gray=False, **self.image_processing_params)
        if img.ndim == 2:
            img = img[..., np.newaxis]
        model_input = np.ascontiguousarray(img[np.newaxis], dtype=np.uint8)  # NHWC
        t1 = time.perf_counter()
        out = self.rknn.inference(inputs=[model_input], data_format='nhwc')[0]
        t2 = time.perf_counter()

        outputs = torch.from_numpy(np.asarray(out, dtype=np.float32).reshape(1, -1))
        predictions_dict = self.label_encoder.decode_label(outputs)
        predictions_arr = np.zeros(len(self.label_names))
        for label_name in self.target_label_names:
            predictions_arr[self.label_names.index(label_name)] = \
                predictions_dict[label_name].numpy().squeeze()

        self.last_timing = (t1 - t0, t2 - t1, time.perf_counter() - t2)
        return predictions_arr

    def release(self):
        self.rknn.release()


if __name__ == '__main__':
    pass
