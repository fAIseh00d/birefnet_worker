from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from PIL import Image
from PIL.Image import Image as PILImage


class BiRefNetSessionONNX:
    """
    This class represents a BiRefNet-General-Lite session.
    """

    def __init__(self, model_path: str, sess_opts: ort.SessionOptions):
        """Initialize an instance of the BiRefNetSessionONNX class."""
        self.model_path = model_path

        # Check providers directly without triggering slow device discovery
        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.inner_session = ort.InferenceSession(
            model_path,
            sess_options=sess_opts,
            providers=providers,
        )

    def normalize(
        self,
        img: PILImage,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        size: Tuple[int, int],
        *args,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        im = img.convert("RGB").resize(size, Image.Resampling.LANCZOS)

        im_ary = np.array(im)
        im_ary = im_ary / max(np.max(im_ary), 1e-6)

        tmpImg = np.zeros((im_ary.shape[0], im_ary.shape[1], 3))
        tmpImg[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
        tmpImg[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
        tmpImg[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]

        tmpImg = tmpImg.transpose((2, 0, 1))

        return {
            self.inner_session.get_inputs()[0]
            .name: np.expand_dims(tmpImg, 0)
            .astype(np.float32)
        }

    def sigmoid(self, mat):
        return 1 / (1 + np.exp(-mat))

    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        """
        Predicts the output masks for the input image using the inner session.

        Parameters:
            img (PILImage): The input image.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[PILImage]: The list of output masks.
        """
        ort_outs = self.inner_session.run(
            None,
            self.normalize(
                img, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), (1024, 1024)
            ),
        )

        pred = self.sigmoid(ort_outs[0][:, 0, :, :])

        ma = np.max(pred)
        mi = np.min(pred)

        pred = (pred - mi) / (ma - mi)
        pred = np.squeeze(pred)

        # Clip near-white to white and near-black to black for crisper masks
        pred = np.where(pred >= 0.98, 1.0, pred)
        pred = np.where(pred <= 0.02, 0.0, pred)

        mask = Image.fromarray((pred * 255).astype("uint8"), mode="L")
        mask = mask.resize(img.size, Image.Resampling.LANCZOS)

        return [mask]
