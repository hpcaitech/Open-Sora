import os
import os.path as osp
from PIL import Image
import numpy as np
import torch

import mmcv
import mmengine
from mmengine import Config
from mmengine.registry import DefaultScope
from mmengine.dataset import Compose, default_collate
from mmocr.registry import MODELS, VISUALIZERS


def visualize(visualizer,
              inputs,
              preds,
              # return_vis: bool = False,
              show: bool = False,
              wait_time: int = 0,
              draw_pred: bool = True,
              pred_score_thr: float = 0.3,
              save_vis: bool = False,
              img_out_dir: str = ''):
    """Visualize predictions.

    Args:
        inputs (List[Union[str, np.ndarray]]): Inputs for the inferencer.
        preds (List[Dict]): Predictions of the model.
        return_vis (bool): Whether to return the visualization result.
            Defaults to False.
        show (bool): Whether to display the image in a popup window.
            Defaults to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        draw_pred (bool): Whether to draw predicted bounding boxes.
            Defaults to True.
        pred_score_thr (float): Minimum score of bboxes to draw.
            Defaults to 0.3.
        save_vis (bool): Whether to save the visualization result. Defaults
            to False.
        img_out_dir (str): Output directory of visualization results.
            If left as empty, no file will be saved. Defaults to ''.

    Returns:
        List[np.ndarray] or None: Returns visualization results only if
        applicable.
    """
    results = []

    for single_input, pred in zip(inputs, preds):
        if isinstance(single_input, str):
            img_bytes = mmengine.fileio.get(single_input)
            img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        elif isinstance(single_input, np.ndarray):
            img = single_input.copy()[:, :, ::-1]  # to RGB
        else:
            raise ValueError('Unsupported input type: '
                             f'{type(single_input)}')
        img_name = osp.splitext(osp.basename(pred.img_path))[0]

        if save_vis and img_out_dir:
            out_file = osp.splitext(img_name)[0]
            out_file = f'{out_file}.jpg'
            out_file = osp.join(img_out_dir, out_file)
        else:
            out_file = None

        visualization = visualizer.add_datasample(
            img_name,
            img,
            pred,
            show=show,
            wait_time=wait_time,
            draw_gt=False,
            draw_pred=draw_pred,
            pred_score_thr=pred_score_thr,
            out_file=out_file,
        )
        results.append(visualization)

    return results


cfg = Config.fromfile('./tools/scoring/ocr/dbnetpp_debug.py')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
DefaultScope.get_instance('ocr', scope_name='mmocr')

model = MODELS.build(cfg.model)
model.init_weights()
model.to(device)
pipeline = Compose(cfg.test_pipeline)
visualizer = VISUALIZERS.build(cfg.visualizer)

inputs = {
    'img_path': './assets/images/ocr/demo_text_ocr.jpg',
    # 'img_path': './assets/images/ocr/demo_text_det.jpg',
}
results = pipeline(inputs)
results['index'] = 0
# imgs = results['inputs'].unsqueeze(0)
# pred = model.predict(imgs, results['data_samples'])
data = default_collate([results])  # list[Dict] to Dict
with torch.no_grad():
    pred = model.test_step(data)
vis_results = visualize(visualizer, [x.img_path for x in data['data_samples']], pred, show=True)
x = 0