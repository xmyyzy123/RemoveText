import os
import cv2
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate

from lama.saicinpainting.evaluation.utils import move_to_device
from lama.saicinpainting.training.data.datasets import make_default_val_dataset
from lama.saicinpainting.training.trainers import load_checkpoint

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

lama_train_config_path = 'lama/big-lama/config.yaml'
lama_predict_config_path = 'lama/configs/prediction/default.yaml'

def lama_inpainting(input_dir, output_dir, window):
    with open(lama_predict_config_path, 'r') as f:
        predict_config = OmegaConf.create(yaml.safe_load(f))
    with open(lama_train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    device = torch.device(predict_config.device)
        
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    out_ext = predict_config.get('out_ext', '.jpg')

    checkpoint_path = os.path.join(predict_config.model.path, 
                                       'models', 
                                       predict_config.model.checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    model.to(device)

    if not input_dir.endswith('/'):
        input_dir += '/'

    dataset = make_default_val_dataset(input_dir, **predict_config.dataset)
    with torch.no_grad():
        i = 0
        num = len(dataset)
        for img_i in tqdm.trange(num):
            mask_fname = dataset.mask_filenames[img_i]
            cur_out_fname = os.path.join(
                output_dir, 
                os.path.splitext(mask_fname[len(input_dir):])[0] + '_inpainted' + out_ext
            )
            os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)

            batch = move_to_device(default_collate([dataset[img_i]]), device)
            batch['mask'] = (batch['mask'] > 0) * 1
            batch = model(batch)
            cur_res = batch[predict_config.out_key][0].permute(1, 2, 0).detach().cpu().numpy()

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            cv2.imwrite(cur_out_fname, cur_res)

            i += 1
            window.write_event_value('-PROGRESS-', 50 + i / num * 50)

    window.write_event_value('-LAMA THREAD-', '')
#lama_inpainting(input_dir='lama/test_img', output_dir='lama/output')