import json
import random

import imageio
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

imageio.plugins.freeimage.download()


def _foreground_mask(mask, height, width):
    """Return a float mask with shape (1, H, W) in [0, 1]."""
    if mask is None:
        out = np.ones((height, width), dtype=np.float32)
    else:
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(np.float32)
        if mask.max() > 1.0:
            mask = mask / 255.0
        if mask.shape[0] != height or mask.shape[1] != width:
            mask_image = Image.fromarray((np.clip(mask, 0.0, 1.0) * 255.0).astype(np.uint8))
            mask = np.array(mask_image.resize((width, height), Image.NEAREST)).astype(np.float32) / 255.0
        out = mask
    return out[None, ...].astype(np.float32)


class ObjectRelightingDataset(Dataset):
    def __init__(
            self,
            data_jsonl=None,
            pretrained_model=None,
            empty_prompt_ratio=0.0,
            eval_mode=False,
            direct_data=None,
    ):
        self.data = []
        if pretrained_model is None:
            raise ValueError("pretrained_model is required.")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model,
            subfolder="tokenizer",
            use_fast=False,
        )
        self.empty_prompt_ratio = empty_prompt_ratio
        self.eval_mode = eval_mode
        if direct_data is not None:
            self.data = direct_data
        elif data_jsonl is not None:
            with open(data_jsonl, 'rt') as f:
                for line in tqdm(f, desc='Loading data'):
                    self.data.append(json.loads(line))
        else:
            raise ValueError("No data source specified.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            parts = item['target'].split('/')
            model_id = '_'.join([parts[6], parts[7], parts[8]])

            target_filename = item['target']
            source_filename = item['fg']
            light_filename = item['bg']
            mask_filename = item['mask']

            mask = imageio.v3.imread(mask_filename) / 255.
            non_mask = 1 - mask
            source = imageio.v3.imread(source_filename)[..., :3] / 255.
            light = imageio.v3.imread(light_filename)[..., :3] / 255.

            source = source[:, :, :3] * mask[:, :, np.newaxis]
            light = light[:, :, :3] * non_mask[:, :, np.newaxis]

            target = imageio.v3.imread(target_filename)[..., :3] / 127.5 - 1.

            source = source * 2.0 - 1.0
            light = light * 2.0 - 1.0

            conds = np.concatenate([source, light], axis=2).astype(np.float32)
            target = target.astype(np.float32)
            mask_out = _foreground_mask(mask, target.shape[0], target.shape[1])

            prompt = item['lighting_description']
            if random.random() < self.empty_prompt_ratio:
                prompt = ""
            inputs = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
        except Exception as e:
            if self.eval_mode:
                raise e
            print(idx, repr(e), e)
            return self.__getitem__(np.random.randint(0, len(self.data)))

        return dict(
            pixel_values=target.transpose(2, 0, 1),
            input_ids=inputs.input_ids[0],
            conditioning_pixel_values=conds.transpose(2, 0, 1),
            mask=mask_out,
            text=prompt,
            target_file=target_filename,
            ref_file=source_filename,
            model_id=model_id,
        )
