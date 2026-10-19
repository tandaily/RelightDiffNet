# [IEEE BigComp 2026] RelightDiffNet: A Dual-Branch Diffusion Model for Background-Conditioned Foreground Relighting

RelightDiffNet performs background-conditioned foreground relighting. Given a foreground, its mask, and a target background, it adjusts the foreground lighting to match the background and composites the two.

## Method

The model is a dual-branch diffusion network.

- The generation branch is the pretrained denoising UNet. Its input is the noisy target latent z_t, the text condition, and the timestep. Its weights stay frozen during training.
- The relighting branch is initialized as a copy of the generation branch and is trainable. Its input is the 12-channel concatenation of z_t, the foreground latent, and the background latent. The extra 8 input channels are zero-initialized.
- Features from every layer of the relighting branch pass through a 1\times1 zero convolution and are added to the matching down, mid, and up features of the generation branch. At initialization the injection is zero, so the output matches the pretrained diffusion model.



## Dataset format

Training and validation sets are JSONL files, one JSON object per line. 


| Field                  | Meaning                                                |
| ---------------------- | ------------------------------------------------------ |
| `fg`                   | Path to the foreground image                           |
| `bg`                   | Path to the target lighting background                 |
| `target`               | Path to the relit harmonious image                     |
| `mask`                 | Path to the foreground mask; white (255) is foreground |
| `lighting_description` | Lighting description used as the text condition        |


Example:

```json
{"fg": "/path/to/fg.png", "bg": "/path/to/bg.png", "target": "/path/to/target.png", "mask": "/path/to/mask.png", "lighting_description": "warm sunlight from the left"}
```



## Training

Edit the paths in `train/train.sh`, then run the following from the `RelightDiffNet` directory:

Paths to replace in `train.sh`:

- `MODEL_ID`: Stable Diffusion v1.5 directory
- `OUTPUT_DIR`: checkpoint directory
- `RMBG_MODEL_PATH`: RMBG-2.0 directory, used to segment real portraits during validation
- `--object_dataset_name`: training JSONL
- `--dataset_name`, `--val_syn_portrait_image_url`, `--val_object_image_url`, `--val_real_portrait_image_url`: validation JSONL files

`--object_dataset_name` is the training set. The other JSONL files are read only during validation. An empty `--validation_prompt` still runs validation. Omitting that argument skips validation.
