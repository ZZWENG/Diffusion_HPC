# Repository for Diffusion HPC: Generate Synthetic Images with Realistic Humans

## 1. Installation
Tested with Python 3.8 and CUDA 11.3.
```
conda create -n diffusion_hpc python==3.8
conda activate diffusion_hpc
conda install -y pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt

# Optional: re-install pyopengl from source if using OSMesa Headless rendering
# pip uninstall pyopengl; git clone https://github.com/mmatl/pyopengl.git; pip install ./pyopengl
```

## 2. Prepare model data and datasets

### Download model data
Please download VPoser weights [(`V02_05`)](https://smpl-x.is.tue.mpg.de/) and [SMPL](https://smpl.is.tue.mpg.de) model weights from their websites and put under `./data`.

*Note: Only `smpl` and `V02_05` are needed for running demo code. If doing evaluation and downstream experiments, see the step below.*

### Optional: Download data and model checkpoints.
#### 1. Download models from [Google Drive](https://drive.google.com/drive/folders/1xZEPIWC2i1SNjZwVHpKh4rVAWHPbktz4?usp=sharing). It contains
- Finetuned HMR model checkpoints (under `checkpoints`)
- EFT fittings for SMART. (under `eft_smart`)
- Weghts for finetuned Stable Diffusion (used for Table 2). (under `sd_ft_mpii` and `sd_ft_smart`)

#### 2. Download MPII, SMART, and SkiPose.
Download [MPII](http://human-pose.mpi-inf.mpg.de/#download), [SMART](https://github.com/ChenFengYe/SportsCap), and [SkiPose](https://www.epfl.ch/labs/cvlab/data/ski-poseptz-dataset/) datasets and put them under `./data`.

*Note: MPII and SMART are only needed for running the evaluation code (Table 1 and 2). SMART and SkiPose are needed for running HMR finetuning (Table 3 and 4).*

In addition, for pose-conditioned generation on MPII, move EFT fittings for MPII `MPII_ver01.json` and SMART `eft_smart` to their respective folder (see below).

File structure should look like the following
```
- Diffusion_HPC
    - data
        - smpl
        - V02_05
        - SportsCap_Dataset_SMART_v1
            - images
            - annotations
            - eft_smart
        - mpii
            - images
            - annnot
            - MPII_ver01.json
        - ski_3d
            - train
            - test
        - sd_ft_mpii
        - sd_ft_smart
        - checkpoints
```

## 3. DiffusionHPC Demo
This script lets you generate several images for a text prompt.
```
python api.py --text_prompt "a photo of a person doing yoga" --num_images 10 --output_path "./yoga" --use_random_for_d2i
```
Use `--save_mesh` to save the meshes. 


## 4. Experiments
### Generation quality evaluation. (Table 1 and 2)
#### 1. Text-conditioned generations for MPII (or SMART).
```
dataset=mpii   # either mpii or smart
text_eval_path=./eval_data
python -m evaluation.${dataset}_quant_eval --save_folder ${text_eval_path}
```
This could take a long time (6 sec/image), so we recommend parallelizing it over multiple GPUs. For example (with SLURM),
```
for part_id in {0..10}; do 
    srun --gres=gpu:1 python -m evaluation.${dataset}_quant_eval --num_images 1000 --part_id ${part_id}; 
done
```
To use the finetuned Stable Diffusion as backbone, add flag `--fintuned_on ${dataset}`.

#### 2. Pose-condiditioned generations.
```
dataset=mpii   # either mpii or smart
pose_eval_path=./eval_data
python -m evaluation.${dataset}_gen --use_real --use_text --out_path ${pose_eval_path}
```
Turn on flags `--use_real` and/or `--use_text` for using real image and/or text as guidance. Turn on `--finetuned_on ${dataset}` for using finetuned Stable Diffusion as backbone. Use `--save_mpii` to save the real images (for evaluation) at the same time. (Real images will be populated at `${pose_eval_path}/${dataset}`)

#### 3. Compute metrics.

Example command for computing FID/KID:
```
gen_name=ours  # could be ours_ft, ours_notext, etc. depending on the flags in the previous step.
python evaluation/compute_metrics.py ${pose_eval_path}/${dataset} ${pose_eval_path}/quant_pose_eval_${dataset}/${gen_name}
```

Example command for computing H-FID/H-KID:
```
python evaluation/produce_masks.py ${pose_eval_path}/${dataset}
python evaluation/produce_masks.py ${pose_eval_path}/quant_pose_eval_${dataset}/${gen_name}
python evaluation/compute_metrics.py ${pose_eval_path}/${dataset}_masked ${pose_eval_path}/quant_pose_eval_${dataset}/${gen_name}_masked
```

## Finetune HMR on the generated dataset

### Generate synthetic images using real images as guidance.
To generate data for SMART, 
```
ACTION=polevault
python generate_dataset.py --action "an athlete doing gymnastics on a balance beam" --use_real_guide --use_random_latents_for_human --out_path $GEN_PATH --apply_pose_aug --add_blur --real_guidance_action ${ACTION}
```
To generate data for SkiPose,
```
ACTION=ski
GEN_PATH=./gen_data/ski_500_rg

python generate_dataset.py --action "skiing" --use_real_guide --use_random_latents_for_human --out_path $GEN_PATH --apply_pose_aug --add_blur --real_guidance_action ${ACTION} --rg_num_samples 500
```

### Convert generated data to the format consumed by SPIN/DAPA.
```
python downstream_utils/parse_data_from_generated.py --generation_path $GEN_PATH --action $ACTION
python downstream_utils/process_real/parse_from_real.py --action $ACTION
```
*Note: For SkiPose experiments, use `--action ski500` if the synthetic dataset was generated with `--rg_num_samples 500` in previous step.*

The first command above will dump a file named `$ACTION_syn_train.npz`.
The second command will dump two files named `$ACTION_real_train.npz`, and `$ACTION_real_test.npz`
We will use `$ACTION_syn_train.npz` and `$ACTION_real_train.npz` during finetuning, and evaluate on `$ACTION_real_test.npz`.

## Downstream finetuning (Table 3 and 4).
Note, we use `wandb` for logging, so please type `wandb login` in command line to log into your account.

### Finetuning HMR with the generated images.
First, set up DAPA's environment following their instructions. Then do to the DAPA repo.
```
cd external/DAPA_release
conda activate dapa
```
See `./external/DAPA_release/scripts/demo.sh` for example commands for launching SPIN/DAPA/DiffusionHPC finetuning comparisons.

### Evaluate finetuned HMR models.
Metrics should be printed to `wandb` already. Alternatively, use `eval_diffusion.py` to evaluate a given checkpoint.

For Table 3 experiments, use `--eval_type 2d` to compute 2D metric. E.g.,
```
action=polevault   # one of polevault | vault | diving | unevenbars | balancebeam | highjump
python eval_diffusion.py --dataset ${action}_real --checkpoint ../../data/checkpoints/${action}.pt --eval_type 2d
```
For reference, the following metrics should be printed for the provided checkpoints:
- polevault: `PCK: 77.70425618737436`
- vault: `PCK: 66.91005496561053`
- diving: `PCK: 79.27329921107761`
- unevenbars: `PCK: 44.143125469194544`
- balancebeam: `PCK: 85.0919520428191`
- highjump: `PCK: 78.13735096862305`

For Table 4 experiments, use `--eval_type 3d_keypoints` to compute 3D metrics. E.g.,
```
python eval_diffusion.py --dataset ski8481_real --checkpoint ../../data/checkpoints/ski.pt --eval_type 3d_keypoints
```
For reference, this prints
```
MPJPE: 111.67059139302978
Reconstruction Error: 81.76326881529707
```