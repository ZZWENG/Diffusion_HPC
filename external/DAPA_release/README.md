# Domain Adaptive 3D Pose Augmentation for In-the-wild Human Mesh Recovery (3DV 2022)
[Project Page](https://zzweng.github.io/DAPA_release/)  |  [Paper](https://arxiv.org/abs/2206.10457)  |  [Poster](https://zzweng.github.io/DAPA_release/219_poster.pdf)

Domain Adaptive 3D Pose Augmentation (DAPA) is a data augmentation method that enhances the HMR model's generalization ability in **in-the-wild** scenarios. DAPA combines the strength of methods based on synthetic datasets by getting direct supervision from the synthesized meshes, and domain adaptation methods by using **only 2D keypoints** from the target dataset.

## Examples on challenging sports poses in the wild.
<p float="center">
  <img src="./assets/Picture1.png" width="20%" />
  <img src="./assets/smart_120_rotation.gif" width="20%" />
  <img src="./assets/Picture2.png" width="20%" />
  <img src="./assets/smart_93_rotation.gif" width="20%" />
</p>

# Installation instructions
Tested on Ubuntu 16.
```
conda create -n dapa python==3.6.9
conda activate dapa
pip install --r requirements.txt
git clone https://github.com/nghorbani/human_body_prior.git
cd human_body_prior; python setup.py install; cd ../
```
1. Fetch the dependency data using the [script](https://github.com/nkolot/SPIN#fetch-data) from SPIN.
2. Put the `smpl_uv.obj` [file](https://drive.google.com/drive/folders/1eLWSAN7GUH7wyJvOkgYeNxnhbjXoYQYt?usp=sharing) and `smpl` [model files](https://smpl.is.tue.mpg.de/), and VPoser prior [checkpoint](https://smpl-x.is.tue.mpg.de/) in `data` folder. Double check that the paths `SMPL_MODEL_DIR`, `VPOSER_PATH` and `UV_MESH_FILE` are set correctly in config.py
3. Install the external dependency for the texture model using this [script](https://github.com/akanazawa/cmr/blob/master/external/install_external.sh)

# Training/Evaluation
## AGORA experiments
### Data Preparation
Download the train/valid/test images and train/valid ground truths from [AGORA website](https://agora.is.tue.mpg.de/index.html). Run OpenPose on the valid/test images. The final data folder has the following structure
```
- data/agora
    - train_images_3840x2160
    - Cam  # ground truth for the train split
    - validation_images_3840x2160
        - validation  # images
        - keypoints  # OpenPose json files
    - validation_SMPL  # ground truth for the val split
    - test_images_3840x2160
        - test   # images
        - keypoints  # OpenPose json files
```
Run the data preprocessing script
```
python -m datasets.agora.preprocess_agora_train
python -m datasets.agora.preprocess_validset_from_openpose valid
python -m datasets.agora.preprocess_validset_from_openpose test
```
These will generate `.npz` files in the path specified by `config.DATASET_NPZ_PATH`

### Run training code
We provide the pretrained model checkpoint as part of the supplementary for ease of reproducing the finetuning results. The pretrained checkpoint and the finetuned checkpoints are can be accessed via [this](https://drive.google.com/drive/folders/1eLWSAN7GUH7wyJvOkgYeNxnhbjXoYQYt?usp=sharing) anonymous link.
```bash
checkpoint=2021_10_23-02_02_06.pt

# our finetuning command
python train.py --name ours \
	--checkpoint ${checkpoint} \
	--resume \
	--checkpoint_steps 500 \
	--log_dir logs \
	--agora \
	--test_steps 1200 \
	--rot_factor 0 \
	--ignore_3d \
	--add_background \
	--use_texture \
	--g_input_noise_scale 0.5 \
	--g_input_noise_type mul \
	--vposer
	
# baseline (SPIN-ft-AGORA-2D) finetuning command
python train.py --name spin \
	--checkpoint ${checkpoint} \
	--resume \
	--checkpoint_steps 500 \
	--log_dir logs \
	--agora \
	--test_steps 1200 \
	--rot_factor 0 \
	--ignore_3d \
	--adapt_baseline \
	--run_smplify
```

The finetuned checkpoint performance on the validation set is

| Models         | filename     | MPJPE | NMJE | MVE | NMVE | F1 | Precision | Recall
|----------------|:-----------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| SPIN-ft-AGORA-2D | agora_ft_spin_2d.pt | 166 | 218.4 | 165.1 | 217.2 | 0.76 | 0.9 | 0.65
| DAPA (Ours)   | agora_dapa.pt | **159.4** | **209.7** | **158.6** | **208.7** | 0.76 | 0.9 | 0.65


### Run evaluation code
1. First, clone and build [agora_evaluation](https://github.com/pixelite1201/agora_evaluation) to get the agora_evaluation cli.
2. Prepare prediction files. 
```bash
name=spin_ft
basePath=path_to_store_pred_files
mkdir ${basePath}/${name}
mkdir ${basePath}/${name}/valid_predictions
python prepare_agora_prediction.py --out_folder ${basePath}/${name}/valid_predictions --checkpoint spin_agora_ft.pt --split validation

```
3. Run evaluation on valid set
```bash
debugPath=${basePath}/${name}/debug
resultPath=${basePath}/${name}/results
pred_path=${basePath}/${name}/valid_predictions

imgFolder=data/validation_images_3840x2160/validation
gtFolder=data/validation_SMPL/SMPL/
utilsPath=agora_evaluation/utils
smplPath=data/body_models/smpl

evaluate_agora --pred_path $pred_path --result_savePath $resultPath --imgFolder $imgFolder --loadPrecomputed $gtFolder --modeltype SMPL --indices_path $utilsPath --kid_template_path $utilsPath/smpl_kid_template.npy  --modelFolder $smplPath --baseline demo_model --debug --debug_path $debugPath
```
4. Submit test predictions to [test server](https://agora-evaluation.is.tuebingen.mpg.de/)
```bash
mkdir ${basePath}/${name}/predictions
python prepare_agora_prediction.py --out_folder ${basePath}/${name}/predictions --checkpoint spin_agora_ft.pt --split test
zip preds.zip ${basePath}/${name}/predictions/*
```


## Run on an arbitraty video
1. Download a youtube video and extract frames
```
pip install youtube-dl
youtube-dl PSBOjqCtpEU

mv PSBOjqCtpEU.mkv ./demo

python demo/preprocess.py --path ./demo --fps 5

```
2. Run OpenPose
```
singularity exec --pwd openpose --nv --bind ./demo/PSBOjqCtpEU:/mnt /oak/stanford/groups/syyeung/containers/openpose.sif ./build/examples/openpose/openpose.bin --image_dir /mnt/images/ --face --hand --display 0 --render_pose 1 --write_images /mnt/keypoints --write_json /mnt/keypoints
```
Now demo folder will have the following structure
```
- demo
	- images
	- keypoints
```

3. Run preprocessing script to store the keypoints and bounding boxes in a npz file.
```
python demo/preprocess_from_keypoints_video.py --input_path ./demo --out_path DATASET_NPZ_PATH
```
DATASET_NPZ_PATH is specified in config.py.

4. Add the new dataset to `DATASET_FILES` and `DATASET_FOLDERS` in `config.py`, and run the training script.
```
./scripts/finetune_gym.sh
```

## Run on SEEDLingS
Coming.


# Acknowledgement
The implementation took reference from [SPIN](https://github.com/nkolot/SPIN), [CMR](https://github.com/akanazawa/cmr). We thank the authors for their generosity to release code.

# Citation
If you find our work useful, please consider citing:

```BibTeX
@inproceedings{weng2022domain,
  title={Domain Adaptive 3D Pose Augmentation for In-the-wild Human Mesh Recovery},
  author={Weng, Zhenzhen and Wang, Kuan-Chieh and Kanazawa, Angjoo and Yeung, Serena},
  booktitle={International Conference on 3D Vision},
  year={2022}
}
```
