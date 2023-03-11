tag=demo
ds_name=polevault

for exp_id in ours spin dapa; do

case "$exp_id" in
ours) 
	checkpoint=data/model_checkpoint.pt
    ds="${ds_name}_real,${ds_name}_syn"
    options="--adapt_baseline --num_epochs 1000"
;;
spin)
	checkpoint=data/model_checkpoint.pt
    ds="${ds_name}_real"
    options="--adapt_baseline --run_smplify --num_epochs 3000"
;;
dapa)
	checkpoint=../../data/2021_10_23-02_02_06.pt
    ds="${ds_name}_real"
    options="--num_epochs 3000 --add_background --openpose_train_weight 0. --gt_train_weight 1. --use_texture --g_input_noise_type mul --vposer"
;;
esac

cmd="train.py --name ${tag}_${exp_id} \
--checkpoint ${checkpoint} \
--resume \
--checkpoint_steps 1000 \
--log_dir ./logs \
--ft_dataset ${ds} \
--wandb_project DAPA \
--test_steps 100 \
--summary_steps 10  \
--rot_factor 30 \
--add_background \
--openpose_train_weight 0. \
--gt_train_weight 1. \
--lr 1e-4 \
${options}
"


if [ $1 == 0 ] 
then

echo python $cmd
python $cmd
break 100

else
echo python $cmd
sbatch <<< \
"#!/bin/bash
#SBATCH --job-name=${tag}
#SBATCH --output=slurm_logs/${tag}-%j-out.txt
#SBATCH --error=slurm_logs/${tag}-%j-err.txt
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --mem=48gb
#SBATCH --time=48:00:00

echo $cmd
python $cmd
"
fi

done

