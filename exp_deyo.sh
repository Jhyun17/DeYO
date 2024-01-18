GPU=0
LOG=0
METHOD=deyo
ETHR=0.5
EMAR=0.4
DTHR=0.3
INTERVAL=100
SEED=2024

ROOT='./data/' # Need to modify (your path of data root directory)

#### Mild setting ####
EXP=normal
MODEL=resnet50_bn_torch
INTERVAL=100
python main.py --method $METHOD --data_root $ROOT --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type $EXP --model $MODEL --wandb_interval $INTERVAL --seed $SEED --wandb_log $LOG

#### Wild settings ####
DTHR=0.2

MODEL=resnet50_gn_timm
# Mix_shifts setting
EXP=mix_shifts
INTERVAL=100
python main.py --method $METHOD --data_root $ROOT --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type $EXP --model $MODEL --wandb_interval $INTERVAL --seed $SEED --wandb_log $LOG

# Label_shifts setting
EXP=label_shifts
INTERVAL=100
python main.py --method $METHOD --data_root $ROOT --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type $EXP --model $MODEL --wandb_interval $INTERVAL --seed $SEED --wandb_log $LOG

# bs1 setting
EXP=bs1
INTERVAL=10000
python main.py --method $METHOD --data_root $ROOT --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type $EXP --model $MODEL --wandb_interval $INTERVAL --seed $SEED --wandb_log $LOG

#### Wild settings ####
MODEL=vitbase_timm
# Mix_shifts setting
EXP=mix_shifts
INTERVAL=100
python main.py --method $METHOD --data_root $ROOT --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type $EXP --model $MODEL --wandb_interval $INTERVAL --seed $SEED --wandb_log $LOG

# Label_shifts setting
EXP=label_shifts
INTERVAL=100
python main.py --method $METHOD --data_root $ROOT --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type $EXP --model $MODEL --wandb_interval $INTERVAL --seed $SEED --wandb_log $LOG

# bs1 setting
EXP=bs1
INTERVAL=10000
python main.py --method $METHOD --data_root $ROOT --dset ImageNet-C --gpu $GPU --plpd_threshold $DTHR --deyo_margin $ETHR --deyo_margin_e0 $EMAR --exp_type $EXP --model $MODEL --wandb_interval $INTERVAL --seed $SEED --wandb_log $LOG

#### Biased setttings ####

# Waterbirds
python pretrain_Waterbirds.py --root_dir $ROOT --dset_dir Waterbirds --gpu $GPU --seed $SEED

LRMUL=5
DTHR=0.5
MODEL=resnet50_bn_torch
INTERVAL=10
python main.py --method $METHOD --data_root $ROOT --dset Waterbirds --gpu $GPU --dset Waterbirds --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --wandb_log $LOG

# ColoredMNIST
python pretrain_ColoredMNIST.py --root_dir $ROOT --dset_dir ColoredMNIST --gpu $GPU --seed $SEED

ETHR=1.0
EMAR=1.0
MODEL=resnet18_bn
INTERVAL=30
python main.py --method $METHOD --data_root $ROOT --dset ColoredMNIST --gpu $GPU --dset ColoredMNIST --wandb_interval $INTERVAL --deyo_margin $ETHR --deyo_margin_e0 $EMAR --plpd_threshold $DTHR --lr_mul $LRMUL --exp_type spurious --model $MODEL --seed $SEED --wandb_log $LOG
