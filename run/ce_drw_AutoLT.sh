# python main.py --dataset cifar100 --imb_ratio 100 --num_max 500 --epochs 200 \
#                --batch-size 256 --gpu 0 \
#                --aug_prob 0.5 --loss_fn ce_drw --aug_type autoaug_cifar \
#                --cutout --doda \
#                > ./run/log/ce_drw_doda.log 2>&1 
echo "PID:$$";
if [ ! -d "./run/log/cifar100_ir=$1/" ];then
    mkdir ./run/log/cifar100_ir=$1/
fi
for seed in {0,1,2}        
do 
    echo "using seed $seed" &&
    python main.py --dataset cifar100 --imb_ratio $1 --num_max 500 --epochs 200 \
               --batch-size 256 --gpu 7 --use_ael_reinforcement \
               --aug_prob 0.5 --loss_fn ce_drw --aug_type autoaug_cifar \
               --seed $seed --AutoLT --cutout \
               > ./run/log/cifar100_ir=$1/ce_drw_doda_$seed.log 2>&1 
done              