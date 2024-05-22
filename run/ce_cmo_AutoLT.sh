# python main.py --dataset cifar100 --imb_ratio 100 --num_max 500 --epochs 200 \
#                --batch-size 256 --gpu 0 --cmo \
#                --aug_prob 0.5 --loss_fn ce --aug_type autoaug_cifar \
#                --doda --cutout \
#                > ./run/log/ce_cmo_doda.log 2>&1 &
echo "PID:$$";
if [ ! -d "./run/log/cifar100_ir=$1/" ];then
    mkdir ./run/log/cifar100_ir=$1/
fi
for seed in {0,}        
do 
    echo "using seed $seed" &&
    python main.py --dataset cifar100 --imb_ratio 100 --num_max 500 --epochs 200 \
               --batch-size 256 --gpu 0 --cmo --use_ael_reinforcement \
               --aug_prob 0.5 --loss_fn ce --aug_type autoaug_cifar \
               --seed $seed --AutoLT\
               > ./run/log/cifar100_ir=$1/ce_cmo_doda_$seed.log 2>&1 
done