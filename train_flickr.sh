dataname="flickr"
output_dir="sample"
distance="mae"
auto=1
lamda=0
a=1.0
b=2.0
c=1.0
d=2.0
e=2.0
f=1.0
g=1.0
train_bs=36
test_bs=36
temp=0.07
temp2=0.07
dist_url=tcp://127.0.0.1:5611
master_port=23734
CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 Retrieval.py --output_dir=$output_dir --dataname=$dataname --distance=$distance --auto=$auto --lamda=$lamda --bs=$train_bs --dist_url=$dist_url --a=$a --b=$b --c=$c --d=$d --e=$e --f=$f --g=$g --temp=$temp --temp2=$temp2
CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 Retrieval.py --output_dir=$output_dir --dataname=$dataname --distance=$distance --auto=$auto --lamda=$lamda --bs=$train_bs --dist_url=$dist_url --a=$a --b=$b --c=$c --d=$d --e=$e --f=$f --g=$g --temp=$temp --temp2=$temp2 --evaluate
