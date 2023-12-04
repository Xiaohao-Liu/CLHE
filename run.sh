python3 -u train.py -g 0 --dataset="pog_dense" --model="CLHE" --item_augment="MD" --bundle_augment="ID" --bundle_ratio=0.5 --bundle_cl_temp=0.05 --bundle_cl_alpha=2 --cl_temp=0.5 --cl_alpha=0.1


python3 -u train.py -g 0 --dataset="pog" --model="CLHE" --item_augment="FN" --bundle_augment="ID" --bundle_ratio=0.5 --bundle_cl_temp=0.01 --bundle_cl_alpha=0.5 --cl_temp=0.5 --cl_alpha=2


python3 -u train.py -g 0 --dataset="spotify" --model="CLHE" --item_augment="MD" --bundle_augment="ID" --bundle_ratio=0.05 --bundle_cl_temp=0.2 --bundle_cl_alpha=0.5 --cl_temp=0.05 --cl_alpha=0.1


python3 -u train.py -g 0 --dataset="spotify_sparse" --model="CLHE" --item_augment="MD" --bundle_augment="ID" --bundle_ratio=0.05 --bundle_cl_temp=1 --bundle_cl_alpha=1 --cl_temp=0.2 --cl_alpha=0.1
