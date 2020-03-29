for seed in 1 2 3 4 5 6 7 8 9 10
do
	python3.6 run_synthetic.py --n 1000 --d1 3 --d2 2 --noise_level 0.1 --seed_data $seed \
	                        --train_iter 2000 --batch_size 100 \
	                        --visual 0 --save_val_result experiments/all_result.txt
done

for seed in 1 2 3 4 5 6 7 8 9 10
do
	python3.6 run_synthetic.py --n 10000 --d1 3 --d2 2 --noise_level 0.1 --seed_data $seed \
	                        --train_iter 2000 --batch_size 100 \
	                        --visual 0 --save_val_result experiments/all_result.txt
done

for seed in 1 2 3 4 5 6 7 8 9 10
do
	python3.6 run_synthetic.py --n 100000 --d1 3 --d2 2 --noise_level 0.1 --seed_data $seed \
	                        --train_iter 2000 --batch_size 100 \
	                        --visual 0 --save_val_result experiments/all_result.txt
done
 
