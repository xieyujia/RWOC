for seed in 1 2 3 .. 10
do
	python run_synthetic.py --n 1000 --d1 3 --d2 2 --noise_level 0.1 --seed_data $seed \
	                        --train_iter 2000 --batch_size 100 \
	                        --visual 0 --save_val_result expriments/all_result.txt
done

for seed in 1 2 3 .. 10
do
	python run_synthetic.py --n 10000 --d1 3 --d2 2 --noise_level 0.1 --seed_data $seed \
	                        --train_iter 2000 --batch_size 100 \
	                        --visual 0 --save_val_result expriments/all_result.txt
done

for seed in 1 2 3 .. 10
do
	python run_synthetic.py --n 100000 --d1 3 --d2 2 --noise_level 0.1 --seed_data $seed \
	                        --train_iter 2000 --batch_size 100 \
	                        --visual 0 --save_val_result expriments/all_result.txt
done
 