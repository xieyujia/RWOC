for seed in 0 1 2 3 4 5 6 7 8 9 
do
	    python main.py --n 1000 --d 5 --shuffle_level 0.5 --snr 100 --seed_data $seed \
	                        --train_iter 50 --batch_size 1000  --method sinkhorn_stablized \
	                        --visual 0 --save_val_result experiments/result_1123_robot_shuffle05_d2_snr100_bs1000_lr-4.txt
done

 
