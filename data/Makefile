test:
	./create_run.py \
	    --label TEST \
	    --Nsignals 20 \
	    --detectors H1 L1 \
	    --nprocesses 16 \
	    --injections \
	    --generate_noise \

cleantest:
	rm -r runfiles/TEST; \
	./create_run.py \
	    --label TEST \
	    --Nsignals 20 \
	    --detectors H1 L1 \
	    --nprocesses 16 \
	    --injections \
	    --generate_noise \
