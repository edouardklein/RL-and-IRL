NBTRAININGEP=10

all:
	if [ -f TrainingSA-1.mat ]; then rm TrainingSA-*.mat; fi
	for i in `seq 1 $(NBTRAININGEP)`; do ./TrainingPipeline.py >> TrainingSA-$$i.mat; done
	./sa2sasreoe.py $(NBTRAININGEP)
	./SCIRL.py TrainingSASREOE.mat
	./RandomPipeline.py > RandomSamples.mat
	./ourLSPI.py

clean:
	rm omega_expert.mat *.pyc RandomSamples.mat theta_scirl.mat Training*.mat 
