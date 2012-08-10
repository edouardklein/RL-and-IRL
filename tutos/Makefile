all: SCIRL_GW SCIRL_IP

SCIRL_GW:
	python SCIRL_GridWorld.py D_GW.mat 

SCIRL_IP: LSTDmu.exe
	./LSTDmu.exe D_IP.mat > omega_mu_E.mat
	python SCIRL_InvertedPendulum.py D_IP.mat omega_mu_E.mat 

LSTDmu.exe: LSTDmu.c
	gcc -c `pkg-config --cflags gsl` --std=c99 LSTDmu.c
	gcc -o LSTDmu.exe `pkg-config --libs gsl` LSTDmu.o 

clean:
	rm -r *.pyc *.o *.exe *~ ltxpng omega*.mat
