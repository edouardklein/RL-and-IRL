CFLAGS+=-I..

main.o: main.c
	gcc -c $(CFLAGS) main.c

phipsi.o: phipsi.c phipsi.h
	gcc -c $(CFLAGS) phipsi.c

VoidCriteria.o: VoidCriteria.c
	gcc -c $(CFLAGS) VoidCriteria.c

main: main.o phipsi.o ../utils.o ../LSTDmu.o ../greedy.o ../LSPI.o VoidCriteria.o ../LSTDQ.o
	gcc -o main $(LFLAGS)  main.o phipsi.o ../utils.o ../LSTDmu.o ../greedy.o ../LSPI.o VoidCriteria.o ../LSTDQ.o

../utils.o:
	make -C .. utils.o

../LSTDmu.o:
	make -C .. LSTDmu.o

../greedy.o:
	make -C .. greedy.o

../LSPI.o:
	make -C .. LSPI.o

../LSTDQ.o:
	make -C .. LSTDQ.o

clean:
	rm *.o main