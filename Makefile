LDFLAG = -ltensorflow
run: a.out
	./a.out

a.out: main.c
	$(CC) main.c $(LDFLAG)

clean:
	$(RM) a.out
