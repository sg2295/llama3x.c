CC = clang
CFLAGS = -O3 -ffast-math -Xclang -fopenmp -std=c23
LDFLAGS = -L/opt/homebrew/opt/libomp/lib -lomp
CPPFLAGS = -I/opt/homebrew/opt/libomp/include

run: run.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $(LDFLAGS) run.c -o run

clean:
	rm -f run

.PHONY: clean
