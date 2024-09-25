CC = gcc
CFLAGS = -fopenmp -lm
SOURCES = src/main.c src/sparse_matrix_multiplication.c src/utils.c
TARGET = build/matvec_mul

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)

