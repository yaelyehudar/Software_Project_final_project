CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors
LDFLAGS = -lm
TARGET = symnmf
OBJECTS = symnmf.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

symnmf.o: symnmf.c symnmf.h
	$(CC) $(CFLAGS) -c symnmf.c

clean:
	rm -f $(TARGET) $(OBJECTS) *.so
	rm -rf build
