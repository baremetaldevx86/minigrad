
CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -Wpedantic -g -O2 -Iinclude
LDFLAGS = -lm
SANFLAGS = -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -O1 -g
VPATH   = src:examples:tests
HEADERS = $(wildcard include/*.h)
LIBSRC  = $(wildcard src/*.c)

all: train train_xor train_mnist

.PHONY: all check check-sanitize clean

check: test test_engine test_safety
	./test
	./test_engine
	./test_safety

# Build separately so sanitizer and normal object files never get mixed.
check-sanitize: test_sanitize test_engine_sanitize test_safety_sanitize
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 ./test_sanitize
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 ./test_engine_sanitize
	ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 ./test_safety_sanitize

train: train.o engine.o nn.o mlp.o loss.o optim.o
	$(CC) $(CFLAGS) -o train train.o engine.o nn.o mlp.o loss.o optim.o $(LDFLAGS)

train_xor: train_xor.o engine.o nn.o mlp.o loss.o optim.o
	$(CC) $(CFLAGS) -o train_xor train_xor.o engine.o nn.o mlp.o loss.o optim.o $(LDFLAGS)

train_mnist: train_mnist.o engine.o nn.o mlp.o loss.o optim.o mnist_loader.o
	$(CC) $(CFLAGS) -o train_mnist train_mnist.o engine.o nn.o mlp.o loss.o optim.o mnist_loader.o $(LDFLAGS)

test: test.o engine.o
	$(CC) $(CFLAGS) -o test test.o engine.o $(LDFLAGS)

test_engine: test_engine.o engine.o loss.o
	$(CC) $(CFLAGS) -o test_engine test_engine.o engine.o loss.o $(LDFLAGS)

test_safety: test_safety.o engine.o loss.o nn.o mlp.o optim.o mnist_loader.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Keep assertions active even in builds configured with -DNDEBUG.
test.o test_engine.o test_safety.o: CFLAGS += -UNDEBUG

%_sanitize: tests/%.c $(LIBSRC) $(HEADERS)
	$(CC) $(CFLAGS) -UNDEBUG $(SANFLAGS) -o $@ $(filter %.c,$^) $(LDFLAGS) $(SANFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

-include $(wildcard *.d)

clean:
	rm -f *.o *.d train test test_engine test_safety *_sanitize train_xor train_mnist
