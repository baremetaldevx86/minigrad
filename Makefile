CC ?= cc
AR ?= ar
CPPFLAGS += -Iinclude
CFLAGS ?= -std=c11 -Wall -Wextra -Wpedantic -g -O2
LDFLAGS ?=
LDLIBS ?= -lm
SANFLAGS = -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer -O1 -g

LIBSRC := $(wildcard src/*.c)
LIBOBJ := $(patsubst src/%.c,build/normal/src/%.o,$(LIBSRC))
SANOBJ := $(patsubst src/%.c,build/sanitize/src/%.o,$(LIBSRC))
TESTSRC := $(wildcard tests/test*.c)
TESTS := $(patsubst tests/%.c,%,$(TESTSRC))
SANTESTS := $(addsuffix _sanitize,$(TESTS))
EXAMPLES := train train_xor train_mnist
EXAMPLEOBJ := $(addprefix build/normal/examples/,$(addsuffix .o,$(EXAMPLES)))
TESTOBJ := $(patsubst tests/%.c,build/normal/tests/%.o,$(TESTSRC))
SANTESTOBJ := $(patsubst tests/%.c,build/sanitize/tests/%.o,$(TESTSRC))
BENCHSRC := $(wildcard benchmarks/bench_matmul.c)
BENCHOBJ := $(patsubst benchmarks/%.c,build/normal/benchmarks/%.o,$(BENCHSRC))

.PHONY: all check check-sanitize bench clean
all: libminigrad.a $(EXAMPLES)

libminigrad.a: $(LIBOBJ)
	$(AR) rcs $@ $^

# Keep assertion checks active even when the caller passes CFLAGS=-DNDEBUG.
build/normal/tests/%.o: tests/%.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -UNDEBUG -MMD -MP -c $< -o $@

build/sanitize/tests/%.o: tests/%.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -UNDEBUG $(SANFLAGS) -MMD -MP -c $< -o $@

build/normal/src/%.o: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

build/sanitize/src/%.o: src/%.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SANFLAGS) -MMD -MP -c $< -o $@

build/normal/examples/%.o: examples/%.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

build/normal/benchmarks/%.o: benchmarks/%.c
	@mkdir -p $(@D)
	$(CC) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(EXAMPLES): %: build/normal/examples/%.o libminigrad.a
	$(CC) $(LDFLAGS) -o $@ $< libminigrad.a $(LDLIBS)

$(TESTS): %: build/normal/tests/%.o libminigrad.a
	$(CC) $(LDFLAGS) -o $@ $< libminigrad.a $(LDLIBS)

$(SANTESTS): %_sanitize: build/sanitize/tests/%.o $(SANOBJ)
	$(CC) $(LDFLAGS) $(SANFLAGS) -o $@ $^ $(LDLIBS)

check: $(TESTS)
	@set -e; $(foreach test,$(TESTS),./$(test);)

check-sanitize: $(SANTESTS)
	@set -e; $(foreach test,$(SANTESTS),ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 ./$(test);)

# The benchmark uses the same reusable library as examples and tests.
bench: bench_matmul
	./bench_matmul

ifneq ($(BENCHSRC),)
bench_matmul: build/normal/benchmarks/bench_matmul.o libminigrad.a
	$(CC) $(LDFLAGS) -o $@ $< libminigrad.a $(LDLIBS)
endif

-include $(LIBOBJ:.o=.d) $(SANOBJ:.o=.d) $(TESTOBJ:.o=.d) $(SANTESTOBJ:.o=.d) $(EXAMPLEOBJ:.o=.d) $(BENCHOBJ:.o=.d)

clean:
	rm -rf build libminigrad.a $(EXAMPLES) $(TESTS) $(SANTESTS) bench_matmul
