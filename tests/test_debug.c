#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "debug.h"

static char *read_file(const char *path) {
    FILE *file = fopen(path, "rb");
    assert(file);
    assert(fseek(file, 0, SEEK_END) == 0);
    long length = ftell(file);
    assert(length >= 0 && fseek(file, 0, SEEK_SET) == 0);
    char *text = malloc((size_t)length + 1);
    assert(text);
    assert(fread(text, 1, (size_t)length, file) == (size_t)length);
    text[length] = 0;
    assert(fclose(file) == 0);
    return text;
}

static size_t occurrences(const char *text, const char *needle) {
    size_t count = 0, length = strlen(needle);
    while ((text = strstr(text, needle)) != NULL) {
        ++count;
        text += length;
    }
    return count;
}

static void test_shared_graph(const char *path) {
    TensorMemoryStats baseline = tensor_memory_stats();
    Tensor *x = tensor_create(2);
    Tensor *y = tensor_create(3);
    Tensor *shared = tensor_mul(x, y);
    Tensor *root = tensor_add(shared, shared);
    assert(root && shared);
    assert(shared->ref_count == 3);
    assert(tensor_dump_dot(root, path));
    char *dot = read_file(path);
    assert(strstr(dot, "digraph tensors {") != NULL);
    assert(occurrences(dot, " [label=") == 4); /* Not one node per edge. */
    assert(occurrences(dot, " -> ") == 4);     /* Both shared-parent edges remain. */
    assert(occurrences(dot, "requires_grad=1") == 4);
    assert(strstr(dot, "shape=scalar"));
    assert(shared->op_name && strstr(dot, shared->op_name));
    free(dot);
    assert(tensor_check_finite(root, 1));
    tensor_release(root);
    tensor_release(shared);
    tensor_release(x);
    tensor_release(y);
    assert(tensor_memory_stats().live_tensors == baseline.live_tensors);
    assert(tensor_memory_stats().live_bytes == baseline.live_bytes);
}

static void test_finite_and_summary(const char *path) {
    TensorMemoryStats baseline = tensor_memory_stats();
    Tensor *frozen = tensor_create_matrix_ex(1, 3, 0);
    Tensor *trainable = tensor_create_matrix_ex(1, 3, 1);
    assert(frozen && trainable && frozen->grad == NULL && trainable->grad);
    Tensor *sum = tensor_add(frozen, trainable);
    assert(sum);
    assert(tensor_check_finite(sum, 1));
    assert(tensor_dump_dot(frozen, path));
    char *dot = read_file(path);
    assert(occurrences(dot, " [label=") == 1);
    assert(strstr(dot, "shape=(1,3)"));
    assert(strstr(dot, "requires_grad=0"));
    free(dot);

    FILE *summary = tmpfile();
    assert(summary);
    tensor_summary(frozen, summary);
    tensor_summary(trainable, summary);
    tensor_summary(NULL, summary);
    assert(fflush(summary) == 0 && fseek(summary, 0, SEEK_SET) == 0);
    char text[2048];
    size_t length = fread(text, 1, sizeof(text) - 1, summary);
    text[length] = 0;
    assert(strstr(text, "shape=(1,3)"));
    assert(strstr(text, "requires_grad=0"));
    assert(strstr(text, "ref_count="));
    assert(strstr(text, "data=[0, 0]"));
    assert(strstr(text, "grad=none"));
    assert(strstr(text, "live_tensors="));
    assert(strstr(text, "Tensor(NULL)"));
    assert(fclose(summary) == 0);

    trainable->grad[1] = INFINITY;
    assert(tensor_check_finite(sum, 0));
    assert(!tensor_check_finite(sum, 1));
    trainable->grad[1] = 0;
    frozen->data[2] = NAN;
    assert(!tensor_check_finite(sum, 0));
    summary = tmpfile();
    assert(summary);
    tensor_summary(frozen, summary);
    assert(fflush(summary) == 0 && fseek(summary, 0, SEEK_SET) == 0);
    length = fread(text, 1, sizeof(text) - 1, summary);
    text[length] = 0;
    assert(strstr(text, "nan=1"));
    assert(fclose(summary) == 0);
    frozen->data[2] = 0;
    sum->data[0] = -INFINITY;
    assert(!tensor_check_finite(sum, 1));
    sum->data[0] = 0;
    assert(tensor_check_finite(sum, 1));
    assert(!tensor_check_finite(NULL, 0));

    tensor_release(sum);
    tensor_release(frozen);
    tensor_release(trainable);
    assert(tensor_memory_stats().live_tensors == baseline.live_tensors);
    assert(tensor_memory_stats().live_bytes == baseline.live_bytes);
}

static void test_bad_paths_and_deep_graph(const char *path, const char *bad_path) {
    Tensor *leaf = tensor_create_ex(1, 0);
    assert(leaf && !leaf->grad);
    assert(!tensor_dump_dot(NULL, path));
    assert(!tensor_dump_dot(leaf, NULL));
    assert(!tensor_dump_dot(leaf, ""));
    assert(!tensor_dump_dot(leaf, bad_path));
    /* On systems providing /dev/full, buffered writes fail at flush/close. */
    if (access("/dev/full", W_OK) == 0) assert(!tensor_dump_dot(leaf, "/dev/full"));
    assert(leaf->ref_count == 1);
    assert(tensor_check_finite(leaf, 1));
    tensor_release(leaf);

    TensorMemoryStats baseline = tensor_memory_stats();
    Tensor *x = tensor_create(1);
    Tensor *root = x;
    tensor_retain(root);
    for (int i = 0; i < 12000; ++i) {
        Tensor *next = tensor_add(root, x);
        assert(next);
        tensor_release(root);
        root = next;
    }
    assert(tensor_check_finite(root, 1));
    assert(tensor_dump_dot(root, path));
    tensor_release(root);
    tensor_release(x);
    assert(tensor_memory_stats().live_tensors == baseline.live_tensors);
    assert(tensor_memory_stats().live_bytes == baseline.live_bytes);
}

int main(void) {
    char directory[] = "/tmp/minigrad-debug-XXXXXX";
    assert(mkdtemp(directory));
    char path[256], bad_path[256];
    assert(snprintf(path, sizeof(path), "%s/graph.dot", directory) > 0);
    assert(snprintf(bad_path, sizeof(bad_path), "%s/nonexistent/graph.dot", directory) > 0);
    test_shared_graph(path);
    test_finite_and_summary(path);
    test_bad_paths_and_deep_graph(path, bad_path);
    assert(remove(path) == 0);
    assert(rmdir(directory) == 0);
    puts("debug tests passed");
    return 0;
}
