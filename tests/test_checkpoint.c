#define _POSIX_C_SOURCE 200809L
#ifdef NDEBUG
#undef NDEBUG
#endif
#include "checkpoint.h"

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void put_u32(unsigned char *p, uint32_t n) {
    for (int i = 0; i < 4; ++i) p[i] = (unsigned char)(n >> (8 * i));
}

static void test_roundtrip(const char *path) {
    int sizes[] = {2, 3, 1};
    MLP *original = mlp_create(sizes, 2);
    assert(original);
    for (int i = 0; i < original->n_layers; ++i) {
        Linearlayer *l = original->layers[i];
        for (int j = 0; j < l->W->size; ++j)
            l->W->data[j] = (float)(j - 4 + i) * 0.125f;
        for (int j = 0; j < l->b->size; ++j)
            l->b->data[j] = (float)(j + i) * 0.0625f;
    }
    assert(mlp_save(original, path));
    srand(1701);
    int expected_random = rand();
    srand(1701);
    MLP *loaded = mlp_load(path);
    assert(rand() == expected_random); // Loading must not perturb initialization/shuffling.
    assert(loaded && loaded->n_layers == original->n_layers);
    for (int i = 0; i < loaded->n_layers; ++i) {
        Linearlayer *a = original->layers[i], *b = loaded->layers[i];
        assert(a->in_features == b->in_features && a->out_features == b->out_features);
        assert(memcmp(a->W->data, b->W->data, (size_t)a->W->size * sizeof(float)) == 0);
        assert(memcmp(a->b->data, b->b->data, (size_t)a->b->size * sizeof(float)) == 0);
        assert(b->W->grad && b->b->grad); /* Reloaded parameters are trainable. */
    }
    Tensor *input = tensor_create_matrix(2, 2);
    assert(input);
    input->data[0] = 0.5f; input->data[1] = -1.0f;
    input->data[2] = 3.0f; input->data[3] = 0.125f;
    Tensor *before = mlp_forward(original, input, 1);
    Tensor *after = mlp_forward(loaded, input, 1);
    assert(before && after && before->size == after->size);
    assert(memcmp(before->data, after->data, (size_t)before->size * sizeof(float)) == 0);
    tensor_backward(after);
    assert(loaded->layers[1]->b->grad[0] != 0.0f);
    tensor_release(before);
    tensor_release(after);
    tensor_release(input);

    /* Saving over an existing path replaces it only after full success. */
    assert(mlp_save(loaded, path));
    MLP *again = mlp_load(path);
    assert(again);
    mlp_free(again);
    original->layers[0]->W->data[0] = NAN;
    assert(!mlp_save(original, path));
    again = mlp_load(path);
    assert(again && again->layers[0]->W->data[0] == loaded->layers[0]->W->data[0]);
    mlp_free(again);
    mlp_free(original);
    mlp_free(loaded);
}

static unsigned char *read_file(const char *path, size_t *length) {
    FILE *f = fopen(path, "rb");
    assert(f);
    assert(fseek(f, 0, SEEK_END) == 0);
    long size = ftell(f);
    assert(size > 0);
    assert(fseek(f, 0, SEEK_SET) == 0);
    unsigned char *bytes = malloc((size_t)size + 1);
    assert(bytes);
    assert(fread(bytes, 1, (size_t)size, f) == (size_t)size);
    assert(fclose(f) == 0);
    *length = (size_t)size;
    return bytes;
}

static void reject_bytes(const char *path, const unsigned char *bytes, size_t size) {
    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite(bytes, 1, size, f) == size);
    assert(fclose(f) == 0);
    assert(mlp_load(path) == NULL);
}

static void test_corruption(const char *valid_path, const char *bad_path) {
    size_t size;
    unsigned char *valid = read_file(valid_path, &size);
    unsigned char *bad = malloc(size + 1);
    assert(bad && size > 40);
    for (size_t n = 0; n < size; ++n) reject_bytes(bad_path, valid, n);
    memcpy(bad, valid, size); bad[0] ^= 1;
    reject_bytes(bad_path, bad, size);
    memcpy(bad, valid, size); put_u32(bad + 8, 2); /* future version */
    reject_bytes(bad_path, bad, size);
    memcpy(bad, valid, size); put_u32(bad + 12, UINT32_MAX); /* huge topology */
    reject_bytes(bad_path, bad, size);
    memcpy(bad, valid, size); put_u32(bad + 16, UINT32_MAX); /* parameter cap */
    reject_bytes(bad_path, bad, size);
    memcpy(bad, valid, size); put_u32(bad + 24, 0); /* zero dimension */
    reject_bytes(bad_path, bad, size);
    memcpy(bad, valid, size); put_u32(bad + 24, INT_MAX); /* product overflow */
    reject_bytes(bad_path, bad, size);
    memcpy(bad, valid, size); put_u32(bad + 28, 9); /* wrong parameter count */
    reject_bytes(bad_path, bad, size);
    memcpy(bad, valid, size); put_u32(bad + 36, UINT32_C(0x7f800000)); /* infinity */
    reject_bytes(bad_path, bad, size);
    memcpy(bad, valid, size); bad[size] = 0; /* trailing data */
    reject_bytes(bad_path, bad, size + 1);
    free(bad);
    free(valid);
}

int main(void) {
    char dir[] = "/tmp/minigrad-checkpoint-XXXXXX";
    assert(mkdtemp(dir));
    char path[sizeof(dir) + 32], bad[sizeof(dir) + 32], missing[sizeof(dir) + 32];
    assert(snprintf(path, sizeof(path), "%s/model.bin", dir) > 0);
    assert(snprintf(bad, sizeof(bad), "%s/bad.bin", dir) > 0);
    assert(snprintf(missing, sizeof(missing), "%s/no-such-dir/model", dir) > 0);
    assert(!mlp_load(missing));
    assert(!mlp_load(dir));
    assert(!mlp_save(NULL, path));
    assert(!mlp_save(NULL, NULL));
    int dims[] = {2, 1};
    MLP *model = mlp_create(dims, 1);
    assert(model);
    assert(!mlp_save(model, missing));
    assert(!mlp_save(model, dir));
    mlp_free(model);
    test_roundtrip(path);
    test_corruption(path, bad);
    assert(unlink(path) == 0);
    assert(unlink(bad) == 0);
    assert(rmdir(dir) == 0);
    puts("checkpoint tests passed");
    return 0;
}
