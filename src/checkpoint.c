#define _POSIX_C_SOURCE 200809L
#include "checkpoint.h"

#include <float.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define MAX_LAYERS 65536u
#define MAX_PARAMS 16000000u
static const unsigned char magic[8] = {'M', 'G', 'M', 'L', 'P', '0', '0', '1'};

static int binary32_supported(void) {
    float one = 1.0f;
    uint32_t bits = 0;
    if (sizeof(float) != sizeof(bits) || FLT_RADIX != 2 ||
        FLT_MANT_DIG != 24 || FLT_MAX_EXP != 128) return 0;
    memcpy(&bits, &one, sizeof(bits));
    return bits == UINT32_C(0x3f800000);
}

static int finite_bits(uint32_t bits) {
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static int write_u32(FILE *f, uint32_t value) {
    unsigned char bytes[4];
    for (unsigned i = 0; i < 4; ++i) bytes[i] = (unsigned char)(value >> (8 * i));
    return fwrite(bytes, 1, sizeof(bytes), f) == sizeof(bytes);
}

static int read_u32(FILE *f, uint32_t *value) {
    unsigned char bytes[4];
    if (fread(bytes, 1, sizeof(bytes), f) != sizeof(bytes)) return 0;
    *value = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return 1;
}

static int write_u64(FILE *f, uint64_t value) {
    return write_u32(f, (uint32_t)value) && write_u32(f, (uint32_t)(value >> 32));
}

static int read_u64(FILE *f, uint64_t *value) {
    uint32_t lo, hi;
    if (!read_u32(f, &lo) || !read_u32(f, &hi)) return 0;
    *value = (uint64_t)lo | ((uint64_t)hi << 32);
    return 1;
}

static int valid_layer(const Linearlayer *l, int in, int out) {
    return l && l->in_features == in && l->out_features == out &&
           l->W && l->b && l->W->ndim == 2 && l->b->ndim == 2 &&
           l->W->shape && l->b->shape && l->W->data && l->b->data &&
           l->W->shape[0] == in && l->W->shape[1] == out &&
           l->b->shape[0] == 1 && l->b->shape[1] == out &&
           l->W->size == in * out && l->b->size == out;
}

static int write_tensor(FILE *f, const Tensor *t) {
    for (int i = 0; i < t->size; ++i) {
        uint32_t bits;
        memcpy(&bits, &t->data[i], sizeof(bits));
        if (!finite_bits(bits) || !write_u32(f, bits)) return 0;
    }
    return 1;
}

int mlp_save(const MLP *model, const char *path) {
    if (!binary32_supported() || !model || !model->layers ||
        model->n_layers <= 0 || (unsigned)model->n_layers > MAX_LAYERS ||
        !path || !*path) return 0;

    uint64_t count = 0;
    int previous = 0;
    for (int i = 0; i < model->n_layers; ++i) {
        const Linearlayer *l = model->layers[i];
        if (!l || l->in_features <= 0 || l->out_features <= 0 ||
            l->in_features > INT_MAX / l->out_features ||
            (i && l->in_features != previous) ||
            !valid_layer(l, l->in_features, l->out_features)) return 0;
        count += (uint64_t)l->in_features * (uint64_t)l->out_features +
                 (uint64_t)l->out_features;
        if (count > MAX_PARAMS) return 0;
        previous = l->out_features;
    }

    /* The temporary resides alongside the target so rename is atomic on the
     * same filesystem. A failed write never replaces an existing checkpoint. */
    size_t length = strlen(path);
    static const char suffix[] = ".tmp.XXXXXX";
    if (length > SIZE_MAX - sizeof(suffix)) return 0;
    char *temp = (char *)malloc(length + sizeof(suffix));
    if (!temp) return 0;
    memcpy(temp, path, length);
    memcpy(temp + length, suffix, sizeof(suffix));
    int fd = mkstemp(temp);
    if (fd < 0) { free(temp); return 0; }
    FILE *f = fdopen(fd, "wb");
    if (!f) { close(fd); unlink(temp); free(temp); return 0; }

    int ok = fwrite(magic, 1, sizeof(magic), f) == sizeof(magic) &&
             write_u32(f, 1) && write_u32(f, (uint32_t)model->n_layers) &&
             write_u64(f, count) &&
             write_u32(f, (uint32_t)model->layers[0]->in_features);
    for (int i = 0; ok && i < model->n_layers; ++i)
        ok = write_u32(f, (uint32_t)model->layers[i]->out_features);
    for (int i = 0; ok && i < model->n_layers; ++i) {
        ok = write_tensor(f, model->layers[i]->W) &&
             write_tensor(f, model->layers[i]->b);
    }
    if (ok) ok = fflush(f) == 0;
    if (ok) ok = fsync(fileno(f)) == 0;
    if (fclose(f) != 0) ok = 0;
    if (ok) ok = rename(temp, path) == 0;
    if (!ok) unlink(temp);
    free(temp);
    return ok;
}

MLP *mlp_load(const char *path) {
    if (!binary32_supported() || !path || !*path) return NULL;
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    MLP *model = NULL;
    int *dims = NULL;
    unsigned char file_magic[8];
    uint32_t version, layers;
    uint64_t declared;
    off_t file_size;
    if (fseeko(f, 0, SEEK_END) != 0 || (file_size = ftello(f)) < 0 ||
        fseeko(f, 0, SEEK_SET) != 0 ||
        fread(file_magic, 1, sizeof(file_magic), f) != sizeof(file_magic) ||
        memcmp(file_magic, magic, sizeof(magic)) != 0 ||
        !read_u32(f, &version) || version != 1 ||
        !read_u32(f, &layers) || !layers || layers > MAX_LAYERS ||
        !read_u64(f, &declared) || declared > MAX_PARAMS) goto done;

    /* Exact length and topology are checked before allocating any tensors.
     * The small dimension array is bounded by MAX_LAYERS. */
    uint64_t expected = 24u + 4u * ((uint64_t)layers + 1u) + 4u * declared;
    if ((uint64_t)file_size != expected) goto done;
    dims = (int *)malloc(((size_t)layers + 1u) * sizeof(*dims));
    if (!dims) goto done;
    uint64_t actual = 0;
    for (uint32_t i = 0; i <= layers; ++i) {
        uint32_t dim;
        if (!read_u32(f, &dim) || !dim || dim > INT_MAX) goto done;
        dims[i] = (int)dim;
        if (i) {
            if (dims[i - 1] > INT_MAX / dims[i]) goto done;
            actual += (uint64_t)dims[i - 1] * dim + dim;
            if (actual > declared) goto done;
        }
    }
    if (actual != declared) goto done;
    off_t parameter_offset = ftello(f);
    if (parameter_offset < 0) goto done;
    for (uint64_t i = 0; i < declared; ++i) {
        uint32_t bits;
        if (!read_u32(f, &bits) || !finite_bits(bits)) goto done;
    }
    if (fseeko(f, parameter_offset, SEEK_SET) != 0) goto done;

    // Loading replaces all initialized values; do not consume the caller's
    // legacy rand() stream as a side effect of constructing the wrappers.
    MinigradRNG initialization;
    rng_seed(&initialization, 0);
    model = mlp_create_ex(dims, (int)layers, INIT_HE, &initialization);
    if (!model) goto done;
    for (uint32_t i = 0; i < layers; ++i) {
        Tensor *params[2] = {model->layers[i]->W, model->layers[i]->b};
        for (int p = 0; p < 2; ++p) {
            for (int j = 0; j < params[p]->size; ++j) {
                uint32_t bits;
                if (!read_u32(f, &bits) || !finite_bits(bits)) goto failed_model;
                memcpy(&params[p]->data[j], &bits, sizeof(bits));
            }
        }
    }
    goto done;
failed_model:
    mlp_free(model);
    model = NULL;
done:
    free(dims);
    if (fclose(f) != 0) { mlp_free(model); model = NULL; }
    return model;
}
