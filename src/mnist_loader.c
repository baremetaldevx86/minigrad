#include "mnist_loader.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* IDX stores header integers in network (big-endian) byte order. */
static int read_uint32(FILE *file, uint32_t *value) {
    unsigned char bytes[4];
    if (fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) return 0;
    *value = ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
             ((uint32_t)bytes[2] << 8) | (uint32_t)bytes[3];
    return 1;
}

MNISTData *load_mnist(const char *images_path, const char *labels_path) {
    if (!images_path || !labels_path) {
        fprintf(stderr, "minigrad: load_mnist: paths must not be NULL\n");
        return NULL;
    }

    FILE *images = fopen(images_path, "rb");
    if (!images) {
        perror(images_path);
        return NULL;
    }
    FILE *labels = fopen(labels_path, "rb");
    if (!labels) {
        perror(labels_path);
        fclose(images);
        return NULL;
    }

    MNISTData *data = NULL;
    unsigned char *row_buffer = NULL;
    uint32_t magic_img, n_img, rows, cols, magic_lbl, n_lbl;
    if (!read_uint32(images, &magic_img) || !read_uint32(images, &n_img) ||
        !read_uint32(images, &rows) || !read_uint32(images, &cols) ||
        !read_uint32(labels, &magic_lbl) || !read_uint32(labels, &n_lbl)) {
        fprintf(stderr, "minigrad: load_mnist: truncated IDX header\n");
        goto fail;
    }

    if (magic_img != 2051 || magic_lbl != 2049 || !n_img || n_img != n_lbl ||
        n_img > INT_MAX || !rows || !cols ||
        (size_t)rows > (size_t)INT_MAX / (size_t)cols) {
        fprintf(stderr, "minigrad: load_mnist: invalid IDX header\n");
        goto fail;
    }
    size_t count = (size_t)n_img;
    size_t dimension = (size_t)rows * (size_t)cols;
    if (dimension > SIZE_MAX / count ||
        count * dimension > SIZE_MAX / sizeof(float) ||
        count > SIZE_MAX / sizeof(int)) {
        fprintf(stderr, "minigrad: load_mnist: allocation size overflow\n");
        goto fail;
    }

    data = (MNISTData *)calloc(1, sizeof(*data));
    if (!data) goto out_of_memory;
    data->n_samples = (int)count;
    data->input_dim = (int)dimension;
    data->images = (float *)malloc(count * dimension * sizeof(float));
    data->labels = (int *)malloc(count * sizeof(int));
    row_buffer = (unsigned char *)malloc(dimension);
    if (!data->images || !data->labels || !row_buffer) goto out_of_memory;

    for (size_t i = 0; i < count; i++) {
        if (fread(row_buffer, 1, dimension, images) != dimension) {
            fprintf(stderr, "minigrad: load_mnist: truncated image data\n");
            goto fail;
        }
        for (size_t j = 0; j < dimension; j++) {
            data->images[i * dimension + j] = (float)row_buffer[j] / 255.0f;
        }
    }
    for (size_t i = 0; i < count; i++) {
        unsigned char label;
        if (fread(&label, 1, 1, labels) != 1) {
            fprintf(stderr, "minigrad: load_mnist: truncated label data\n");
            goto fail;
        }
        if (label > 9) {
            fprintf(stderr, "minigrad: load_mnist: invalid digit label\n");
            goto fail;
        }
        data->labels[i] = (int)label;
    }

    free(row_buffer);
    fclose(images);
    fclose(labels);
    printf("Loaded MNIST data: %d samples, %d features\n",
           data->n_samples, data->input_dim);
    return data;

out_of_memory:
    fprintf(stderr, "minigrad: load_mnist: out of memory\n");
fail:
    free(row_buffer);
    mnist_free(data);
    fclose(images);
    fclose(labels);
    return NULL;
}

void mnist_free(MNISTData *data) {
    if (!data) return;
    free(data->images);
    free(data->labels);
    free(data);
}
