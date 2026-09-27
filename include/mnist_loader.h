#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

typedef struct {
    float* images;  // Flattened images (n_samples x input_dim), values [0, 1]
    int* labels;    // Labels (n_samples), values [0-9]
    int n_samples;
    int input_dim;  // rows * cols from IDX header; normally 784 (28x28)
} MNISTData;

// Load MNIST data from IDX files
// Returns NULL for invalid, truncated or unopenable files and allocation failure.
// Caller owns the result and must release it with mnist_free().
MNISTData* load_mnist(const char* images_path, const char* labels_path);

void mnist_free(MNISTData* data);

#endif
