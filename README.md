# minigrad

A small dependency-free C11 reverse-mode autodiff engine for scalars and contiguous 2D matrices. The repository includes an MLP, MNIST IDX loader, optimizers, numerical gradient checking, safe checkpoints, diagnostics and a matmul benchmark. The original 784→256→128→10 MNIST training run reported **98.01% test accuracy** (SGD, 20 epochs, batch 32); this is a historical example result, not a benchmark of the current kernels.

<img width="823" height="780" alt="mnist-training" src="https://github.com/user-attachments/assets/bc408345-be08-4b17-a631-2288f440e64d" />

## Build and run

Requires a C11 compiler and `libm`, with no third-party libraries. Checkpoint persistence and temporary-file tests use POSIX filesystem APIs (for example Linux/macOS). `make` also produces `libminigrad.a` for linking your own programs.

```sh
make                 # train, train_xor, train_mnist
make check           # complete test suite (assertions remain enabled)
make check-sanitize  # isolated ASan/UBSan tests with leak detection
make bench           # build and run rectangular matmul forward/backward benchmark
./train_xor
./train_mnist         # needs raw MNIST IDX files in data/
```

For MNIST, supply `data/train-images-idx3-ubyte`, `data/train-labels-idx1-ubyte`, `data/t10k-images-idx3-ubyte`, and `data/t10k-labels-idx1-ubyte`. `make bench` reports measured timings and checksums on your own machine; no speedup is assumed. Debugging via `tensor_dump_dot` produces a `.dot` file you can optionally render with Graphviz (Graphviz is not required to build or run minigrad).

## Tensor ownership, tracking and backward

Include `engine.h` (and other feature headers as needed). `tensor_create(x)` and `tensor_create_matrix(rows, cols)` create **trainable** leaves by default. To avoid allocating gradient buffers for inputs and labels, use `tensor_create_ex(x, 0)` or `tensor_create_matrix_ex(rows, cols, 0)`; the final argument is `requires_grad`. Frozen tensors have `grad == NULL`. `tensor_set_requires_grad(t, enabled)` changes tracking only on an unshared leaf (no attached graph references); it returns 1 on success, 0 on failure. Do not modify graph metadata, shapes, tracking fields or reference counts yourself; fill `data` via the public buffer before forward and leave it unchanged until backward finishes.

Each returned tensor is an **owned reference**: release it exactly once with `tensor_release`. Tracked operation results retain *all* parents, including frozen ones whose values are needed in backward. You may release your parent reference after creating the result; the graph keeps the data alive. A no-grad/untracked operation result stores **no parents and no gradient buffer**; if you need an input again, retain your own reference. Do not use a pointer after its final release. `tensor_retain` takes an additional reference; `tensor_release(NULL)` is safe. Model layers own their parameter tensors; `mlp_params`/`linear_params` return a caller-owned **array** of borrowed parameters: free that array separately, and keep the model alive as long as you use the parameters. Optimizers also borrow both that array and its tensors, so free the optimizer before freeing either one.

`grad_set_enabled(0)` temporarily disables graph construction for operation results and returns the previous (thread-local) setting; restore it with `grad_set_enabled(previous)` on **all** paths. Constructors still honor their own `requires_grad` argument regardless of the current grad mode. Use no-grad for inference when backpropagation is unnecessary:

```c
int previous = grad_set_enabled(0);
Tensor *prediction = mlp_forward(model, input, 1);
grad_set_enabled(previous);   /* restore even if prediction == NULL */
if (prediction) tensor_release(prediction);
```

`tensor_backward_with_grad(output, upstream)` returns 1 on success, 0 for invalid inputs/seed shapes; `upstream == NULL` supplies all ones. The legacy `tensor_backward(output)` returns void and uses that all-ones seed, differentiating the sum for a nonscalar output. A non-NULL upstream must have the **same shape** as the output; it is read-only. **Leaf gradients accumulate across backward calls and across graphs**, including backward on a leaf root. Intermediate gradients are cleared before every traversal, preventing repeated graph propagation from stale intermediates. Explicitly call `tensor_zero_grad(leaf)` or `sgd_zero_grad`/`adam_zero_grad` before a fresh optimization step. No-grad outputs cannot be differentiated. `tensor_memory_stats()` reports live tensor counts and tensor-owned bytes (including a peak); it excludes traversal and optimizer allocations. Tensor reference counts and these global counters are not thread-safe; thread-local grad mode alone does not make the engine safe for concurrent use.

```c
Tensor *x = tensor_create_matrix_ex(32, 784, 0);
Tensor *labels = tensor_create_matrix_ex(32, 10, 0);
/* Fill x->data and labels->data, then compute logits from a model. */
Tensor *logits = mlp_forward(model, x, 1);
Tensor *loss = logits ? cross_entropy_loss(logits, labels) : NULL;
if (loss) {
    sgd_zero_grad(optimizer);
    tensor_backward(loss);
    sgd_step(optimizer);
}
tensor_release(loss);
tensor_release(logits);
tensor_release(labels);
tensor_release(x);
```

## Differentiable operations and loss

Core operations include `tensor_add`, `tensor_sub`, `tensor_mul`, `tensor_div`, `tensor_pow`, `tensor_expn`, `tensor_sqrt`, `tensor_relu`, `tensor_Tanh`, `tensor_mean`, `tensor_softmax` (row-wise 2D) and `tensor_matmul` (2D). Elementwise binary operations broadcast matching 2D axes of size 1: scalar/2D, row `(1, cols)`, column `(rows, 1)`, and `(1, 1)` are supported. Expanded dimensions reduce gradients into the operand. This is **not** general N-dimensional broadcasting. Matrix multiplication requires compatible 2D shapes.

`tensor_ops.h` adds elementwise `tensor_log`, `tensor_sigmoid` and `tensor_softplus`; `tensor_sum_axis(a, axis)` and `tensor_mean_axis(a, axis)` use `axis = -1` for an all-element scalar reduction or axis 0/1 for a 2D keep-dimension result. `tensor_transpose(a)` (2D only) and `tensor_reshape(a, rows, cols)` return **independent contiguous copies**, not shared-storage views. Reshape permits scalar/2D input and requires the same element count; no aliasing or in-place changes are propagated to the input. `tensor_log` follows libm's domain behavior (`log(0) = -Inf`, negative values yield NaN); ordinary math operations do not clamp invalid domains.

`mse_loss(pred, target)` builds mean squared error from differentiable operations. `cross_entropy_loss(logits, targets)` is a fused, stable log-sum-exp loss returning a scalar batch mean. Both inputs must be equal, nonempty `(batch, classes)` matrices. Logits must be finite; targets must be finite and nonnegative, but may be **unnormalized weights**, not just one-hot labels or probabilities. Nonfinite/unrepresentable loss or invalid shape returns `NULL`. The backward pass updates either or both trainable operands using the upstream scalar seed:

```text
loss      = -sum_rows sum_classes target * logsoftmax(logits) / batch
∂logits   = (softmax(logits) * sum_classes(target) - target) / batch
∂targets  = -logsoftmax(logits) / batch
```

Internally the fused loss uses double precision for log-sum-exp/weighted sums and differences between extreme finite float logits, before storing the scalar result/gradients as float. A mathematical derivative outside float range can still overflow its float gradient buffer. When no-grad is active or neither operand tracks gradients, loss has no graph or grad buffer.

## Training utilities

- `optim.h`: `sgd_create(params, n, lr)` and `sgd_create_momentum(params, n, lr, momentum)` (`v = momentum*v + grad`, then `p -= lr*v`); `adam_create` and `adamw_create(params, n, lr, weight_decay)` with standard Adam moments/bias correction and decoupled AdamW weight decay. Use `sgd_step`, `sgd_zero_grad`, `sgd_set_lr`, `sgd_free` or corresponding `adam_*` functions. The optimizer **borrows** the parameter array and tensors but owns its state buffers; frozen/no-grad parameters are skipped. Duplicate parameters are rejected. `optim_clip_grad_norm(params, n, max_norm)` clips in place and returns the *preclip* L2 norm, or NaN without modification on invalid/nonfinite inputs.
- `rng.h`, `nn.h`, `mlp.h`: seed a `MinigradRNG` with `rng_seed(&rng, seed)`; `rng_uniform` and `rng_normal` use an explicit SplitMix64 state with cached Box–Muller normal values. `linear_create_ex(in, out, INIT_HE /* or INIT_XAVIER */, &rng)` and `mlp_create_ex(sizes, n_layers, INIT_HE, &rng)` use that reproducible stream and do not touch global `rand`. Existing `linear_create`/`mlp_create` remain He-initialized with the legacy `rand` stream; passing `NULL` RNG to the `_ex` constructors also uses it. Biases initialize to zero. `linear_forward`, `mlp_forward`, `linear_free` and `mlp_free` manage their usual layer/model ownership.
- `gradcheck.h`: `tensor_gradcheck(fn, inputs, n_inputs, context, upstream, epsilon, atol, rtol)` compares reverse-mode VJPs against centered finite differences and returns 1 for agreement. The callback returns one owned output; inputs must be distinct, trainable, unshared leaves. An optional upstream selects a nonscalar VJP; NULL means ones. The checker restores input data, leaf gradients and grad mode, and releases callback results.
- `checkpoint.h`: `mlp_save(model, path)` returns 1/0; `mlp_load(path)` returns a new owned, trainable MLP or NULL. The versioned, little-endian format stores topology and finite IEEE-754 binary32 weights/biases, with checked dimensions, truncation and resource limits (65,536 layers / 16 million parameters). Loading does not consume the global `rand()` stream. It does **not** store optimizer state, RNG state, gradients or training mode. A load starts with fresh trainable parameters; use `mlp_free` on the result. Failed saves do not replace an existing checkpoint.
- `debug.h`: `tensor_dump_dot(root, path)` writes the reachable computation graph in Graphviz DOT format (1 success/0 failure); `tensor_check_finite(root, check_gradients)` checks graph data and optionally allocated gradients (1/0); `tensor_summary(tensor, stream)` prints shapes, tracking and value/gradient ranges. These diagnostics work without Graphviz installed. `tensor_memory_stats` is declared in `engine.h`.
- `kernels.h`: `mg_matmul_forward(a, b, out, m, n, k)` writes contiguous row-major `(m,n) @ (n,k)` into `(m,k)`; `mg_matmul_backward(a, b, upstream, da, db, m, n, k)` **accumulates** gradients into `da` and/or `db` (pass NULL to skip one). These are low-level pure-C kernels; call `tensor_matmul` instead when you need shape validation and autograd graph tracking.

## Scope and limitations

The current implementation is a scalar/contiguous-2D educational engine with per-tensor owned data. It does **not** implement arbitrary-rank tensors, shared-storage views, higher-order automatic differentiation, distributed/GPU execution or optional BLAS integration. Transpose and reshape are copied, not lazy, and cannot share storage; checkpoint loading restores model topology and values only. Matmul has pure-C kernels and a reproducible local benchmark; consult the benchmark output instead of assuming performance figures. Allocation failures in core tensor/graph construction follow the documented fail-fast policy; invalid inputs to pointer-returning APIs return NULL. Floating-point undefined domains retain IEEE/libm behavior unless an API explicitly validates them.

## License

MIT
