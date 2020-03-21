import mxnet

__all__ = ['NAGSparse', 'NadamSparse']

class _sparse():
    def _update_mask(self, t, name, initial_sparsity, weight, mask):
        if all((k not in name) for k in self.keywords_no_sparse) and (self.pretrain_step <= t < self.pretrain_step + self.sparse_step) and ((t - self.pretrain_step) % self.frequency == 0):
            target_sparsity = self.special_sparsity_dict.get(name, self.target_sparsity)
            current_sparse_step = (t - self.pretrain_step) // self.frequency
            total_srarse_step = (self.sparse_step) // self.frequency
            current_sparsity = target_sparsity + (initial_sparsity - target_sparsity) * (1.0 - current_sparse_step / total_srarse_step) ** 3
            keep_k = int(max(weight.size * (1.0 - current_sparsity), 1.0))
            # mask[:] = mxnet.ndarray.reshape(mxnet.ndarray.topk(weight.abs().reshape(-1), k=keep_k, ret_typ='mask'), mask.shape)

            if len(weight.shape) == 1:
                setup = self.sparse_block['1d']
            elif len(weight.shape) == 2:
                setup = self.sparse_block['2d']
            elif len(weight.shape) == 4:
                setup = self.sparse_block['4d']

            loop = []
            stride = []
            for i, tmp in enumerate(setup):
                if tmp > 0:
                    loop.append(tmp)
                else:
                    loop.append(weight.shape[i])
                stride.append(weight.shape[i] // loop[i])
            total_block = mxnet.ndarray.array(loop, mxnet.cpu(0), 'int32').prod().asnumpy().tolist()[0]
            base_num = mxnet.ndarray.ones(loop, mxnet.cpu(0), 'int32') * (keep_k // total_block)
            remain_num = keep_k % total_block
            if remain_num != 0:
                tmp1 = mxnet.ndarray.ones(remain_num, mxnet.cpu(0), 'int32')
                tmp2 = mxnet.ndarray.zeros(total_block - remain_num, mxnet.cpu(0), 'int32')
                tmp3 = mxnet.ndarray.concat(tmp1, tmp2, dim=0).reshape(loop)
                base_num = base_num + tmp3
            base_num = base_num.asnumpy()

            if len(weight.shape) == 1:
                for b0 in range(loop[0]):
                    block_weight = weight[
                        b0 * stride[0] : b0 * stride[0] + stride[0],
                        ].copyto(mxnet.cpu())
                    block_mask = mask[
                        b0 * stride[0] : b0 * stride[0] + stride[0],
                        ].copyto(mxnet.cpu())
                    tmp_k = base_num[
                        b0,
                        ].tolist()
                    if tmp_k >= 1:
                        block_mask = mxnet.ndarray.where(block_weight.abs() >= mxnet.ndarray.topk(block_weight.abs().reshape(-1), k=tmp_k, ret_typ='both')[0][-1], mxnet.ndarray.ones_like(block_mask), mxnet.ndarray.zeros_like(block_mask))
                    else:
                        block_mask = block_mask * 0
                    mask[
                        b0 * stride[0] : b0 * stride[0] + stride[0],
                        ] = block_mask
            elif len(weight.shape) == 2:
                for b0 in range(loop[0]):
                    for b1 in range(loop[1]):
                        block_weight = weight[
                            b0 * stride[0] : b0 * stride[0] + stride[0],
                            b1 * stride[1] : b1 * stride[1] + stride[1],
                            ].copyto(mxnet.cpu())
                        block_mask = mask[
                            b0 * stride[0] : b0 * stride[0] + stride[0],
                            b1 * stride[1] : b1 * stride[1] + stride[1],
                            ].copyto(mxnet.cpu())
                        tmp_k = base_num[
                            b0,
                            b1,
                            ].tolist()
                        if tmp_k >= 1:
                            block_mask = mxnet.ndarray.where(block_weight.abs() >= mxnet.ndarray.topk(block_weight.abs().reshape(-1), k=tmp_k, ret_typ='both')[0][-1], mxnet.ndarray.ones_like(block_mask), mxnet.ndarray.zeros_like(block_mask))
                        else:
                            block_mask = block_mask * 0
                        mask[
                            b0 * stride[0] : b0 * stride[0] + stride[0],
                            b1 * stride[1] : b1 * stride[1] + stride[1],
                            ] = block_mask
            elif len(weight.shape) == 4:
                for b0 in range(loop[0]):
                    for b1 in range(loop[1]):
                        for b2 in range(loop[2]):
                            for b3 in range(loop[3]):
                                block_weight = weight[
                                    b0 * stride[0] : b0 * stride[0] + stride[0],
                                    b1 * stride[1] : b1 * stride[1] + stride[1],
                                    b2 * stride[2] : b2 * stride[2] + stride[2],
                                    b3 * stride[3] : b3 * stride[3] + stride[3],
                                    ].copyto(mxnet.cpu())
                                block_mask = mask[
                                    b0 * stride[0] : b0 * stride[0] + stride[0],
                                    b1 * stride[1] : b1 * stride[1] + stride[1],
                                    b2 * stride[2] : b2 * stride[2] + stride[2],
                                    b3 * stride[3] : b3 * stride[3] + stride[3],
                                    ].copyto(mxnet.cpu())
                                tmp_k = base_num[
                                    b0,
                                    b1,
                                    b2,
                                    b3,
                                    ].tolist()
                                if tmp_k >= 1:
                                    block_mask = mxnet.ndarray.where(block_weight.abs() >= mxnet.ndarray.topk(block_weight.abs().reshape(-1), k=tmp_k, ret_typ='both')[0][-1], mxnet.ndarray.ones_like(block_mask), mxnet.ndarray.zeros_like(block_mask))
                                else:
                                    block_mask = block_mask * 0
                                mask[
                                    b0 * stride[0] : b0 * stride[0] + stride[0],
                                    b1 * stride[1] : b1 * stride[1] + stride[1],
                                    b2 * stride[2] : b2 * stride[2] + stride[2],
                                    b3 * stride[3] : b3 * stride[3] + stride[3],
                                    ] = block_mask
            weight[:] *= mask

@mxnet.optimizer.Optimizer.register
class NAGSparse(mxnet.optimizer.Optimizer, _sparse):
    """
        NAG优化器的稀疏化版本。

        Parameters
        ----------
        momentum : float, optional

            The momentum value.

        multi_precision: bool, optional

            Flag to control the internal precision of the optimizer.
            False: results in using the same precision as the weights (default),
            True: makes internal 32-bit copy of the weights and applies gradients in 32-bit precision even if actual weights used in the model have lower precision.
            Turning this on can improve convergence and accuracy when training with float16.
    """
    def __init__(self,
        momentum=0,
        L1_regularization=False,
        target_sparsity=0,
        pretrain_step=0,
        sparse_step=0,
        frequency=100,
        keywords_no_sparse=['bias', 'beta', 'gamma'],
        special_sparsity_dict={},
        sparse_block={'1d': [1], '2d': [1, 1], '4d': [-1, 4, -1, 1]},
        resume=False,
        **kwargs):
        super(NAGSparse, self).__init__(**kwargs)
        self.momentum = momentum
        self.L1_regularization = L1_regularization
        self.target_sparsity = target_sparsity
        self.pretrain_step = pretrain_step
        self.sparse_step = sparse_step
        self.frequency = frequency
        self.keywords_no_sparse = keywords_no_sparse
        self.special_sparsity_dict = special_sparsity_dict
        self.sparse_block = sparse_block
        self.resume = resume

    def create_state(self, index, weight):
        momentum = mxnet.ndarray.zeros(weight.shape, weight.context, dtype=weight.dtype)

        if self.resume == True:
            cpu_weight = weight.copyto(mxnet.cpu())
            ones = mxnet.ndarray.ones_like(cpu_weight)
            zeros = mxnet.ndarray.zeros_like(cpu_weight)
            cpu_mask = mxnet.ndarray.where(cpu_weight == 0, zeros, ones)
            initial_sparsity = 1 - cpu_mask.sum().asnumpy()[0] / cpu_mask.size
            mask = cpu_mask.copyto(weight.context)
        else:
            initial_sparsity = 0
            mask = mxnet.ndarray.ones_like(weight)
        return (momentum, initial_sparsity, mask)

    def _get_names(self, indices):
        names = ['' for _ in indices]
        for i, index in enumerate(indices):
            if index in self.param_dict:
                names[i] = self.param_dict[index].name
            elif index in self.lr_mult:
                names[i] = self.lr_mult[index].name
            elif index in self.idx2name:
                names[i] = self.lr_mult.get(self.idx2name[index], 1.0).name
        return names
    def _get_name(self, index):
        return self._get_names([index])[0]

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, mxnet.ndarray.NDArray))
        assert(isinstance(grad, mxnet.ndarray.NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        t = self._index_update_count[index]

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = mxnet.ndarray.clip(grad, -self.clip_gradient, self.clip_gradient)

        name = self._get_name(index)
        m, initial_sparsity, mask = state
        # grad *= mask
        m[:] = self.momentum * m + grad + wd * weight
        if self.L1_regularization == True:
            m[:] = wd / 2 * mxnet.ndarray.sign(weight)
        weight[:] = weight - (lr * (grad + self.momentum * m)) * mask

        self._update_mask(t, name, initial_sparsity, weight, mask)

@mxnet.optimizer.Optimizer.register
class NadamSparse(mxnet.optimizer.Optimizer, _sparse):
    """
        Nadam优化器的稀疏化版本。

        Parameters
        ----------
        beta1 : float, optional

            Exponential decay rate for the first moment estimates.

        beta2 : float, optional

            Exponential decay rate for the second moment estimates.

        epsilon : float, optional

            Small value to avoid division by 0.

        schedule_decay : float, optional

            Exponential decay rate for the momentum schedule
    """
    def __init__(self,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        schedule_decay=0.004,
        L1_regularization=False,
        target_sparsity=0,
        pretrain_step=0,
        sparse_step=0,
        frequency=100,
        keywords_no_sparse=['bias', 'beta', 'gamma'],
        special_sparsity_dict={},
        sparse_block={'1d': [1], '2d': [1, 1], '4d': [-1, 4, -1, 1]},
        resume=False,
        **kwargs):
        super(NadamSparse, self).__init__(**kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay
        self.m_schedule = 1.
        self.L1_regularization = L1_regularization
        self.target_sparsity = target_sparsity
        self.pretrain_step = pretrain_step
        self.sparse_step = sparse_step
        self.frequency = frequency
        self.keywords_no_sparse = keywords_no_sparse
        self.special_sparsity_dict = special_sparsity_dict
        self.sparse_block = sparse_block
        self.resume = resume

    def create_state(self, index, weight):
        mean = mxnet.ndarray.zeros(weight.shape, weight.context, dtype=weight.dtype)
        variance = mxnet.ndarray.zeros(weight.shape, weight.context, dtype=weight.dtype)

        if self.resume == True:
            cpu_weight = weight.copyto(mxnet.cpu())
            ones = mxnet.ndarray.ones_like(cpu_weight)
            zeros = mxnet.ndarray.zeros_like(cpu_weight)
            cpu_mask = mxnet.ndarray.where(cpu_weight == 0, zeros, ones)
            initial_sparsity = 1 - cpu_mask.sum().asnumpy()[0] / cpu_mask.size
            mask = cpu_mask.copyto(weight.context)
        else:
            initial_sparsity = 0
            mask = mxnet.ndarray.ones_like(weight)
        return (mean, variance, initial_sparsity, mask)

    def _get_names(self, indices):
        names = ['' for _ in indices]
        for i, index in enumerate(indices):
            if index in self.param_dict:
                names[i] = self.param_dict[index].name
            elif index in self.lr_mult:
                names[i] = self.lr_mult[index].name
            elif index in self.idx2name:
                names[i] = self.lr_mult.get(self.idx2name[index], 1.0).name
        return names
    def _get_name(self, index):
        return self._get_names([index])[0]

    def update(self, index, weight, grad, state):
        assert(isinstance(weight, mxnet.ndarray.NDArray))
        assert(isinstance(grad, mxnet.ndarray.NDArray))
        self._update_count(index)
        lr = self._get_lr(index)
        wd = self._get_wd(index)

        t = self._index_update_count[index]

        # preprocess grad
        grad = grad * self.rescale_grad + wd * weight
        if self.L1_regularization == True:
            grad = wd / 2 * mxnet.ndarray.sign(weight)
        if self.clip_gradient is not None:
            grad = mxnet.ndarray.clip(grad, -self.clip_gradient, self.clip_gradient)

        # warming momentum schedule
        momentum_t = self.beta1 * (1. - 0.5 * (pow(0.96, t * self.schedule_decay)))
        momentum_t_1 = self.beta1 * (1. - 0.5 * (pow(0.96, (t + 1) * self.schedule_decay)))
        self.m_schedule = self.m_schedule * momentum_t
        m_schedule_next = self.m_schedule * momentum_t_1

        # update m_t and v_t
        name = self._get_name(index)
        m_t, v_t, initial_sparsity, mask = state
        # grad *= mask
        m_t[:] *= self.beta1
        m_t[:] += (1. - self.beta1) * grad
        v_t[:] *= self.beta2
        v_t[:] += (1. - self.beta2) * grad * grad

        grad_prime = grad / (1. - self.m_schedule)
        m_t_prime = m_t / (1. - m_schedule_next)
        v_t_prime = v_t / (1. - pow(self.beta2, t))
        m_t_bar = (1. - momentum_t) * grad_prime + momentum_t_1 * m_t_prime

        # update weight
        weight[:] -= lr * m_t_bar / (mxnet.ndarray.sqrt(v_t_prime) + self.epsilon) * mask

        self._update_mask(t, name, initial_sparsity, weight, mask)

if __name__ == '__main__':
    import os
    import mxnet

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    num_gpus = mxnet.context.num_gpus()
    ctx = [mxnet.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mxnet.cpu()]

    dtype = 'float16'

    x = mxnet.ndarray.random.normal(shape=[2, 16, 8, 8], dtype=dtype, ctx=ctx[0])
    y = mxnet.ndarray.array([[0, 1, 1, 2], [2, 0, 0, -1]], ctx[0], dtype)

    Sequential = mxnet.gluon.nn.Sequential()
    Sequential.add(mxnet.gluon.nn.Conv2D(32, 3, weight_initializer=mxnet.initializer.Normal()))
    Sequential.add(mxnet.gluon.nn.Flatten())
    Sequential.add(mxnet.gluon.nn.Dense(4, weight_initializer=mxnet.initializer.Normal()))
    Sequential.cast(dtype)
    Sequential.initialize(ctx=ctx)

    # opt = mxnet.optimizer.Nadam(learning_rate=0.001, multi_precision=True, wd=0.01)
    # tr = mxnet.gluon.Trainer(Dense.params, opt)
    opt = NadamSparse(learning_rate=0.001, multi_precision=True, wd=0.01, target_sparsity=0.9, pretrain_step=5, sparse_step=10, frequency=3, keywords_no_sparse=[], sparse_block={'vector': [1], 'matrix': [1, 1], 'tensor': [4, -1, 1, -1]})
    tr = mxnet.gluon.Trainer(Sequential.collect_params(), opt)

    L1Loss = mxnet.gluon.loss.L1Loss()

    while True:
        with mxnet.autograd.record():
            predict = Sequential(x)
            loss = L1Loss(predict, y)
        loss.backward()

        tr.step(2)
        # print(Dense.weight.data(), Dense.bias.data())