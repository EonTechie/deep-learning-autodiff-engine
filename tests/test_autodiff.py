import sys
sys.path.append('./src')
import numdifftools as nd

import numpy as np
import autodiff as auto


##############################################################################
### TESTS/SUBMISSION CODE FOR forward passes
def test_divide_forward():
    np.testing.assert_allclose(auto.divide(auto.Tensor([[3.3 , 4.35, 1.2 ],
       [2.45, 0.95, 2.55]]), auto.Tensor([[4.6 , 4.35, 4.8 ],
       [0.65, 0.7 , 4.4 ]])).numpy(), np.array([[0.717391304348, 1.            , 0.25          ],
       [3.769230769231, 1.357142857143, 0.579545454545]]))


def test_divide_scalar_forward():
    np.testing.assert_allclose(auto.divide_scalar(auto.Tensor([[1.7 , 1.45]]), scalar=12).numpy(), 
        np.array([[0.141666666667, 0.120833333333]]))


def test_matmul_forward():
    np.testing.assert_allclose(auto.matmul(auto.Tensor([[4.95, 1.75, 0.25],
       [4.15, 4.25, 0.3 ],
       [0.3 , 0.4 , 2.1 ]]), auto.Tensor([[1.35, 2.2 , 1.55],
       [3.85, 4.8 , 2.6 ],
       [1.15, 0.85, 4.15]])).numpy(), np.array([[13.7075, 19.5025, 13.26  ],
       [22.31  , 29.785 , 18.7275],
       [ 4.36  ,  4.365 , 10.22  ]]))
    np.testing.assert_allclose(auto.matmul(auto.Tensor([[3.8 , 0.05],
       [2.3 , 3.35],
       [1.6 , 2.6 ]]), auto.Tensor([[1.1 , 3.5 , 3.7 ],
       [0.05, 1.25, 1.  ]])).numpy(), np.array([[ 4.1825, 13.3625, 14.11  ],
       [ 2.6975, 12.2375, 11.86  ],
       [ 1.89  ,  8.85  ,  8.52  ]]))
    np.testing.assert_allclose(auto.matmul(auto.Tensor([[[4.  , 2.15],
        [1.25, 1.35],
        [0.75, 1.6 ]],
       [[2.9 , 2.15],
        [3.3 , 4.1 ],
        [2.5 , 0.25]],
       [[2.9 , 4.35],
        [1.2 , 3.5 ],
        [3.55, 3.95]],
       [[2.55, 4.35],
        [4.25, 0.2 ],
        [3.95, 3.4 ]],
       [[2.2 , 2.05],
        [0.95, 1.8 ],
        [2.7 , 2.  ]],
       [[0.45, 1.1 ],
        [3.15, 0.7 ],
        [2.9 , 1.95]]]), auto.Tensor([[[2.7 , 4.05, 0.1 ],
        [1.75, 3.05, 2.3 ]],
       [[0.55, 4.1 , 2.3 ],
        [4.45, 2.35, 2.55]],
       [[1.2 , 3.95, 4.6 ],
        [4.2 , 3.5 , 3.35]],
       [[2.55, 4.4 , 2.05],
        [2.4 , 0.6 , 4.65]],
       [[2.95, 0.8 , 0.6 ],
        [0.45, 1.3 , 0.75]],
       [[1.25, 2.1 , 0.4 ],
        [0.85, 3.5 , 3.7 ]]])).numpy(), np.array([[[14.5625, 22.7575,  5.345 ],
        [ 5.7375,  9.18  ,  3.23  ],
        [ 4.825 ,  7.9175,  3.755 ]],
       [[11.1625, 16.9425, 12.1525],
        [20.06  , 23.165 , 18.045 ],
        [ 2.4875, 10.8375,  6.3875]],
       [[21.75  , 26.68  , 27.9125],
        [16.14  , 16.99  , 17.245 ],
        [20.85  , 27.8475, 29.5625]],
       [[16.9425, 13.83  , 25.455 ],
        [11.3175, 18.82  ,  9.6425],
        [18.2325, 19.42  , 23.9075]],
       [[ 7.4125,  4.425 ,  2.8575],
        [ 3.6125,  3.1   ,  1.92  ],
        [ 8.865 ,  4.76  ,  3.12  ]],
       [[ 1.4975,  4.795 ,  4.25  ],
        [ 4.5325,  9.065 ,  3.85  ],
        [ 5.2825, 12.915 ,  8.375 ]]]))
    np.testing.assert_allclose(auto.matmul(auto.Tensor([[1.9 , 1.9 ],
       [4.8 , 4.9 ],
       [3.25, 3.75]]), auto.Tensor([[[1.25, 1.8 , 1.95],
        [3.75, 2.85, 2.25]],
       [[1.75, 2.7 , 3.3 ],
        [2.95, 1.55, 3.85]],
       [[4.2 , 3.05, 3.35],
        [3.3 , 4.75, 2.1 ]]])).numpy(), np.array([[[ 9.5   ,  8.835 ,  7.98  ],
        [24.375 , 22.605 , 20.385 ],
        [18.125 , 16.5375, 14.775 ]],
       [[ 8.93  ,  8.075 , 13.585 ],
        [22.855 , 20.555 , 34.705 ],
        [16.75  , 14.5875, 25.1625]],
       [[14.25  , 14.82  , 10.355 ],
        [36.33  , 37.915 , 26.37  ],
        [26.025 , 27.725 , 18.7625]]]))
    np.testing.assert_allclose(auto.matmul(auto.Tensor([[[3.4 , 2.95],
        [0.25, 1.95],
        [4.4 , 4.4 ]],
       [[0.55, 1.1 ],
        [0.75, 1.55],
        [4.1 , 1.2 ]],
       [[1.5 , 4.05],
        [1.5 , 1.55],
        [2.3 , 1.25]]]), auto.Tensor([[2.2 , 0.65, 2.5 ],
       [2.5 , 1.3 , 0.15]])).numpy(), np.array([[[14.855 ,  6.045 ,  8.9425],
        [ 5.425 ,  2.6975,  0.9175],
        [20.68  ,  8.58  , 11.66  ]],
       [[ 3.96  ,  1.7875,  1.54  ],
        [ 5.525 ,  2.5025,  2.1075],
        [12.02  ,  4.225 , 10.43  ]],
       [[13.425 ,  6.24  ,  4.3575],
        [ 7.175 ,  2.99  ,  3.9825],
        [ 8.185 ,  3.12  ,  5.9375]]]))


def test_summation_forward():
    np.testing.assert_allclose(auto.summation(auto.Tensor([[2.2 , 4.35, 1.4 , 0.3 , 2.65],
       [1.  , 0.85, 2.75, 3.8 , 1.55],
       [3.2 , 2.3 , 3.45, 0.7 , 0.  ]])).numpy(), np.array(30.5))
    np.testing.assert_allclose(auto.summation(auto.Tensor([[1.05, 2.55, 1.  ],
       [2.95, 3.7 , 2.6 ],
       [0.1 , 4.1 , 3.3 ],
       [1.1 , 3.4 , 3.4 ],
       [1.8 , 4.55, 2.3 ]]), axes=1).numpy(), np.array([4.6 , 9.25, 7.5 , 7.9 , 8.65]))
    np.testing.assert_allclose(auto.summation(auto.Tensor([[1.5 , 3.85, 3.45],
       [1.35, 1.3 , 0.65],
       [2.6 , 4.55, 0.25]]), axes=0).numpy(), np.array([5.45, 9.7 , 4.35]))


def test_broadcast_to_forward():
    np.testing.assert_allclose(auto.broadcast_to(auto.Tensor([[1.85, 0.85, 0.6 ]]), shape=(3, 3, 3)).numpy(), np.array([[[1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ]],
       [[1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ]],
       [[1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ]]]))


def test_reshape_forward():
    np.testing.assert_allclose(auto.reshape(auto.Tensor([[2.9 , 2.  , 2.4 ],
       [3.95, 3.95, 4.65],
       [2.1 , 2.5 , 2.7 ],
       [1.9 , 4.85, 3.25],
       [3.35, 3.45, 3.45]]), shape=(15,)).numpy(), np.array([2.9 , 2.  , 2.4 , 3.95, 3.95, 4.65, 2.1 , 2.5 , 2.7 , 1.9 , 4.85,
       3.25, 3.35, 3.45, 3.45]))
    np.testing.assert_allclose(auto.reshape(auto.Tensor([[[4.1 , 4.05, 1.35, 1.65],
        [3.65, 0.9 , 0.65, 4.15]],
       [[4.7 , 1.4 , 2.55, 4.8 ],
        [2.8 , 1.75, 2.8 , 0.6 ]],
       [[3.75, 0.6 , 0.  , 3.5 ],
        [0.15, 1.9 , 4.75, 2.8 ]]]), shape=(2, 3, 4)).numpy(), np.array([[[4.1 , 4.05, 1.35, 1.65],
        [3.65, 0.9 , 0.65, 4.15],
        [4.7 , 1.4 , 2.55, 4.8 ]],
       [[2.8 , 1.75, 2.8 , 0.6 ],
        [3.75, 0.6 , 0.  , 3.5 ],
        [0.15, 1.9 , 4.75, 2.8 ]]]))

def test_negate_forward():
    np.testing.assert_allclose(auto.negate(auto.Tensor([[1.45, 0.55]])).numpy(), np.array([[-1.45, -0.55]]))


def test_transpose_forward():
    np.testing.assert_allclose(auto.transpose(auto.Tensor([[[1.95]],
       [[2.7 ]],
       [[3.75]]]), axes=(1, 2)).numpy(), np.array([[[1.95]],
       [[2.7 ]],
       [[3.75]]]))
    np.testing.assert_allclose(auto.transpose(auto.Tensor([[[[0.95]]],
       [[[2.55]]],
       [[[0.45]]]]), axes=(2, 3)).numpy(), np.array([[[[0.95]]],
       [[[2.55]]],
       [[[0.45]]]]))
    np.testing.assert_allclose(auto.transpose(auto.Tensor([[[[0.4 , 0.05],
         [2.95, 1.3 ]],
        [[4.8 , 1.2 ],
         [1.65, 3.1 ]]],
       [[[1.45, 3.05],
         [2.25, 0.1 ]],
        [[0.45, 4.75],
         [1.5 , 1.8 ]]],
       [[[1.5 , 4.65],
         [1.35, 2.7 ]],
        [[2.  , 1.65],
         [2.05, 1.2 ]]]])).numpy(), np.array([[[[0.4 , 2.95],
         [0.05, 1.3 ]],
        [[4.8 , 1.65],
         [1.2 , 3.1 ]]],
       [[[1.45, 2.25],
         [3.05, 0.1 ]],
        [[0.45, 1.5 ],
         [4.75, 1.8 ]]],
       [[[1.5 , 1.35],
         [4.65, 2.7 ]],
        [[2.  , 2.05],
         [1.65, 1.2 ]]]]))
    np.testing.assert_allclose(auto.transpose(auto.Tensor([[[2.45]],
       [[3.5 ]],
       [[0.9 ]]]), axes=(0, 1)).numpy(), np.array([[[2.45],
        [3.5 ],
        [0.9 ]]]))
    np.testing.assert_allclose(auto.transpose(auto.Tensor([[4.4 , 2.05],
       [1.85, 2.25],
       [0.15, 1.4 ]])).numpy(), np.array([[4.4 , 1.85, 0.15],
       [2.05, 2.25, 1.4 ]]))
    np.testing.assert_allclose(auto.transpose(auto.Tensor([[0.05, 3.7 , 1.35],
       [4.45, 3.25, 1.95],
       [2.45, 4.4 , 4.5 ]])).numpy(), np.array([[0.05, 4.45, 2.45],
       [3.7 , 3.25, 4.4 ],
       [1.35, 1.95, 4.5 ]]))
    np.testing.assert_allclose(auto.transpose(auto.Tensor([[[0.55, 1.8 , 0.2 ],
        [0.8 , 2.75, 3.7 ],
        [0.95, 1.4 , 0.8 ]],
       [[0.75, 1.6 , 1.35],
        [3.75, 4.  , 4.55],
        [1.85, 2.5 , 4.8 ]],
       [[0.2 , 3.35, 3.4 ],
        [0.3 , 4.85, 4.85],
        [4.35, 4.25, 3.05]]]), axes=(0, 1)).numpy(), np.array([[[0.55, 1.8 , 0.2 ],
        [0.75, 1.6 , 1.35],
        [0.2 , 3.35, 3.4 ]],
       [[0.8 , 2.75, 3.7 ],
        [3.75, 4.  , 4.55],
        [0.3 , 4.85, 4.85]],
       [[0.95, 1.4 , 0.8 ],
        [1.85, 2.5 , 4.8 ],
        [4.35, 4.25, 3.05]]]))


### TESTS/SUBMISSION CODE FOR backward passes

def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [x.numpy() for x in out.op.gradient_as_tuple(auto.Tensor(np.ones(out.shape)), out)]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i])
        for i in range(len(args))
    )
    assert error < tol
    return computed_grads


def test_divide_backward():
    gradient_check(auto.divide, auto.Tensor(np.random.randn(5, 4)), auto.Tensor(5 + np.random.randn(5, 4)))


def test_divide_scalar_backward():
    gradient_check(auto.divide_scalar, auto.Tensor(np.random.randn(5, 4)), scalar=np.random.randn(1))


def test_matmul_simple_backward():
    gradient_check(auto.matmul, auto.Tensor(np.random.randn(5, 4)), auto.Tensor(np.random.randn(4, 5)))


def test_matmul_batched_backward():
    gradient_check(auto.matmul, auto.Tensor(np.random.randn(6, 6, 5, 4)), auto.Tensor(np.random.randn(6, 6, 4, 3)))
    gradient_check(auto.matmul, auto.Tensor(np.random.randn(6, 6, 5, 4)), auto.Tensor(np.random.randn(4, 3)))
    gradient_check(auto.matmul, auto.Tensor(np.random.randn(5, 4)), auto.Tensor(np.random.randn(6, 6, 4, 3)))


def test_reshape_backward():
    gradient_check(auto.reshape, auto.Tensor(np.random.randn(5, 4)), shape=(4, 5))


def test_negate_backward():
    gradient_check(auto.negate, auto.Tensor(np.random.randn(5, 4)))


def test_transpose_backward():
    gradient_check(auto.transpose, auto.Tensor(np.random.randn(3, 5, 4)), axes=(1, 2))
    gradient_check(auto.transpose, auto.Tensor(np.random.randn(3, 5, 4)), axes=(0, 1))

def test_broadcast_to_backward():
    gradient_check(auto.broadcast_to, auto.Tensor(np.random.randn(3, 1)), shape=(3, 3))
    gradient_check(auto.broadcast_to, auto.Tensor(np.random.randn(1, 3)), shape=(3, 3))
    gradient_check(auto.broadcast_to, auto.Tensor(np.random.randn(1,)), shape=(3, 3, 3))
    gradient_check(auto.broadcast_to, auto.Tensor(np.random.randn()), shape=(3, 3, 3))
    gradient_check(auto.broadcast_to, auto.Tensor(np.random.randn(5,4,1)), shape=(5,4,3))


def test_summation_backward():
    gradient_check(auto.summation, auto.Tensor(np.random.randn(5,4)), axes=(1,))
    gradient_check(auto.summation, auto.Tensor(np.random.randn(5,4)), axes=(0,))
    gradient_check(auto.summation, auto.Tensor(np.random.randn(5,4)), axes=(0,1))
    gradient_check(auto.summation, auto.Tensor(np.random.randn(5,4,1)), axes=(0,1))




def test_topo_sort():
    # Test case 1
    a1, b1 = auto.Tensor(np.asarray([[0.88282157]])), auto.Tensor(np.asarray([[0.90170084]]))
    c1 = 3*a1*a1 + 4*b1*a1 - a1

    soln = np.array([np.array([[0.88282157]]),
                     np.array([[2.64846471]]),
                     np.array([[2.33812177]]),
                     np.array([[0.90170084]]),
                     np.array([[3.60680336]]),
                     np.array([[3.1841638]]),
                     np.array([[5.52228558]]),
                     np.array([[-0.88282157]]),
                     np.array([[4.63946401]])])

    topo_order = np.array([x.numpy() for x in auto.autograd.find_topo_sort([c1])])

    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)

    # Test case 2
    a1, b1 = auto.Tensor(np.asarray([[0.20914675], [0.65264178]])), auto.Tensor(np.asarray([[0.65394286, 0.08218317]]))
    c1 = 3 * ((b1 @ a1) + (2.3412 * b1) @ a1) + 1.5

    soln = [np.array([[0.65394286, 0.08218317]]),
            np.array([[0.20914675], [0.65264178]]),
            np.array([[0.19040619]]),
            np.array([[1.53101102, 0.19240724]]),
            np.array([[0.44577898]]), np.array([[0.63618518]]),
            np.array([[1.90855553]]), np.array([[3.40855553]])]

    topo_order = [x.numpy() for x in auto.autograd.find_topo_sort([c1])]

    assert len(soln) == len(topo_order)
    # step through list as entries differ in length
    for t, s in zip(topo_order, soln):
        np.testing.assert_allclose(t, s, rtol=1e-06, atol=1e-06)

    # Test case 3
    a = auto.Tensor(np.asarray([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]))
    b = auto.Tensor(np.asarray([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]))
    e = (a@b + b - a)@a

    topo_order = np.array([x.numpy() for x in auto.autograd.find_topo_sort([e])])

    soln = np.array([np.array([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]),
                     np.array([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]),
                     np.array([[1.6252339, -1.38248184], [1.25355725, -0.03148146]]),
                     np.array([[2.97095081, -2.33832617], [0.25927152, -0.07165645]]),
                     np.array([[-1.4335016, -0.30559972], [-0.08130171, 1.15072371]]),
                     np.array([[1.53744921, -2.64392589], [0.17796981, 1.07906726]]),
                     np.array([[1.98898021, 3.51227226], [0.34285002, -1.18732075]])])

    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)

### TESTS/SUBMISSION CODE FOR compute_gradient_of_variables

def test_compute_gradient():
    gradient_check(lambda A,B,C : auto.summation((A@B+C)*(A@B), axes=None),
                   auto.Tensor(np.random.randn(10,9)),
                   auto.Tensor(np.random.randn(9,8)),
                   auto.Tensor(np.random.randn(10,8)), backward=True)
    gradient_check(lambda A,B : auto.summation(auto.broadcast_to(A,shape=(10,9))*B, axes=None),
                   auto.Tensor(np.random.randn(10,1)),
                   auto.Tensor(np.random.randn(10,9)), backward=True)
    gradient_check(lambda A,B,C : auto.summation(auto.reshape(A,shape=(10,10))@B/5+C, axes=None),
                   auto.Tensor(np.random.randn(100)),
                   auto.Tensor(np.random.randn(10,5)),
                   auto.Tensor(np.random.randn(10,5)), backward=True)

    # check gradient of gradient
    x2 = auto.Tensor([6])
    x3 = auto.Tensor([0])
    y = x2 * x2 + x2 * x3
    y.backward()
    grad_x2 = x2.grad
    grad_x3 = x3.grad
    # gradient of gradient
    grad_x2.backward()
    grad_x2_x2 = x2.grad
    grad_x2_x3 = x3.grad
    x2_val = x2.numpy()
    x3_val = x3.numpy()
    assert y.numpy() == x2_val * x2_val + x2_val * x3_val
    assert grad_x2.numpy() == 2 * x2_val + x3_val
    assert grad_x3.numpy() == x2_val
    assert grad_x2_x2.numpy() == 2
    assert grad_x2_x3.numpy() == 1


