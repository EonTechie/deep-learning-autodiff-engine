"""Core data structures."""
import autodiff
from typing import List, Optional, NamedTuple, Tuple, Union
from collections import namedtuple
import numpy

# autodiff version
LAZY_MODE = False
TENSOR_COUNTER = 0

# NOTE: we will import numpy as the array_api

import numpy as array_api
NDArray = numpy.ndarray


class Device:
    """Indicates the device supporting an NDArray."""


class CPUDevice(Device):
    """Represents data that sits in CPU"""

    def __repr__(self):
        return "autodiff.cpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True

def cpu():
    """Return cpu device"""
    return CPUDevice()

def all_devices():
    """return a list of all available devices"""
    return [cpu()]


class Op:
    """Operator definition."""

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function

        Returns
        -------
        output: nd.array
            Array output of the operation

        """
        raise NotImplementedError()

    def gradient(
        self, out_grad: "Value", node: "Value"
    ) -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") -> Tuple["Value"]:
        """ Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)


class TensorOp(Op):
    """ Op class specialized to output tensors, will be alternate subclasses for other structures """

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""

    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Optional[Op]
    inputs: List["Value"]
    # The following fields are cached fields for
    # dynamic computation
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        self.cached_data
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(
        self,
        op: Optional[Op],
        inputs: List["Tensor"],
        *,
        num_outputs: int = 1,
        cached_data: List[object] = None,
        requires_grad: Optional[bool] = None
    ):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value



class TensorTuple(Value):
    """Represent a tuple of tensors.

    To keep things simple, we do not support nested tuples.
    """

    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return autodiff.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "autodiff.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return autodiff.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tuple.make_const(self.realize_cached_data())


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # fall back, copy through numpy conversion
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        data = self.realize_cached_data()
        # numpy array always sits on cpu
        if array_api is numpy:
            return cpu()
        return data.device

    def backward(self, out_grad=None):
        out_grad = out_grad if out_grad else Tensor(numpy.ones(self.shape))
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "autodiff.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return autodiff.ops.EWiseAdd()(self, other)
        else:
            return autodiff.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return autodiff.ops.EWiseMul()(self, other)
        else:
            return autodiff.ops.MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            raise NotImplementedError()
        else:
            return autodiff.ops.PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return autodiff.ops.EWiseAdd()(self, autodiff.ops.Negate()(other))
        else:
            return autodiff.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return autodiff.ops.EWiseDiv()(self, other)
        else:
            return autodiff.ops.DivScalar(other)(self)

    def __matmul__(self, other):
        return autodiff.ops.MatMul()(self, other)

    def matmul(self, other):
        return autodiff.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return autodiff.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return autodiff.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return autodiff.ops.Reshape(shape)(self)

    def __neg__(self):
        return autodiff.ops.Negate()(self)

    def transpose(self, axes=None):
        return autodiff.ops.Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__


def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    ### BEGIN YOUR SOLUTION
    # BACKPROPAGATION ALGORİTMASI
    # 
    # Amaç: Çıktıdan başlayarak, tüm node'ların gradyanlarını hesapla
    # Yöntem: Reverse-mode automatic differentiation (backprop)
    #
    # Algoritma:
    # 1. Çıktıdan başla, geriye doğru git (reverse topological order)
    # 2. Her node için: gradyan katkılarını topla
    # 3. Bu node'un gradyanını kullanarak, input node'larının gradyanlarını hesapla
    # 4. Chain rule ile gradyanları geriye yay
    
    # Reverse topological order'da node'ları işle (çıktıdan girişe doğru)
    for node in reverse_topo_order:
        
        # Eğer bu node gradyan gerektirmiyorsa, atla
        if not node.requires_grad:
            continue
            
        # ADIM 1: Bu node için tüm gradyan katkılarını topla
        # (Bir node'a birden fazla yerden gradyan gelebilir)
        if node in node_to_output_grads_list:
            grad_list = node_to_output_grads_list[node]
            
            if len(grad_list) == 1:
                # Tek gradyan katkısı varsa, direkt ata
                node.grad = grad_list[0]
            else:
                # Birden fazla gradyan katkısı varsa, topla
                # Örnek: node = x, y = x + z, w = x * 2 ise
                # x'e hem y'den hem w'den gradyan gelir, bunları topla
                node.grad = sum_node_list(grad_list)
        
        # ADIM 2: Eğer bu node bir işlem sonucuysa, input'larına gradyan yay
        if node.op is not None:
            
            # Bu node'un gradyanını kullanarak, input'larının gradyanlarını hesapla
            # Her işlemin kendi gradient() fonksiyonu var (chain rule uygular)
            input_grads = node.op.gradient_as_tuple(node.grad, node)
            
            # ADIM 3: Hesaplanan gradyanları input node'lara yay
            for i, input_node in enumerate(node.inputs):
                if input_node.requires_grad:
                    
                    # Input node'un gradyan listesini hazırla
                    if input_node not in node_to_output_grads_list:
                        node_to_output_grads_list[input_node] = []
                    
                    # Bu input node'a gradyan katkısı ekle
                    # (Daha sonra toplanacak, çünkü başka yerlerden de gelebilir)
                    node_to_output_grads_list[input_node].append(input_grads[i])
    ### END YOUR SOLUTION


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    # Topological Sort: Computational graph'taki node'ları doğru sırada sıralar
    # 
    # Amaç: Her node'u, bağımlı olduğu node'lardan SONRA sıralar
    # Örnek: C = A + B ise, sıralama: [A, B, C] (A ve B önce, C sonra)
    #
    # Algoritma: Post-order DFS (Depth-First Search)
    # - Önce bağımlılıkları ziyaret et
    # - Sonra kendini listeye ekle
    
    visited = set()      # Ziyaret edilen node'ları takip et (tekrar ziyaret etme)
    topo_order = []      # Sonuç listesi (topological sıralama)
    
    # Her verilen node için DFS başlat
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    
    return topo_order    # Sıralanmış node listesini döndür
    ### END YOUR SOLUTION


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    # DFS (Depth-First Search) - Post-order traversal
    # 
    # Post-order: Önce çocukları ziyaret et, sonra kendini işle
    # Bu sayede bağımlılıklar önce, kendisi sonra listeye eklenir
    
    # Eğer bu node daha önce ziyaret edildiyse, tekrar işleme
    if node in visited:
        return
    
    # Bu node'u ziyaret edildi olarak işaretle
    visited.add(node)
    
    # ÖNCE: Tüm bağımlılıkları (input node'ları) ziyaret et
    # Örnek: C = A + B ise, C'yi işlemeden önce A ve B'yi işle
    for input_node in node.inputs:
        topo_sort_dfs(input_node, visited, topo_order)
    
    # SONRA: Tüm bağımlılıklar işlendikten sonra kendini listeye ekle
    # Bu sayede bağımlılıklar listede önce, bu node sonra gelir
    topo_order.append(node)
    ### END YOUR SOLUTION


##############################
####### Helper Methods #######
##############################


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce

    return reduce(add, node_list)
