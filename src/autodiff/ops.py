"""Operator implementations."""

import functools
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api

import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # PowerScalar: f(x) = x^n işleminin türevi
        # Türev kuralı: d/dx (x^n) = n * x^(n-1)
        
        input_tensor = node.inputs[0]  # Giriş tensörünü al (x)
        
        # Chain rule: out_grad * (local gradient)
        # Local gradient = n * x^(n-1)
        # Örnek: x^3'ün türevi = 3 * x^2
        return out_grad * self.scalar * power_scalar(input_tensor, self.scalar - 1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # EWiseDiv: f(x,y) = x/y işleminin türevi
        # İki girişi var: x (lhs) ve y (rhs)
        
        lhs, rhs = node.inputs  # x ve y'yi al
        
        # Bölme işleminin türev kuralları:
        # ∂/∂x (x/y) = 1/y        (x'e göre türev)
        # ∂/∂y (x/y) = -x/y²      (y'ye göre türev)
        
        # Chain rule uygula: out_grad * local_gradient
        grad_lhs = out_grad / rhs                    # x için: out_grad * (1/y)
        grad_rhs = out_grad * (-lhs) / (rhs * rhs)   # y için: out_grad * (-x/y²)
        
        return grad_lhs, grad_rhs  # Her iki giriş için gradyanları döndür
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # DivScalar: f(x) = x/c işleminin türevi (c sabit sayı)
        # Türev kuralı: d/dx (x/c) = 1/c
        
        # Chain rule: out_grad * local_gradient
        # Local gradient = 1/c (sabit ile bölmenin türevi)
        # Örnek: x/5'in türevi = 1/5
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            # Default transpose: swap last two dimensions
            if a.ndim >= 2:
                axes = list(range(a.ndim))
                axes[-2], axes[-1] = axes[-1], axes[-2]
                return array_api.transpose(a, axes)
            else:
                return array_api.transpose(a)
        else:
            # Handle specific axes
            if len(self.axes) != a.ndim:
                # If axes length doesn't match ndim, create full permutation
                axes = list(range(a.ndim))
                # Only swap the specified axes if they're valid
                if len(self.axes) == 2 and all(0 <= ax < a.ndim for ax in self.axes):
                    axes[self.axes[0]], axes[self.axes[1]] = axes[self.axes[1]], axes[self.axes[0]]
                return array_api.transpose(a, axes)
            else:
                # Full permutation provided
                return array_api.transpose(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Transpose: Matrisin satır-sütunlarını değiştirme işlemi
        # Transpose işleminin türevi = aynı transpose işlemini tekrar uygula
        # 
        # Yola çıktığım nokta: Eğer A'yı transpose ettiysen, gradyanı da transpose et
        # Çünkü: (A^T)^T = A (transpose'un tersi kendisidir)
        #
        # Örnek: A = [[1,2], [3,4]] → A^T = [[1,3], [2,4]]
        #        Gradyan da aynı şekilde transpose edilir
        
        return transpose(out_grad, self.axes)  # Aynı axes ile transpose uygula
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Reshape: Tensörün şeklini değiştirme işlemi (elemanlar aynı kalır)
        # Reshape işleminin türevi = gradyanı orijinal şekle geri döndür
        #
        # Yola çıktığım nokta: Reshape sadece şekil değiştirir, değerler aynı kalır
        # Bu yüzden gradyan da aynı değerlere sahip olmalı, sadece şekli farklı
        #
        # Örnek: [1,2,3,4,5,6] → reshape(2,3) → [[1,2,3], [4,5,6]]
        #        Gradyan: [[a,b,c], [d,e,f]] → reshape(6,) → [a,b,c,d,e,f]
        
        input_shape = node.inputs[0].shape  # Orijinal girişin şeklini al
        return reshape(out_grad, input_shape)  # Gradyanı orijinal şekle döndür
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # BroadcastTo: Küçük tensörü büyük şekle "yayma" işlemi
        # Broadcast işleminin türevi = yayılan boyutları tekrar topla
        #
        # Yola çıktığım nokta: Broadcast bir değeri kopyalar, türevde bu kopyaları toplarız
        # Örnek: [1,2] → broadcast_to(3,2) → [[1,2], [1,2], [1,2]]
        #        Gradyan: [[a,b], [c,d], [e,f]] → sum → [a+c+e, b+d+f]
        
        input_shape = node.inputs[0].shape  # Orijinal girişin şekli
        
        # 1. Eklenen boyutları topla (baştan eklenen boyutlar)
        ndims_added = len(self.shape) - len(input_shape)
        for i in range(ndims_added):
            # İlk boyutu sürekli topla (çünkü bu boyutlar eklenmişti)
            out_grad = summation(out_grad, axes=(0,))
        
        # 2. Boyutu 1 olan ama broadcast edilen boyutları topla
        for i, (input_dim, output_dim) in enumerate(zip(input_shape, self.shape[ndims_added:])):
            if input_dim == 1 and output_dim > 1:
                # Bu boyut 1'den büyük boyuta broadcast edilmişti, topla
                out_grad = summation(out_grad, axes=(i,))
                # Boyutu 1 olarak geri ekle (orijinal şekle uygun olması için)
                out_grad = reshape(out_grad, out_grad.shape[:i] + (1,) + out_grad.shape[i:])
        
        return out_grad
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Summation: Belirli boyutlar boyunca toplama işlemi
        # Sum işleminin türevi = gradyanı orijinal şekle geri yay (broadcast)
        #
        # Yola çıktığım nokta: Sum işlemi boyutları "sıkıştırır", türevde bu boyutları geri açarız
        # Örnek: [[1,2,3], [4,5,6]] → sum(axis=0) → [5,7,9]
        #        Gradyan: [a,b,c] → broadcast → [[a,b,c], [a,b,c]]
        
        input_shape = node.inputs[0].shape  # Orijinal girişin şekli
        
        if self.axes is None:
            # Tüm boyutlar boyunca toplama (sonuç skaler)
            # Gradyanı (skaler) orijinal şekle broadcast et
            return broadcast_to(out_grad, input_shape)
        else:
            # Belirli boyutlar boyunca toplama
            # Önce toplanan boyutları geri ekle (boyut 1 ile)
            
            grad_shape = list(out_grad.shape)  # Mevcut gradyan şekli
            
            # axes'i tuple'a çevir (tek sayı da olabilir)
            if isinstance(self.axes, int):
                axes = (self.axes,)
            else:
                axes = self.axes
            
            # Toplanan her boyutu geri ekle (boyut 1 olarak)
            for axis in sorted(axes):
                grad_shape.insert(axis, 1)  # axis pozisyonuna boyut 1 ekle
            
            # Gradyanı yeni şekle getir
            out_grad = reshape(out_grad, grad_shape)
            
            # Sonra orijinal şekle broadcast et
            return broadcast_to(out_grad, input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # MatMul: Matris çarpımı A @ B işleminin türevi
        # Matris çarpımının türev kuralları:
        # ∂/∂A (A @ B) = out_grad @ B^T    (A'ya göre türev)
        # ∂/∂B (A @ B) = A^T @ out_grad    (B'ye göre türev)
        #
        # Yola çıktığım nokta: Chain rule + matris çarpımının türev kuralları
        # Örnek: C = A @ B ise, dC/dA = dC @ B^T, dC/dB = A^T @ dC
        
        lhs, rhs = node.inputs  # A ve B matrislerini al
        
        # Türev kurallarını uygula:
        lhs_grad = matmul(out_grad, transpose(rhs))    # A için: out_grad @ B^T
        rhs_grad = matmul(transpose(lhs), out_grad)    # B için: A^T @ out_grad
        
        # Batched (çok boyutlu) matris çarpımı için broadcasting kontrolü
        # Eğer orijinal matris daha küçük boyutluysa, ekstra boyutları topla
        
        # Sol matris (A) için broadcast kontrolü
        if len(lhs.shape) < len(lhs_grad.shape):
            # Ekstra boyutlar eklenmişti, bunları topla
            axes_to_sum = tuple(range(len(lhs_grad.shape) - len(lhs.shape)))
            lhs_grad = summation(lhs_grad, axes=axes_to_sum)
        
        # Sağ matris (B) için broadcast kontrolü  
        if len(rhs.shape) < len(rhs_grad.shape):
            # Ekstra boyutlar eklenmişti, bunları topla
            axes_to_sum = tuple(range(len(rhs_grad.shape) - len(rhs.shape)))
            rhs_grad = summation(rhs_grad, axes=axes_to_sum)
        
        return lhs_grad, rhs_grad  # Her iki matris için gradyanları döndür
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Negate: f(x) = -x işleminin türevi
        # Türev kuralı: d/dx (-x) = -1
        #
        # Yola çıktığım nokta: Negatif işareti türevde de kalır
        # Chain rule: out_grad * (-1) = -out_grad
        # Örnek: y = -x ise, dy/dx = -1
        
        return negate(out_grad)  # Gradyanın işaretini değiştir
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Log: f(x) = ln(x) işlemi (doğal logaritma)
        # Not: x > 0 olmalı (log negatif sayılardan tanımsız)
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Log: f(x) = ln(x) işleminin türevi
        # Türev kuralı: d/dx (ln(x)) = 1/x
        #
        # Yola çıktığım nokta: Logaritmanın temel türevi
        # Chain rule: out_grad * (1/x)
        # Örnek: y = ln(x) ise, dy/dx = 1/x
        
        input_tensor = node.inputs[0]  # Giriş tensörünü al (x)
        return out_grad / input_tensor  # out_grad * (1/x)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Exp: f(x) = e^x işlemi (exponential fonksiyon)
        # e ≈ 2.71828 (Euler sayısı)
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Exp: f(x) = e^x işleminin türevi
        # Türev kuralı: d/dx (e^x) = e^x
        #
        # Yola çıktığım nokta: Exponential fonksiyonun özel özelliği
        # Türevi kendisine eşittir!
        # Chain rule: out_grad * e^x
        # Örnek: y = e^x ise, dy/dx = e^x
        
        input_tensor = node.inputs[0]  # Giriş tensörünü al (x)
        return out_grad * exp(input_tensor)  # out_grad * e^x
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # ReLU: f(x) = max(0, x) işlemi (Rectified Linear Unit)
        # x > 0 ise x, x ≤ 0 ise 0 döndürür
        # Neural network'lerde çok kullanılan aktivasyon fonksiyonu
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # ReLU: f(x) = max(0, x) işleminin türevi
        # Türev kuralı: 
        # d/dx (ReLU(x)) = 1 if x > 0
        #                = 0 if x ≤ 0
        #
        # Yola çıktığım nokta: ReLU parçalı fonksiyondur
        # Pozitif bölgede eğim 1, negatif bölgede eğim 0
        # Chain rule: out_grad * (1 veya 0)
        
        input_tensor = node.inputs[0]  # Giriş tensörünü al (x)
        
        # x > 0 olan yerlerde 1, x ≤ 0 olan yerlerde 0 olan mask oluştur
        relu_mask = Tensor(input_tensor.realize_cached_data() > 0, 
                          requires_grad=False)
        
        return out_grad * relu_mask  # out_grad * mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
