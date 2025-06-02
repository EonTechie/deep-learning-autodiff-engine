# my_transpose_test.py dosyası oluştur:
import sys
sys.path.append('./src')
import autodiff as auto
import numpy as np

def test_basic_transpose():
    print("=== Basic Transpose Test ===")
    
    # Basit 2D test:
    x = auto.Tensor([[1, 2, 3], [4, 5, 6]])
    result = auto.transpose(x)
    
    print("Input:", x.numpy())
    print("Output:", result.numpy())
    print("Expected: [[1, 4], [2, 5], [3, 6]]")
    
    expected = np.array([[1, 4], [2, 5], [3, 6]])
    if np.allclose(result.numpy(), expected):
        print("✅ TEST PASSED!")
    else:
        print("❌ TEST FAILED!")

def test_3d_transpose():
    print("\n=== 3D Transpose Test ===")
    
    # 3D test:
    x = auto.Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    result = auto.transpose(x, axes=(1, 0, 2))
    
    print("Input shape:", x.shape)
    print("Output shape:", result.shape)
    print("Input:\n", x.numpy())
    print("Output:\n", result.numpy())

if __name__ == "__main__":
    test_basic_transpose()
    test_3d_transpose()