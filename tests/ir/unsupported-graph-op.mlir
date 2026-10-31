// Expected failure: calls between graph functions are not in the prototype.
module {
  func.func @other(%x: tensor<10xi32>) -> tensor<10xi32> {
    return %x : tensor<10xi32>
  }
  func.func @main(%x: tensor<10xi32>) -> tensor<10xi32> {
    %y = func.call @other(%x) : (tensor<10xi32>) -> tensor<10xi32>
    return %y : tensor<10xi32>
  }
}
