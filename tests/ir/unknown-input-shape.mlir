// Expected failure: input requires a concrete ranked i32 tensor.
// A static return declaration must not be used as proof of an input dimension.
module {
  func.func @main(%x: tensor<?xi32>) -> tensor<?xi32> {
    return %x : tensor<?xi32>
  }
}
