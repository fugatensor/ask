// Expected failure: a malformed revision is not silently interpreted as zero.
module attributes {ask.ir_revision = "old"} {
  func.func @main(%x: tensor<10xi32>) -> tensor<10xi32> {
    return %x : tensor<10xi32>
  }
}
