// Expected failure: only schema_version = 1 : i64 is supported.
module attributes {ask.schema_version = 2 : i64} {
  func.func @main(%x: tensor<10xi32>) -> tensor<10xi32> {
    return %x : tensor<10xi32>
  }
}
