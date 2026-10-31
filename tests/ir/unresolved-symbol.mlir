// Expected failure: semantics must resolve to an ask.semantics symbol.
module {
  func.func @main(%x: tensor<10xi32>) -> tensor<10xi32> {
    %y = "ask.external"(%x) {semantics = @missing, implementation = @missing_impl} : (tensor<10xi32>) -> tensor<10xi32>
    return %y : tensor<10xi32>
  }
}
