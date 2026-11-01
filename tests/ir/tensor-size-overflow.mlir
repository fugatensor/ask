// Shape inference must reject byte-count overflow before a backend sees it.
module {
  func.func @bad(%x: tensor<2305843009213693952xi32>) -> tensor<2305843009213693952xi32> {
    return %x : tensor<2305843009213693952xi32>
  }
}
