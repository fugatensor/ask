// Expected failure: requires equal input shapes; no implicit broadcasting.
module {
  func.func @main(%x: tensor<10xi32>) -> tensor<10xi32> {
    %one = "ask.constant"() {value = 1 : i32} : () -> i32
    %ones = "ask.splat"(%one) {shape = array<i64: 9>} : (i32) -> tensor<9xi32>
    %t = "ask.add"(%x, %ones) : (tensor<10xi32>, tensor<9xi32>) -> tensor<10xi32>
    return %t : tensor<10xi32>
  }
}
