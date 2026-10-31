// Expected: graph return and all tensor results refine to tensor<10xi32>;
// the marker is reported once, removed, and shares the logical value P.result0.
module {
  func.func @main(%x: tensor<10xi32>) -> tensor<?xi32> {
    %one = "ask.constant"() {value = 1 : i32} : () -> i32
    %ones = "ask.splat"(%one) {shape = array<i64: 10>} : (i32) -> tensor<?xi32>
    %t = "ask.add"(%x, %ones) {ask.id = "P", overflow = "wrap"} : (tensor<10xi32>, tensor<?xi32>) -> tensor<?xi32>
    %m = "ask.mark"(%t) {mark_id = "observe_P", value_id = "P.result0", role = "value"} : (tensor<?xi32>) -> tensor<?xi32>
    return %m : tensor<?xi32>
  }
}
