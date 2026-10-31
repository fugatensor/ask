// Expected failure: marker value_id does not match operand identity v_x.
module {
  func.func @main(%x: tensor<10xi32> {ask.value_id = "v_x"}) -> tensor<10xi32> {
    %m = "ask.mark"(%x) {mark_id = "bad", value_id = "unbound", role = "value"} : (tensor<10xi32>) -> tensor<10xi32>
    return %m : tensor<10xi32>
  }
}
