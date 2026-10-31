// Expected failure: Ghost.in is not an existing connected data port.
module {
  func.func @main(%x: tensor<10xi32> {ask.value_id = "v_x"}) -> tensor<10xi32> {
    %m = "ask.mark"(%x) {mark_id = "bad", value_id = "v_x", role = "input", port_id = "Ghost.in"} : (tensor<10xi32>) -> tensor<10xi32>
    return %m : tensor<10xi32>
  }
}
