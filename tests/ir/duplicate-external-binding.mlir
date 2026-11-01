// Two source parameters cannot both stand in for input:0 while output:0 is absent.
module {
  "ask.semantics"() {sym_name = "copy", function_type = (tensor<?xi32>) -> tensor<?xi32>, shape_rule = "same_as", effects = "none"} : () -> ()
  "ask.implementation"() {
    sym_name = "copy_impl", compiler = "triton", entry = "copy",
    source = "not compiled by prepare",
    signature = {x = "*i32", y = "*i32"},
    source_bindings = {x = "input:0", y = "input:0"},
    constants = {}, options = {}, target = {backend = "cuda"},
    grid = array<i64: 1, 1, 1>
  } : () -> ()
  func.func @bad(%x: tensor<10xi32>) -> tensor<10xi32> {
    %y = "ask.external"(%x) {semantics = @copy, implementation = @copy_impl} : (tensor<10xi32>) -> tensor<10xi32>
    return %y : tensor<10xi32>
  }
}
