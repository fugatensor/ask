// A complete frontend fixture. Preparing this module never compiles its DSL.
module {
  "ask.semantics"() {
    sym_name = "scale2_i32_v1",
    function_type = (tensor<?xi32>) -> tensor<?xi32>,
    shape_rule = "same_as", shape_operand = 0 : i64, effects = "none"
  } : () -> ()
  "ask.implementation"() {
    sym_name = "triton_scale2", compiler = "triton", entry = "scale2",
    source = "import triton\0Aimport triton.language as tl\0A@triton.jit\0Adef scale2(t, u, N: tl.constexpr, BLOCK: tl.constexpr):\0A    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\0A    x = tl.load(t + i, mask=i < N, other=0)\0A    tl.store(u + i, x * 2, mask=i < N)\0A",
    signature = {t = "*i32", u = "*i32"},
    source_bindings = {t = "input:0", u = "output:0"},
    constants = {N = 10 : i64, BLOCK = 128 : i64},
    options = {num_warps = 4 : i64},
    target = {backend = "cuda", arch = 80 : i64, warp_size = 32 : i64},
    grid = array<i64: 1, 1, 1>
  } : () -> ()
  func.func @pipeline(%x: tensor<10xi32> {ask.value_id = "v_x"}) -> tensor<10xi32> {
    %one = "ask.constant"() {value = 1 : i32} : () -> i32
    %ones = "ask.splat"(%one) {shape = array<i64: 10>} : (i32) -> tensor<10xi32>
    %t = "ask.add"(%x, %ones) {ask.id = "P", ask.value_id = "v_t", overflow = "wrap"} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    %tm = "ask.mark"(%t) {mark_id = "m_F_in", value_id = "v_t", port_id = "F.in", role = "input"} : (tensor<10xi32>) -> tensor<10xi32>
    %u = "ask.external"(%tm) {ask.id = "F", ask.value_id = "v_u", semantics = @scale2_i32_v1, implementation = @triton_scale2} : (tensor<10xi32>) -> tensor<10xi32>
    %um = "ask.mark"(%u) {mark_id = "m_F_out", value_id = "v_u", port_id = "F.out", role = "output"} : (tensor<10xi32>) -> tensor<10xi32>
    %three = "ask.constant"() {value = 3 : i32} : () -> i32
    %threes = "ask.splat"(%three) {shape = array<i64: 10>} : (i32) -> tensor<10xi32>
    %y = "ask.add"(%um, %threes) {ask.id = "Q", ask.value_id = "v_y", overflow = "wrap"} : (tensor<10xi32>, tensor<10xi32>) -> tensor<10xi32>
    return %y : tensor<10xi32>
  }
}
