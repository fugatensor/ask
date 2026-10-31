// Expected failure: only scalar i32 constants are implemented.
module {
  func.func @main() -> f32 {
    %x = "ask.constant"() {value = 1.0 : f32} : () -> f32
    return %x : f32
  }
}
