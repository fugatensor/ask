// Expected failure: f64 constants are not yet implemented.
module {
  func.func @main() -> f64 {
    %x = "ask.constant"() {value = 1.0 : f64} : () -> f64
    return %x : f64
  }
}
