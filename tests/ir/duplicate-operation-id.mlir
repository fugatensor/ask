// Expected failure: operation identities are module-wide in this prototype.
module {
  func.func @first() -> i32 {
    %x = "ask.constant"() {ask.id = "same", value = 1 : i32} : () -> i32
    return %x : i32
  }
  func.func @second() -> i32 {
    %x = "ask.constant"() {ask.id = "same", value = 2 : i32} : () -> i32
    return %x : i32
  }
}
