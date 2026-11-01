// A known schema field cannot silently change type, even under the fixed rule.
module {
  "ask.semantics"() {
    sym_name = "fixed", function_type = () -> tensor<10xi32>,
    shape_rule = "fixed", shape_operand = "first", effects = "none"
  } : () -> ()
  func.func @main() {
    return
  }
}
