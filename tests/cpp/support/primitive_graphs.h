#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ask::test {
inline std::string tensor(const std::vector<int64_t> &shape, const std::string &dtype) {
  std::string result = "tensor<";
  for (auto n : shape) result += std::to_string(n) + "x";
  return result + dtype + ">";
}
inline std::string dimensions(const std::vector<int64_t> &values) {
  std::string result = "array<i64";
  for (std::size_t i = 0; i < values.size(); ++i)
    result += (i ? ", " : ": ") + std::to_string(values[i]);
  return result + ">";
}
inline std::string pointwise(const std::vector<int64_t> &shape, const std::string &dtype) {
  auto type = tensor(shape, dtype), mask = tensor(shape, "i1");
  return "module { func.func @pointwise(%x: " + type + ", %y: " + type + ") -> " + type + " {\n"
      "%q = \"ask.div\"(%x, %y) : (" + type + ", " + type + ") -> " + type + "\n"
      "%e = \"ask.exp\"(%q) : (" + type + ") -> " + type + "\n"
      "%l = \"ask.log\"(%e) : (" + type + ") -> " + type + "\n"
      "%r = \"ask.sqrt\"(%l) : (" + type + ") -> " + type + "\n"
      "%c = \"ask.compare\"(%q, %y) {predicate = \"lt\"} : (" + type + ", " + type + ") -> " + mask + "\n"
      "%s = \"ask.select\"(%c, %r, %q) : (" + mask + ", " + type + ", " + type + ") -> " + type + "\n"
      "return %s : " + type + " } }";
}
inline std::string spatial(const std::string &dtype) {
  auto input = tensor({1, 3}, dtype), wide = tensor({2, 3}, dtype), output = tensor({3, 2}, dtype);
  return "module { func.func @spatial(%x: " + input + ") -> " + output + " {\n"
      "%b = \"ask.broadcast_in_dim\"(%x) {shape = array<i64: 2, 3>, dimensions = array<i64: 0, 1>} : (" + input + ") -> " + wide + "\n"
      "%t = \"ask.transpose\"(%b) {permutation = array<i64: 1, 0>} : (" + wide + ") -> " + output + "\n"
      "return %t : " + output + " } }";
}
inline std::string reductionBroadcast(bool full, const std::string &operation) {
  std::string reduced = full ? "f32" : "tensor<2xf32>";
  return "module { func.func @reduce_broadcast(%x: tensor<2x3xf32>) -> tensor<2x3xf32> {\n"
      "%s = \"ask." + operation + "\"(%x) {axes = " + dimensions(full ? std::vector<int64_t>{0, 1} : std::vector<int64_t>{1}) + "} : (tensor<2x3xf32>) -> " + reduced + "\n"
      "%b = \"ask.broadcast_in_dim\"(%s) {shape = array<i64: 2, 3>, dimensions = " + dimensions(full ? std::vector<int64_t>{} : std::vector<int64_t>{0}) + "} : (" + reduced + ") -> tensor<2x3xf32>\n"
      "%r = \"ask.div\"(%x, %b) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>\n"
      "return %r : tensor<2x3xf32> } }";
}
inline std::string matmul(const std::string &dtype, int64_t batch, bool transposeA, bool transposeB) {
  std::vector<int64_t> a, b, c;
  if (batch > 0) { a.push_back(batch);b.push_back(batch);c.push_back(batch); }
  a.insert(a.end(), transposeA ? std::initializer_list<int64_t>{3, 5} : std::initializer_list<int64_t>{5, 3});
  b.insert(b.end(), transposeB ? std::initializer_list<int64_t>{7, 3} : std::initializer_list<int64_t>{3, 7});
  c.insert(c.end(), {5, 7});
  auto at = tensor(a, dtype), bt = tensor(b, dtype), ct = tensor(c, dtype);
  return "module { func.func @matmul(%a: " + at + ", %b: " + bt + ") -> " + ct + " {\n"
      "%x = \"ask.mul\"(%a, %a) : (" + at + ", " + at + ") -> " + at + "\n"
      "%y = \"ask.matmul\"(%x, %b) {transpose_a = " + (transposeA ? "true" : "false") + ", transpose_b = " + (transposeB ? "true" : "false") + "} : (" + at + ", " + bt + ") -> " + ct + "\n"
      "%z = \"ask.add\"(%y, %y) : (" + ct + ", " + ct + ") -> " + ct + "\n"
      "return %z : " + ct + " } }";
}
inline std::string reductionExpansion(int64_t extent, bool reduceAxis, const std::string &operation) {
  const std::vector<int64_t> shape{2, extent};
  auto reducedShape = reduceAxis ? std::vector<int64_t>{2} : shape;
  auto expandedShape = reducedShape;
  expandedShape.insert(expandedShape.begin(), 4);
  auto input = tensor(shape, "f32"), reduced = tensor(reducedShape, "f32"), output = tensor(expandedShape, "f32");
  auto axes = dimensions(reduceAxis ? std::vector<int64_t>{1} : std::vector<int64_t>{});
  auto mapping = dimensions(reduceAxis ? std::vector<int64_t>{1} : std::vector<int64_t>{1, 2});
  return "module { func.func @expand_reduction(%x: " + input + ") -> " + output + " {\n"
      "%r = \"ask." + operation + "\"(%x) {axes = " + axes + "} : (" + input + ") -> " + reduced + "\n"
      "%y = \"ask.broadcast_in_dim\"(%r) {shape = " + dimensions(expandedShape) + ", dimensions = " + mapping + "} : (" + reduced + ") -> " + output + "\n"
      "return %y : " + output + " } }";
}
inline std::string scalarReductionExpansion(int64_t inputSize, int64_t outputSize, const std::string &operation) {
  auto input = tensor({inputSize}, "f32"), output = tensor({outputSize}, "f32");
  return "module { func.func @scalar_expansion(%x: " + input + ") -> " + output + " {\n"
      "%r = \"ask." + operation + "\"(%x) {axes = array<i64: 0>} : (" + input + ") -> f32\n"
      "%y = \"ask.splat\"(%r) {shape = " + dimensions({outputSize}) + "} : (f32) -> " + output + "\n"
      "return %y : " + output + " } }";
}
inline std::string publicScalar(const std::string &dtype, bool constant) {
  auto type = tensor({7}, dtype);
  if (constant)
    return "module { func.func @scalar_constant() -> " + dtype + " {\n"
        "%c = \"ask.constant\"() {value = 2.0 : " + dtype + "} : () -> " + dtype + "\n"
        "return %c : " + dtype + " } }";
  return "module { func.func @scalar_ports(%x: " + type + ", %s: " + dtype + ") -> (" + type + ", " + dtype + ") {\n"
      "%b = \"ask.splat\"(%s) {shape = array<i64: 7>} : (" + dtype + ") -> " + type + "\n"
      "%y = \"ask.add\"(%x, %b) : (" + type + ", " + type + ") -> " + type + "\n"
      "%z = \"ask.exp\"(%s) : (" + dtype + ") -> " + dtype + "\n"
      "return %y, %z : " + type + ", " + dtype + " } }";
}
}
