; Restricted ABI fixture, not a substitute for real compiler export.
target datalayout = "e-p:64:64-i64:64-n16:32:64"
define ptx_kernel void @scale2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) readnone captures(none) %2, ptr addrspace(1) readnone captures(none) %3) local_unnamed_addr #0 {
  ret void
}
attributes #0 = { nounwind "nvvm.reqntid"="128" }
