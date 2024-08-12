; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%Qubit = type opaque
%Result = type opaque

@cstr.72303030303000 = private constant [7 x i8] c"r00000\00"

declare void @__quantum__qis__x__body(%Qubit*) local_unnamed_addr

declare void @__quantum__qis__ry__body(double, %Qubit*) local_unnamed_addr

declare void @__quantum__qis__cnot__body(%Qubit*, %Qubit*) local_unnamed_addr

declare void @__quantum__rt__result_record_output(%Result*, i8*) local_unnamed_addr

declare void @__quantum__qis__mz__body(%Qubit*, %Result* writeonly) local_unnamed_addr #0

define void @__nvqpp__mlirgen__ansatz() local_unnamed_addr #1 {
  tail call void @__quantum__qis__x__body(%Qubit* null)
  tail call void @__quantum__qis__ry__body(double 5.900000e-01, %Qubit* nonnull inttoptr (i64 1 to %Qubit*))
  tail call void @__quantum__qis__cnot__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Qubit* null)
  tail call void @__quantum__qis__mz__body(%Qubit* nonnull inttoptr (i64 1 to %Qubit*), %Result* writeonly null)
  tail call void @__quantum__rt__result_record_output(%Result* null, i8* nonnull getelementptr inbounds ([7 x i8], [7 x i8]* @cstr.72303030303000, i64 0, i64 0))
  ret void
}

attributes #0 = { "irreversible" }
attributes #1 = { "entry_point" "output_labeling_schema"="schema_id" "output_names"="[[[0,[1,\22r00000\22]]]]" "qir_profiles"="base_profile" "requiredQubits"="2" "requiredResults"="1" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !{i32 1, !"qir_major_version", i32 1}
!2 = !{i32 7, !"qir_minor_version", i32 0}
!3 = !{i32 1, !"dynamic_qubit_management", i1 false}
!4 = !{i32 1, !"dynamic_result_management", i1 false}