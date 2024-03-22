
%Tuple = type opaque
%Array = type opaque
%Callable = type opaque
%Range = type { i64, i64, i64 }
%Qubit = type opaque
%String = type opaque
%Result = type opaque

@PartialApplication__1__FunctionTable = internal constant [4 x void (%Tuple*, %Tuple*, %Tuple*)*] [void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__1__body__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__1__adj__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__1__ctl__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__1__ctladj__wrapper]
@Microsoft__Quantum__Intrinsic__Rx__FunctionTable = internal constant [4 x void (%Tuple*, %Tuple*, %Tuple*)*] [void (%Tuple*, %Tuple*, %Tuple*)* @Microsoft__Quantum__Intrinsic__Rx__body__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Microsoft__Quantum__Intrinsic__Rx__adj__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Microsoft__Quantum__Intrinsic__Rx__ctl__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Microsoft__Quantum__Intrinsic__Rx__ctladj__wrapper]
@MemoryManagement__1__FunctionTable = internal constant [2 x void (%Tuple*, i32)*] [void (%Tuple*, i32)* @MemoryManagement__1__RefCount, void (%Tuple*, i32)* @MemoryManagement__1__AliasCount]
@PartialApplication__2__FunctionTable = internal constant [4 x void (%Tuple*, %Tuple*, %Tuple*)*] [void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__2__body__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__2__adj__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__2__ctl__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__2__ctladj__wrapper]
@PartialApplication__3__FunctionTable = internal constant [4 x void (%Tuple*, %Tuple*, %Tuple*)*] [void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__3__body__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__3__adj__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__3__ctl__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__3__ctladj__wrapper]
@PartialApplication__4__FunctionTable = internal constant [4 x void (%Tuple*, %Tuple*, %Tuple*)*] [void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__4__body__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__4__adj__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__4__ctl__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Lifted__PartialApplication__4__ctladj__wrapper]
@0 = internal constant [80 x i8] c"Currently, HamiltonianCouplings only supports given constraints for 6 segments.\00"
@1 = internal constant [68 x i8] c"Currently, IsSatisfactory only supports constraints for 6 segments.\00"
@2 = internal constant [40 x i8] c"timeZ and timeX are not the same length\00"
@Microsoft__Quantum__Intrinsic__H__FunctionTable = internal constant [4 x void (%Tuple*, %Tuple*, %Tuple*)*] [void (%Tuple*, %Tuple*, %Tuple*)* @Microsoft__Quantum__Intrinsic__H__body__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Microsoft__Quantum__Intrinsic__H__adj__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Microsoft__Quantum__Intrinsic__H__ctl__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* @Microsoft__Quantum__Intrinsic__H__ctladj__wrapper]
@3 = internal constant [10 x i8] c"result = \00"
@4 = internal constant [3 x i8] c", \00"
@5 = internal constant [2 x i8] c"[\00"
@6 = internal constant [5 x i8] c"true\00"
@7 = internal constant [6 x i8] c"false\00"
@8 = internal constant [2 x i8] c"]\00"
@9 = internal constant [10 x i8] c", cost = \00"
@10 = internal constant [18 x i8] c", satisfactory = \00"
@11 = internal constant [24 x i8] c"Simulation is complete\0A\00"
@12 = internal constant [23 x i8] c"Best itinerary found: \00"
@13 = internal constant [36 x i8] c"% of runs found the best itinerary\0A\00"
@14 = internal constant [2 x i8] c"\22\00"
@15 = internal constant [13 x i8] c"\0A\09Expected:\09\00"
@16 = internal constant [11 x i8] c"\0A\09Actual:\09\00"
@Microsoft__Quantum__Convert__ResultAsBool__FunctionTable = internal constant [4 x void (%Tuple*, %Tuple*, %Tuple*)*] [void (%Tuple*, %Tuple*, %Tuple*)* @Microsoft__Quantum__Convert__ResultAsBool__body__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* null, void (%Tuple*, %Tuple*, %Tuple*)* null, void (%Tuple*, %Tuple*, %Tuple*)* null]
@Microsoft__Quantum__Intrinsic__M__FunctionTable = internal constant [4 x void (%Tuple*, %Tuple*, %Tuple*)*] [void (%Tuple*, %Tuple*, %Tuple*)* @Microsoft__Quantum__Intrinsic__M__body__wrapper, void (%Tuple*, %Tuple*, %Tuple*)* null, void (%Tuple*, %Tuple*, %Tuple*)* null, void (%Tuple*, %Tuple*, %Tuple*)* null]
@17 = internal constant [3 x i8] c"()\00"

define internal void @Microsoft__Quantum__Samples__QAOA__ApplyDriverHamiltonian__body(double %time, %Array* %target) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %target, i32 1)
  %0 = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @Microsoft__Quantum__Intrinsic__Rx__FunctionTable, [2 x void (%Tuple*, i32)*]* null, %Tuple* null)
  %1 = fmul double -2.000000e+00, %time
  %2 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Callable*, double }* getelementptr ({ %Callable*, double }, { %Callable*, double }* null, i32 1) to i64))
  %3 = bitcast %Tuple* %2 to { %Callable*, double }*
  %4 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %3, i32 0, i32 0
  %5 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %3, i32 0, i32 1
  store %Callable* %0, %Callable** %4, align 8
  store double %1, double* %5, align 8
  %6 = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @PartialApplication__1__FunctionTable, [2 x void (%Tuple*, i32)*]* @MemoryManagement__1__FunctionTable, %Tuple* %2)
  call void @Microsoft__Quantum__Canon___ce218f8734ae41e6b081ce1fbc7f6eb7_ApplyToEachCA__body(%Callable* %6, %Array* %target)
  call void @__quantum__rt__array_update_alias_count(%Array* %target, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %6, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %6, i32 -1)
  ret void
}

declare void @__quantum__rt__array_update_alias_count(%Array*, i32)

define internal void @Microsoft__Quantum__Canon___ce218f8734ae41e6b081ce1fbc7f6eb7_ApplyToEachCA__body(%Callable* %singleElementOperation, %Array* %register) {
entry:
  call void @__quantum__rt__capture_update_alias_count(%Callable* %singleElementOperation, i32 1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %singleElementOperation, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %register, i32 1)
  %0 = call %Range @Microsoft__Quantum__Arrays___aff6ba86ed7b447f90c2e700aa07a9b4_IndexRange__body(%Array* %register)
  %1 = extractvalue %Range %0, 0
  %2 = extractvalue %Range %0, 1
  %3 = extractvalue %Range %0, 2
  br label %preheader__1

preheader__1:                                     ; preds = %entry
  %4 = icmp sgt i64 %2, 0
  br label %header__1

header__1:                                        ; preds = %exiting__1, %preheader__1
  %idxQubit = phi i64 [ %1, %preheader__1 ], [ %14, %exiting__1 ]
  %5 = icmp sle i64 %idxQubit, %3
  %6 = icmp sge i64 %idxQubit, %3
  %7 = select i1 %4, i1 %5, i1 %6
  br i1 %7, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %8 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %register, i64 %idxQubit)
  %9 = bitcast i8* %8 to %Qubit**
  %10 = load %Qubit*, %Qubit** %9, align 8
  %11 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Qubit* }* getelementptr ({ %Qubit* }, { %Qubit* }* null, i32 1) to i64))
  %12 = bitcast %Tuple* %11 to { %Qubit* }*
  %13 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %12, i32 0, i32 0
  store %Qubit* %10, %Qubit** %13, align 8
  call void @__quantum__rt__callable_invoke(%Callable* %singleElementOperation, %Tuple* %11, %Tuple* null)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %11, i32 -1)
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %14 = add i64 %idxQubit, %2
  br label %header__1

exit__1:                                          ; preds = %header__1
  call void @__quantum__rt__capture_update_alias_count(%Callable* %singleElementOperation, i32 -1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %singleElementOperation, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %register, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__1__body__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %1 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 1
  %2 = load double, double* %1, align 8
  %3 = bitcast %Tuple* %arg-tuple to { %Qubit* }*
  %4 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %3, i32 0, i32 0
  %5 = load %Qubit*, %Qubit** %4, align 8
  %6 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %7 = bitcast %Tuple* %6 to { double, %Qubit* }*
  %8 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 0
  %9 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 1
  store double %2, double* %8, align 8
  store %Qubit* %5, %Qubit** %9, align 8
  %10 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 0
  %11 = load %Callable*, %Callable** %10, align 8
  call void @__quantum__rt__callable_invoke(%Callable* %11, %Tuple* %6, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %6, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__1__adj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %1 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 1
  %2 = load double, double* %1, align 8
  %3 = bitcast %Tuple* %arg-tuple to { %Qubit* }*
  %4 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %3, i32 0, i32 0
  %5 = load %Qubit*, %Qubit** %4, align 8
  %6 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %7 = bitcast %Tuple* %6 to { double, %Qubit* }*
  %8 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 0
  %9 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 1
  store double %2, double* %8, align 8
  store %Qubit* %5, %Qubit** %9, align 8
  %10 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 0
  %11 = load %Callable*, %Callable** %10, align 8
  %12 = call %Callable* @__quantum__rt__callable_copy(%Callable* %11, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %12, i32 1)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %12)
  call void @__quantum__rt__callable_invoke(%Callable* %12, %Tuple* %6, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %6, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %12, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %12, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__1__ctl__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Array*, %Qubit* }*
  %1 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 1
  %3 = load %Array*, %Array** %1, align 8
  %4 = load %Qubit*, %Qubit** %2, align 8
  %5 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %6 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 1
  %7 = load double, double* %6, align 8
  %8 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %9 = bitcast %Tuple* %8 to { double, %Qubit* }*
  %10 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 0
  %11 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 1
  store double %7, double* %10, align 8
  store %Qubit* %4, %Qubit** %11, align 8
  %12 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Array*, { double, %Qubit* }* }* getelementptr ({ %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* null, i32 1) to i64))
  %13 = bitcast %Tuple* %12 to { %Array*, { double, %Qubit* }* }*
  %14 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 0
  %15 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 1
  store %Array* %3, %Array** %14, align 8
  store { double, %Qubit* }* %9, { double, %Qubit* }** %15, align 8
  %16 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 0
  %17 = load %Callable*, %Callable** %16, align 8
  %18 = call %Callable* @__quantum__rt__callable_copy(%Callable* %17, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 1)
  call void @__quantum__rt__callable_make_controlled(%Callable* %18)
  call void @__quantum__rt__callable_invoke(%Callable* %18, %Tuple* %12, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %8, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %12, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %18, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__1__ctladj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Array*, %Qubit* }*
  %1 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 1
  %3 = load %Array*, %Array** %1, align 8
  %4 = load %Qubit*, %Qubit** %2, align 8
  %5 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %6 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 1
  %7 = load double, double* %6, align 8
  %8 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %9 = bitcast %Tuple* %8 to { double, %Qubit* }*
  %10 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 0
  %11 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 1
  store double %7, double* %10, align 8
  store %Qubit* %4, %Qubit** %11, align 8
  %12 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Array*, { double, %Qubit* }* }* getelementptr ({ %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* null, i32 1) to i64))
  %13 = bitcast %Tuple* %12 to { %Array*, { double, %Qubit* }* }*
  %14 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 0
  %15 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 1
  store %Array* %3, %Array** %14, align 8
  store { double, %Qubit* }* %9, { double, %Qubit* }** %15, align 8
  %16 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 0
  %17 = load %Callable*, %Callable** %16, align 8
  %18 = call %Callable* @__quantum__rt__callable_copy(%Callable* %17, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 1)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %18)
  call void @__quantum__rt__callable_make_controlled(%Callable* %18)
  call void @__quantum__rt__callable_invoke(%Callable* %18, %Tuple* %12, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %8, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %12, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %18, i32 -1)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__Rx__body__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { double, %Qubit* }*
  %1 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %0, i32 0, i32 1
  %3 = load double, double* %1, align 8
  %4 = load %Qubit*, %Qubit** %2, align 8
  call void @Microsoft__Quantum__Intrinsic__Rx__body(double %3, %Qubit* %4)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__Rx__adj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { double, %Qubit* }*
  %1 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %0, i32 0, i32 1
  %3 = load double, double* %1, align 8
  %4 = load %Qubit*, %Qubit** %2, align 8
  call void @Microsoft__Quantum__Intrinsic__Rx__adj(double %3, %Qubit* %4)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__Rx__ctl__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Array*, { double, %Qubit* }* }*
  %1 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %0, i32 0, i32 1
  %3 = load %Array*, %Array** %1, align 8
  %4 = load { double, %Qubit* }*, { double, %Qubit* }** %2, align 8
  call void @Microsoft__Quantum__Intrinsic__Rx__ctl(%Array* %3, { double, %Qubit* }* %4)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__Rx__ctladj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Array*, { double, %Qubit* }* }*
  %1 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %0, i32 0, i32 1
  %3 = load %Array*, %Array** %1, align 8
  %4 = load { double, %Qubit* }*, { double, %Qubit* }** %2, align 8
  call void @Microsoft__Quantum__Intrinsic__Rx__ctladj(%Array* %3, { double, %Qubit* }* %4)
  ret void
}

declare %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]*, [2 x void (%Tuple*, i32)*]*, %Tuple*)

declare %Tuple* @__quantum__rt__tuple_create(i64)

define internal void @MemoryManagement__1__RefCount(%Tuple* %capture-tuple, i32 %count-change) {
entry:
  %0 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %1 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 0
  %2 = load %Callable*, %Callable** %1, align 8
  call void @__quantum__rt__capture_update_reference_count(%Callable* %2, i32 %count-change)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %2, i32 %count-change)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %capture-tuple, i32 %count-change)
  ret void
}

define internal void @MemoryManagement__1__AliasCount(%Tuple* %capture-tuple, i32 %count-change) {
entry:
  %0 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %1 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 0
  %2 = load %Callable*, %Callable** %1, align 8
  call void @__quantum__rt__capture_update_alias_count(%Callable* %2, i32 %count-change)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %2, i32 %count-change)
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %capture-tuple, i32 %count-change)
  ret void
}

declare void @__quantum__rt__capture_update_reference_count(%Callable*, i32)

declare void @__quantum__rt__callable_update_reference_count(%Callable*, i32)

define internal void @Microsoft__Quantum__Intrinsic__Rx__body(double %theta, %Qubit* %qubit) {
entry:
  call void @__quantum__qis__r__body(i2 1, double %theta, %Qubit* %qubit)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__Rx__adj(double %theta, %Qubit* %qubit) {
entry:
  %theta__1 = fneg double %theta
  call void @__quantum__qis__r__body(i2 1, double %theta__1, %Qubit* %qubit)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__Rx__ctl(%Array* %__controlQubits__, { double, %Qubit* }* %0) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %1 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %0, i32 0, i32 0
  %theta = load double, double* %1, align 8
  %2 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %0, i32 0, i32 1
  %qubit = load %Qubit*, %Qubit** %2, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %3 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ i2, double, %Qubit* }* getelementptr ({ i2, double, %Qubit* }, { i2, double, %Qubit* }* null, i32 1) to i64))
  %4 = bitcast %Tuple* %3 to { i2, double, %Qubit* }*
  %5 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %4, i32 0, i32 0
  %6 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %4, i32 0, i32 1
  %7 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %4, i32 0, i32 2
  store i2 1, i2* %5, align 1
  store double %theta, double* %6, align 8
  store %Qubit* %qubit, %Qubit** %7, align 8
  call void @__quantum__qis__r__ctl(%Array* %__controlQubits__, { i2, double, %Qubit* }* %4)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %3, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__Rx__ctladj(%Array* %__controlQubits__, { double, %Qubit* }* %0) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %1 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %0, i32 0, i32 0
  %theta = load double, double* %1, align 8
  %2 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %0, i32 0, i32 1
  %qubit = load %Qubit*, %Qubit** %2, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %theta__1 = fneg double %theta
  %3 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ i2, double, %Qubit* }* getelementptr ({ i2, double, %Qubit* }, { i2, double, %Qubit* }* null, i32 1) to i64))
  %4 = bitcast %Tuple* %3 to { i2, double, %Qubit* }*
  %5 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %4, i32 0, i32 0
  %6 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %4, i32 0, i32 1
  %7 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %4, i32 0, i32 2
  store i2 1, i2* %5, align 1
  store double %theta__1, double* %6, align 8
  store %Qubit* %qubit, %Qubit** %7, align 8
  call void @__quantum__qis__r__ctl(%Array* %__controlQubits__, { i2, double, %Qubit* }* %4)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %3, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

declare void @__quantum__rt__callable_invoke(%Callable*, %Tuple*, %Tuple*)

declare void @__quantum__rt__tuple_update_reference_count(%Tuple*, i32)

declare %Callable* @__quantum__rt__callable_copy(%Callable*, i1)

declare void @__quantum__rt__callable_make_adjoint(%Callable*)

declare void @__quantum__rt__callable_make_controlled(%Callable*)

declare void @__quantum__rt__capture_update_alias_count(%Callable*, i32)

declare void @__quantum__rt__callable_update_alias_count(%Callable*, i32)

declare void @__quantum__rt__tuple_update_alias_count(%Tuple*, i32)

define internal void @Microsoft__Quantum__Samples__QAOA__ApplyDriverHamiltonian__adj(double %time, %Array* %target) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %target, i32 1)
  %0 = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @Microsoft__Quantum__Intrinsic__Rx__FunctionTable, [2 x void (%Tuple*, i32)*]* null, %Tuple* null)
  %1 = fmul double -2.000000e+00, %time
  %2 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Callable*, double }* getelementptr ({ %Callable*, double }, { %Callable*, double }* null, i32 1) to i64))
  %3 = bitcast %Tuple* %2 to { %Callable*, double }*
  %4 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %3, i32 0, i32 0
  %5 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %3, i32 0, i32 1
  store %Callable* %0, %Callable** %4, align 8
  store double %1, double* %5, align 8
  %6 = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @PartialApplication__2__FunctionTable, [2 x void (%Tuple*, i32)*]* @MemoryManagement__1__FunctionTable, %Tuple* %2)
  call void @Microsoft__Quantum__Canon___ce218f8734ae41e6b081ce1fbc7f6eb7_ApplyToEachCA__adj(%Callable* %6, %Array* %target)
  call void @__quantum__rt__array_update_alias_count(%Array* %target, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %6, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %6, i32 -1)
  ret void
}

define internal void @Microsoft__Quantum__Canon___ce218f8734ae41e6b081ce1fbc7f6eb7_ApplyToEachCA__adj(%Callable* %singleElementOperation, %Array* %register) {
entry:
  call void @__quantum__rt__capture_update_alias_count(%Callable* %singleElementOperation, i32 1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %singleElementOperation, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %register, i32 1)
  %0 = call %Range @Microsoft__Quantum__Arrays___aff6ba86ed7b447f90c2e700aa07a9b4_IndexRange__body(%Array* %register)
  %1 = extractvalue %Range %0, 0
  %2 = extractvalue %Range %0, 1
  %3 = extractvalue %Range %0, 2
  %4 = sub i64 %3, %1
  %5 = sdiv i64 %4, %2
  %6 = mul i64 %2, %5
  %7 = add i64 %1, %6
  %8 = sub i64 0, %2
  %9 = insertvalue %Range zeroinitializer, i64 %7, 0
  %10 = insertvalue %Range %9, i64 %8, 1
  %11 = insertvalue %Range %10, i64 %1, 2
  %12 = extractvalue %Range %11, 0
  %13 = extractvalue %Range %11, 1
  %14 = extractvalue %Range %11, 2
  br label %preheader__1

preheader__1:                                     ; preds = %entry
  %15 = icmp sgt i64 %13, 0
  br label %header__1

header__1:                                        ; preds = %exiting__1, %preheader__1
  %__qsVar0__idxQubit__ = phi i64 [ %12, %preheader__1 ], [ %26, %exiting__1 ]
  %16 = icmp sle i64 %__qsVar0__idxQubit__, %14
  %17 = icmp sge i64 %__qsVar0__idxQubit__, %14
  %18 = select i1 %15, i1 %16, i1 %17
  br i1 %18, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %19 = call %Callable* @__quantum__rt__callable_copy(%Callable* %singleElementOperation, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %19, i32 1)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %19)
  %20 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %register, i64 %__qsVar0__idxQubit__)
  %21 = bitcast i8* %20 to %Qubit**
  %22 = load %Qubit*, %Qubit** %21, align 8
  %23 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Qubit* }* getelementptr ({ %Qubit* }, { %Qubit* }* null, i32 1) to i64))
  %24 = bitcast %Tuple* %23 to { %Qubit* }*
  %25 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %24, i32 0, i32 0
  store %Qubit* %22, %Qubit** %25, align 8
  call void @__quantum__rt__callable_invoke(%Callable* %19, %Tuple* %23, %Tuple* null)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %19, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %19, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %23, i32 -1)
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %26 = add i64 %__qsVar0__idxQubit__, %13
  br label %header__1

exit__1:                                          ; preds = %header__1
  call void @__quantum__rt__capture_update_alias_count(%Callable* %singleElementOperation, i32 -1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %singleElementOperation, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %register, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__2__body__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %1 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 1
  %2 = load double, double* %1, align 8
  %3 = bitcast %Tuple* %arg-tuple to { %Qubit* }*
  %4 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %3, i32 0, i32 0
  %5 = load %Qubit*, %Qubit** %4, align 8
  %6 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %7 = bitcast %Tuple* %6 to { double, %Qubit* }*
  %8 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 0
  %9 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 1
  store double %2, double* %8, align 8
  store %Qubit* %5, %Qubit** %9, align 8
  %10 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 0
  %11 = load %Callable*, %Callable** %10, align 8
  call void @__quantum__rt__callable_invoke(%Callable* %11, %Tuple* %6, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %6, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__2__adj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %1 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 1
  %2 = load double, double* %1, align 8
  %3 = bitcast %Tuple* %arg-tuple to { %Qubit* }*
  %4 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %3, i32 0, i32 0
  %5 = load %Qubit*, %Qubit** %4, align 8
  %6 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %7 = bitcast %Tuple* %6 to { double, %Qubit* }*
  %8 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 0
  %9 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 1
  store double %2, double* %8, align 8
  store %Qubit* %5, %Qubit** %9, align 8
  %10 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 0
  %11 = load %Callable*, %Callable** %10, align 8
  %12 = call %Callable* @__quantum__rt__callable_copy(%Callable* %11, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %12, i32 1)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %12)
  call void @__quantum__rt__callable_invoke(%Callable* %12, %Tuple* %6, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %6, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %12, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %12, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__2__ctl__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Array*, %Qubit* }*
  %1 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 1
  %3 = load %Array*, %Array** %1, align 8
  %4 = load %Qubit*, %Qubit** %2, align 8
  %5 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %6 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 1
  %7 = load double, double* %6, align 8
  %8 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %9 = bitcast %Tuple* %8 to { double, %Qubit* }*
  %10 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 0
  %11 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 1
  store double %7, double* %10, align 8
  store %Qubit* %4, %Qubit** %11, align 8
  %12 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Array*, { double, %Qubit* }* }* getelementptr ({ %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* null, i32 1) to i64))
  %13 = bitcast %Tuple* %12 to { %Array*, { double, %Qubit* }* }*
  %14 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 0
  %15 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 1
  store %Array* %3, %Array** %14, align 8
  store { double, %Qubit* }* %9, { double, %Qubit* }** %15, align 8
  %16 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 0
  %17 = load %Callable*, %Callable** %16, align 8
  %18 = call %Callable* @__quantum__rt__callable_copy(%Callable* %17, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 1)
  call void @__quantum__rt__callable_make_controlled(%Callable* %18)
  call void @__quantum__rt__callable_invoke(%Callable* %18, %Tuple* %12, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %8, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %12, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %18, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__2__ctladj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Array*, %Qubit* }*
  %1 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 1
  %3 = load %Array*, %Array** %1, align 8
  %4 = load %Qubit*, %Qubit** %2, align 8
  %5 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %6 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 1
  %7 = load double, double* %6, align 8
  %8 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %9 = bitcast %Tuple* %8 to { double, %Qubit* }*
  %10 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 0
  %11 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 1
  store double %7, double* %10, align 8
  store %Qubit* %4, %Qubit** %11, align 8
  %12 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Array*, { double, %Qubit* }* }* getelementptr ({ %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* null, i32 1) to i64))
  %13 = bitcast %Tuple* %12 to { %Array*, { double, %Qubit* }* }*
  %14 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 0
  %15 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 1
  store %Array* %3, %Array** %14, align 8
  store { double, %Qubit* }* %9, { double, %Qubit* }** %15, align 8
  %16 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 0
  %17 = load %Callable*, %Callable** %16, align 8
  %18 = call %Callable* @__quantum__rt__callable_copy(%Callable* %17, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 1)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %18)
  call void @__quantum__rt__callable_make_controlled(%Callable* %18)
  call void @__quantum__rt__callable_invoke(%Callable* %18, %Tuple* %12, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %8, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %12, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %18, i32 -1)
  ret void
}

define internal void @Microsoft__Quantum__Samples__QAOA__ApplyDriverHamiltonian__ctl(%Array* %__controlQubits__, { double, %Array* }* %0) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %1 = getelementptr inbounds { double, %Array* }, { double, %Array* }* %0, i32 0, i32 0
  %time = load double, double* %1, align 8
  %2 = getelementptr inbounds { double, %Array* }, { double, %Array* }* %0, i32 0, i32 1
  %target = load %Array*, %Array** %2, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %target, i32 1)
  %3 = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @Microsoft__Quantum__Intrinsic__Rx__FunctionTable, [2 x void (%Tuple*, i32)*]* null, %Tuple* null)
  %4 = fmul double -2.000000e+00, %time
  %5 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Callable*, double }* getelementptr ({ %Callable*, double }, { %Callable*, double }* null, i32 1) to i64))
  %6 = bitcast %Tuple* %5 to { %Callable*, double }*
  %7 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %6, i32 0, i32 0
  %8 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %6, i32 0, i32 1
  store %Callable* %3, %Callable** %7, align 8
  store double %4, double* %8, align 8
  %9 = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @PartialApplication__3__FunctionTable, [2 x void (%Tuple*, i32)*]* @MemoryManagement__1__FunctionTable, %Tuple* %5)
  call void @__quantum__rt__array_update_reference_count(%Array* %target, i32 1)
  %10 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Callable*, %Array* }* getelementptr ({ %Callable*, %Array* }, { %Callable*, %Array* }* null, i32 1) to i64))
  %11 = bitcast %Tuple* %10 to { %Callable*, %Array* }*
  %12 = getelementptr inbounds { %Callable*, %Array* }, { %Callable*, %Array* }* %11, i32 0, i32 0
  %13 = getelementptr inbounds { %Callable*, %Array* }, { %Callable*, %Array* }* %11, i32 0, i32 1
  store %Callable* %9, %Callable** %12, align 8
  store %Array* %target, %Array** %13, align 8
  call void @Microsoft__Quantum__Canon___ce218f8734ae41e6b081ce1fbc7f6eb7_ApplyToEachCA__ctl(%Array* %__controlQubits__, { %Callable*, %Array* }* %11)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %target, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %9, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %9, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %target, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %10, i32 -1)
  ret void
}

define internal void @Microsoft__Quantum__Canon___ce218f8734ae41e6b081ce1fbc7f6eb7_ApplyToEachCA__ctl(%Array* %__controlQubits__, { %Callable*, %Array* }* %0) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %1 = getelementptr inbounds { %Callable*, %Array* }, { %Callable*, %Array* }* %0, i32 0, i32 0
  %singleElementOperation = load %Callable*, %Callable** %1, align 8
  call void @__quantum__rt__capture_update_alias_count(%Callable* %singleElementOperation, i32 1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %singleElementOperation, i32 1)
  %2 = getelementptr inbounds { %Callable*, %Array* }, { %Callable*, %Array* }* %0, i32 0, i32 1
  %register = load %Array*, %Array** %2, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %register, i32 1)
  %3 = call %Range @Microsoft__Quantum__Arrays___aff6ba86ed7b447f90c2e700aa07a9b4_IndexRange__body(%Array* %register)
  %4 = extractvalue %Range %3, 0
  %5 = extractvalue %Range %3, 1
  %6 = extractvalue %Range %3, 2
  br label %preheader__1

preheader__1:                                     ; preds = %entry
  %7 = icmp sgt i64 %5, 0
  br label %header__1

header__1:                                        ; preds = %exiting__1, %preheader__1
  %idxQubit = phi i64 [ %4, %preheader__1 ], [ %19, %exiting__1 ]
  %8 = icmp sle i64 %idxQubit, %6
  %9 = icmp sge i64 %idxQubit, %6
  %10 = select i1 %7, i1 %8, i1 %9
  br i1 %10, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %11 = call %Callable* @__quantum__rt__callable_copy(%Callable* %singleElementOperation, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %11, i32 1)
  call void @__quantum__rt__callable_make_controlled(%Callable* %11)
  call void @__quantum__rt__array_update_reference_count(%Array* %__controlQubits__, i32 1)
  %12 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %register, i64 %idxQubit)
  %13 = bitcast i8* %12 to %Qubit**
  %14 = load %Qubit*, %Qubit** %13, align 8
  %15 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Array*, %Qubit* }* getelementptr ({ %Array*, %Qubit* }, { %Array*, %Qubit* }* null, i32 1) to i64))
  %16 = bitcast %Tuple* %15 to { %Array*, %Qubit* }*
  %17 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %16, i32 0, i32 0
  %18 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %16, i32 0, i32 1
  store %Array* %__controlQubits__, %Array** %17, align 8
  store %Qubit* %14, %Qubit** %18, align 8
  call void @__quantum__rt__callable_invoke(%Callable* %11, %Tuple* %15, %Tuple* null)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %11, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %11, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %__controlQubits__, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %15, i32 -1)
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %19 = add i64 %idxQubit, %5
  br label %header__1

exit__1:                                          ; preds = %header__1
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  call void @__quantum__rt__capture_update_alias_count(%Callable* %singleElementOperation, i32 -1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %singleElementOperation, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %register, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__3__body__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %1 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 1
  %2 = load double, double* %1, align 8
  %3 = bitcast %Tuple* %arg-tuple to { %Qubit* }*
  %4 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %3, i32 0, i32 0
  %5 = load %Qubit*, %Qubit** %4, align 8
  %6 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %7 = bitcast %Tuple* %6 to { double, %Qubit* }*
  %8 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 0
  %9 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 1
  store double %2, double* %8, align 8
  store %Qubit* %5, %Qubit** %9, align 8
  %10 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 0
  %11 = load %Callable*, %Callable** %10, align 8
  call void @__quantum__rt__callable_invoke(%Callable* %11, %Tuple* %6, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %6, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__3__adj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %1 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 1
  %2 = load double, double* %1, align 8
  %3 = bitcast %Tuple* %arg-tuple to { %Qubit* }*
  %4 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %3, i32 0, i32 0
  %5 = load %Qubit*, %Qubit** %4, align 8
  %6 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %7 = bitcast %Tuple* %6 to { double, %Qubit* }*
  %8 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 0
  %9 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 1
  store double %2, double* %8, align 8
  store %Qubit* %5, %Qubit** %9, align 8
  %10 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 0
  %11 = load %Callable*, %Callable** %10, align 8
  %12 = call %Callable* @__quantum__rt__callable_copy(%Callable* %11, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %12, i32 1)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %12)
  call void @__quantum__rt__callable_invoke(%Callable* %12, %Tuple* %6, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %6, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %12, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %12, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__3__ctl__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Array*, %Qubit* }*
  %1 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 1
  %3 = load %Array*, %Array** %1, align 8
  %4 = load %Qubit*, %Qubit** %2, align 8
  %5 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %6 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 1
  %7 = load double, double* %6, align 8
  %8 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %9 = bitcast %Tuple* %8 to { double, %Qubit* }*
  %10 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 0
  %11 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 1
  store double %7, double* %10, align 8
  store %Qubit* %4, %Qubit** %11, align 8
  %12 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Array*, { double, %Qubit* }* }* getelementptr ({ %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* null, i32 1) to i64))
  %13 = bitcast %Tuple* %12 to { %Array*, { double, %Qubit* }* }*
  %14 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 0
  %15 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 1
  store %Array* %3, %Array** %14, align 8
  store { double, %Qubit* }* %9, { double, %Qubit* }** %15, align 8
  %16 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 0
  %17 = load %Callable*, %Callable** %16, align 8
  %18 = call %Callable* @__quantum__rt__callable_copy(%Callable* %17, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 1)
  call void @__quantum__rt__callable_make_controlled(%Callable* %18)
  call void @__quantum__rt__callable_invoke(%Callable* %18, %Tuple* %12, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %8, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %12, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %18, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__3__ctladj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Array*, %Qubit* }*
  %1 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 1
  %3 = load %Array*, %Array** %1, align 8
  %4 = load %Qubit*, %Qubit** %2, align 8
  %5 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %6 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 1
  %7 = load double, double* %6, align 8
  %8 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %9 = bitcast %Tuple* %8 to { double, %Qubit* }*
  %10 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 0
  %11 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 1
  store double %7, double* %10, align 8
  store %Qubit* %4, %Qubit** %11, align 8
  %12 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Array*, { double, %Qubit* }* }* getelementptr ({ %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* null, i32 1) to i64))
  %13 = bitcast %Tuple* %12 to { %Array*, { double, %Qubit* }* }*
  %14 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 0
  %15 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 1
  store %Array* %3, %Array** %14, align 8
  store { double, %Qubit* }* %9, { double, %Qubit* }** %15, align 8
  %16 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 0
  %17 = load %Callable*, %Callable** %16, align 8
  %18 = call %Callable* @__quantum__rt__callable_copy(%Callable* %17, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 1)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %18)
  call void @__quantum__rt__callable_make_controlled(%Callable* %18)
  call void @__quantum__rt__callable_invoke(%Callable* %18, %Tuple* %12, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %8, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %12, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %18, i32 -1)
  ret void
}

declare void @__quantum__rt__array_update_reference_count(%Array*, i32)

define internal void @Microsoft__Quantum__Samples__QAOA__ApplyDriverHamiltonian__ctladj(%Array* %__controlQubits__, { double, %Array* }* %0) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %1 = getelementptr inbounds { double, %Array* }, { double, %Array* }* %0, i32 0, i32 0
  %time = load double, double* %1, align 8
  %2 = getelementptr inbounds { double, %Array* }, { double, %Array* }* %0, i32 0, i32 1
  %target = load %Array*, %Array** %2, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %target, i32 1)
  %3 = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @Microsoft__Quantum__Intrinsic__Rx__FunctionTable, [2 x void (%Tuple*, i32)*]* null, %Tuple* null)
  %4 = fmul double -2.000000e+00, %time
  %5 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Callable*, double }* getelementptr ({ %Callable*, double }, { %Callable*, double }* null, i32 1) to i64))
  %6 = bitcast %Tuple* %5 to { %Callable*, double }*
  %7 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %6, i32 0, i32 0
  %8 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %6, i32 0, i32 1
  store %Callable* %3, %Callable** %7, align 8
  store double %4, double* %8, align 8
  %9 = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @PartialApplication__4__FunctionTable, [2 x void (%Tuple*, i32)*]* @MemoryManagement__1__FunctionTable, %Tuple* %5)
  call void @__quantum__rt__array_update_reference_count(%Array* %target, i32 1)
  %10 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Callable*, %Array* }* getelementptr ({ %Callable*, %Array* }, { %Callable*, %Array* }* null, i32 1) to i64))
  %11 = bitcast %Tuple* %10 to { %Callable*, %Array* }*
  %12 = getelementptr inbounds { %Callable*, %Array* }, { %Callable*, %Array* }* %11, i32 0, i32 0
  %13 = getelementptr inbounds { %Callable*, %Array* }, { %Callable*, %Array* }* %11, i32 0, i32 1
  store %Callable* %9, %Callable** %12, align 8
  store %Array* %target, %Array** %13, align 8
  call void @Microsoft__Quantum__Canon___ce218f8734ae41e6b081ce1fbc7f6eb7_ApplyToEachCA__ctladj(%Array* %__controlQubits__, { %Callable*, %Array* }* %11)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %target, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %9, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %9, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %target, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %10, i32 -1)
  ret void
}

define internal void @Microsoft__Quantum__Canon___ce218f8734ae41e6b081ce1fbc7f6eb7_ApplyToEachCA__ctladj(%Array* %__controlQubits__, { %Callable*, %Array* }* %0) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %1 = getelementptr inbounds { %Callable*, %Array* }, { %Callable*, %Array* }* %0, i32 0, i32 0
  %singleElementOperation = load %Callable*, %Callable** %1, align 8
  call void @__quantum__rt__capture_update_alias_count(%Callable* %singleElementOperation, i32 1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %singleElementOperation, i32 1)
  %2 = getelementptr inbounds { %Callable*, %Array* }, { %Callable*, %Array* }* %0, i32 0, i32 1
  %register = load %Array*, %Array** %2, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %register, i32 1)
  %3 = call %Range @Microsoft__Quantum__Arrays___aff6ba86ed7b447f90c2e700aa07a9b4_IndexRange__body(%Array* %register)
  %4 = extractvalue %Range %3, 0
  %5 = extractvalue %Range %3, 1
  %6 = extractvalue %Range %3, 2
  %7 = sub i64 %6, %4
  %8 = sdiv i64 %7, %5
  %9 = mul i64 %5, %8
  %10 = add i64 %4, %9
  %11 = sub i64 0, %5
  %12 = insertvalue %Range zeroinitializer, i64 %10, 0
  %13 = insertvalue %Range %12, i64 %11, 1
  %14 = insertvalue %Range %13, i64 %4, 2
  %15 = extractvalue %Range %14, 0
  %16 = extractvalue %Range %14, 1
  %17 = extractvalue %Range %14, 2
  br label %preheader__1

preheader__1:                                     ; preds = %entry
  %18 = icmp sgt i64 %16, 0
  br label %header__1

header__1:                                        ; preds = %exiting__1, %preheader__1
  %__qsVar0__idxQubit__ = phi i64 [ %15, %preheader__1 ], [ %30, %exiting__1 ]
  %19 = icmp sle i64 %__qsVar0__idxQubit__, %17
  %20 = icmp sge i64 %__qsVar0__idxQubit__, %17
  %21 = select i1 %18, i1 %19, i1 %20
  br i1 %21, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %22 = call %Callable* @__quantum__rt__callable_copy(%Callable* %singleElementOperation, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %22, i32 1)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %22)
  call void @__quantum__rt__callable_make_controlled(%Callable* %22)
  call void @__quantum__rt__array_update_reference_count(%Array* %__controlQubits__, i32 1)
  %23 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %register, i64 %__qsVar0__idxQubit__)
  %24 = bitcast i8* %23 to %Qubit**
  %25 = load %Qubit*, %Qubit** %24, align 8
  %26 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Array*, %Qubit* }* getelementptr ({ %Array*, %Qubit* }, { %Array*, %Qubit* }* null, i32 1) to i64))
  %27 = bitcast %Tuple* %26 to { %Array*, %Qubit* }*
  %28 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %27, i32 0, i32 0
  %29 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %27, i32 0, i32 1
  store %Array* %__controlQubits__, %Array** %28, align 8
  store %Qubit* %25, %Qubit** %29, align 8
  call void @__quantum__rt__callable_invoke(%Callable* %22, %Tuple* %26, %Tuple* null)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %22, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %22, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %__controlQubits__, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %26, i32 -1)
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %30 = add i64 %__qsVar0__idxQubit__, %16
  br label %header__1

exit__1:                                          ; preds = %header__1
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  call void @__quantum__rt__capture_update_alias_count(%Callable* %singleElementOperation, i32 -1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %singleElementOperation, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %register, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__4__body__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %1 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 1
  %2 = load double, double* %1, align 8
  %3 = bitcast %Tuple* %arg-tuple to { %Qubit* }*
  %4 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %3, i32 0, i32 0
  %5 = load %Qubit*, %Qubit** %4, align 8
  %6 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %7 = bitcast %Tuple* %6 to { double, %Qubit* }*
  %8 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 0
  %9 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 1
  store double %2, double* %8, align 8
  store %Qubit* %5, %Qubit** %9, align 8
  %10 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 0
  %11 = load %Callable*, %Callable** %10, align 8
  call void @__quantum__rt__callable_invoke(%Callable* %11, %Tuple* %6, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %6, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__4__adj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %1 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 1
  %2 = load double, double* %1, align 8
  %3 = bitcast %Tuple* %arg-tuple to { %Qubit* }*
  %4 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %3, i32 0, i32 0
  %5 = load %Qubit*, %Qubit** %4, align 8
  %6 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %7 = bitcast %Tuple* %6 to { double, %Qubit* }*
  %8 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 0
  %9 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 1
  store double %2, double* %8, align 8
  store %Qubit* %5, %Qubit** %9, align 8
  %10 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %0, i32 0, i32 0
  %11 = load %Callable*, %Callable** %10, align 8
  %12 = call %Callable* @__quantum__rt__callable_copy(%Callable* %11, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %12, i32 1)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %12)
  call void @__quantum__rt__callable_invoke(%Callable* %12, %Tuple* %6, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %6, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %12, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %12, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__4__ctl__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Array*, %Qubit* }*
  %1 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 1
  %3 = load %Array*, %Array** %1, align 8
  %4 = load %Qubit*, %Qubit** %2, align 8
  %5 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %6 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 1
  %7 = load double, double* %6, align 8
  %8 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %9 = bitcast %Tuple* %8 to { double, %Qubit* }*
  %10 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 0
  %11 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 1
  store double %7, double* %10, align 8
  store %Qubit* %4, %Qubit** %11, align 8
  %12 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Array*, { double, %Qubit* }* }* getelementptr ({ %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* null, i32 1) to i64))
  %13 = bitcast %Tuple* %12 to { %Array*, { double, %Qubit* }* }*
  %14 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 0
  %15 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 1
  store %Array* %3, %Array** %14, align 8
  store { double, %Qubit* }* %9, { double, %Qubit* }** %15, align 8
  %16 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 0
  %17 = load %Callable*, %Callable** %16, align 8
  %18 = call %Callable* @__quantum__rt__callable_copy(%Callable* %17, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 1)
  call void @__quantum__rt__callable_make_controlled(%Callable* %18)
  call void @__quantum__rt__callable_invoke(%Callable* %18, %Tuple* %12, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %8, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %12, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %18, i32 -1)
  ret void
}

define internal void @Lifted__PartialApplication__4__ctladj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Array*, %Qubit* }*
  %1 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 1
  %3 = load %Array*, %Array** %1, align 8
  %4 = load %Qubit*, %Qubit** %2, align 8
  %5 = bitcast %Tuple* %capture-tuple to { %Callable*, double }*
  %6 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 1
  %7 = load double, double* %6, align 8
  %8 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %9 = bitcast %Tuple* %8 to { double, %Qubit* }*
  %10 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 0
  %11 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %9, i32 0, i32 1
  store double %7, double* %10, align 8
  store %Qubit* %4, %Qubit** %11, align 8
  %12 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Array*, { double, %Qubit* }* }* getelementptr ({ %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* null, i32 1) to i64))
  %13 = bitcast %Tuple* %12 to { %Array*, { double, %Qubit* }* }*
  %14 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 0
  %15 = getelementptr inbounds { %Array*, { double, %Qubit* }* }, { %Array*, { double, %Qubit* }* }* %13, i32 0, i32 1
  store %Array* %3, %Array** %14, align 8
  store { double, %Qubit* }* %9, { double, %Qubit* }** %15, align 8
  %16 = getelementptr inbounds { %Callable*, double }, { %Callable*, double }* %5, i32 0, i32 0
  %17 = load %Callable*, %Callable** %16, align 8
  %18 = call %Callable* @__quantum__rt__callable_copy(%Callable* %17, i1 false)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 1)
  call void @__quantum__rt__callable_make_adjoint(%Callable* %18)
  call void @__quantum__rt__callable_make_controlled(%Callable* %18)
  call void @__quantum__rt__callable_invoke(%Callable* %18, %Tuple* %12, %Tuple* %result-tuple)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %8, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %12, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %18, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %18, i32 -1)
  ret void
}

define internal void @Microsoft__Quantum__Samples__QAOA__ApplyInstanceHamiltonian__body(i64 %numSegments, double %time, %Array* %weights, %Array* %coupling, %Array* %target) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %weights, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %coupling, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %target, i32 1)
  %auxiliary = call %Qubit* @__quantum__rt__qubit_allocate()
  %0 = call %Array* @Microsoft__Quantum__Arrays___1a054001fedc47909895b06dea008574_Zipped__body(%Array* %weights, %Array* %target)
  %1 = call i64 @__quantum__rt__array_get_size_1d(%Array* %0)
  %2 = sub i64 %1, 1
  br label %header__1

header__1:                                        ; preds = %exiting__1, %entry
  %3 = phi i64 [ 0, %entry ], [ %12, %exiting__1 ]
  %4 = icmp sle i64 %3, %2
  br i1 %4, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %5 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %0, i64 %3)
  %6 = bitcast i8* %5 to { double, %Qubit* }**
  %7 = load { double, %Qubit* }*, { double, %Qubit* }** %6, align 8
  %8 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 0
  %h = load double, double* %8, align 8
  %9 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %7, i32 0, i32 1
  %qubit = load %Qubit*, %Qubit** %9, align 8
  %10 = fmul double 2.000000e+00, %time
  %11 = fmul double %10, %h
  call void @Microsoft__Quantum__Intrinsic__Rz__body(double %11, %Qubit* %qubit)
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %12 = add i64 %3, 1
  br label %header__1

exit__1:                                          ; preds = %header__1
  br label %header__2

header__2:                                        ; preds = %exiting__2, %exit__1
  %i = phi i64 [ 0, %exit__1 ], [ %15, %exiting__2 ]
  %13 = icmp sle i64 %i, 5
  br i1 %13, label %body__2, label %exit__2

body__2:                                          ; preds = %header__2
  %14 = add i64 %i, 1
  br label %header__3

exiting__2:                                       ; preds = %exit__3
  %15 = add i64 %i, 1
  br label %header__2

exit__2:                                          ; preds = %header__2
  %16 = sub i64 %1, 1
  br label %header__4

header__3:                                        ; preds = %exiting__3, %body__2
  %j = phi i64 [ %14, %body__2 ], [ %37, %exiting__3 ]
  %17 = icmp sle i64 %j, 5
  br i1 %17, label %body__3, label %exit__3

body__3:                                          ; preds = %header__3
  %18 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %target, i64 %i)
  %19 = bitcast i8* %18 to %Qubit**
  %20 = load %Qubit*, %Qubit** %19, align 8
  call void @Microsoft__Quantum__Intrinsic__CNOT__body(%Qubit* %20, %Qubit* %auxiliary)
  %21 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %target, i64 %j)
  %22 = bitcast i8* %21 to %Qubit**
  %23 = load %Qubit*, %Qubit** %22, align 8
  call void @Microsoft__Quantum__Intrinsic__CNOT__body(%Qubit* %23, %Qubit* %auxiliary)
  %24 = fmul double 2.000000e+00, %time
  %25 = mul i64 %numSegments, %i
  %26 = add i64 %25, %j
  %27 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %coupling, i64 %26)
  %28 = bitcast i8* %27 to double*
  %29 = load double, double* %28, align 8
  %30 = fmul double %24, %29
  call void @Microsoft__Quantum__Intrinsic__Rz__body(double %30, %Qubit* %auxiliary)
  %31 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %target, i64 %j)
  %32 = bitcast i8* %31 to %Qubit**
  %33 = load %Qubit*, %Qubit** %32, align 8
  call void @Microsoft__Quantum__Intrinsic__CNOT__adj(%Qubit* %33, %Qubit* %auxiliary)
  %34 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %target, i64 %i)
  %35 = bitcast i8* %34 to %Qubit**
  %36 = load %Qubit*, %Qubit** %35, align 8
  call void @Microsoft__Quantum__Intrinsic__CNOT__adj(%Qubit* %36, %Qubit* %auxiliary)
  br label %exiting__3

exiting__3:                                       ; preds = %body__3
  %37 = add i64 %j, 1
  br label %header__3

exit__3:                                          ; preds = %header__3
  br label %exiting__2

header__4:                                        ; preds = %exiting__4, %exit__2
  %38 = phi i64 [ 0, %exit__2 ], [ %44, %exiting__4 ]
  %39 = icmp sle i64 %38, %16
  br i1 %39, label %body__4, label %exit__4

body__4:                                          ; preds = %header__4
  %40 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %0, i64 %38)
  %41 = bitcast i8* %40 to { double, %Qubit* }**
  %42 = load { double, %Qubit* }*, { double, %Qubit* }** %41, align 8
  %43 = bitcast { double, %Qubit* }* %42 to %Tuple*
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %43, i32 -1)
  br label %exiting__4

exiting__4:                                       ; preds = %body__4
  %44 = add i64 %38, 1
  br label %header__4

exit__4:                                          ; preds = %header__4
  call void @__quantum__rt__array_update_reference_count(%Array* %0, i32 -1)
  call void @__quantum__rt__qubit_release(%Qubit* %auxiliary)
  call void @__quantum__rt__array_update_alias_count(%Array* %weights, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %coupling, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %target, i32 -1)
  ret void
}

declare %Qubit* @__quantum__rt__qubit_allocate()

declare %Array* @__quantum__rt__qubit_allocate_array(i64)

declare void @__quantum__rt__qubit_release(%Qubit*)

define internal %Array* @Microsoft__Quantum__Arrays___1a054001fedc47909895b06dea008574_Zipped__body(%Array* %left, %Array* %right) {
entry:
  %output = alloca %Array*, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %left, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %right, i32 1)
  %0 = call i64 @__quantum__rt__array_get_size_1d(%Array* %left)
  %1 = call i64 @__quantum__rt__array_get_size_1d(%Array* %right)
  %2 = icmp slt i64 %0, %1
  br i1 %2, label %condTrue__1, label %condFalse__1

condTrue__1:                                      ; preds = %entry
  br label %condContinue__1

condFalse__1:                                     ; preds = %entry
  br label %condContinue__1

condContinue__1:                                  ; preds = %condFalse__1, %condTrue__1
  %nElements = phi i64 [ %0, %condTrue__1 ], [ %1, %condFalse__1 ]
  %3 = icmp eq i64 %nElements, 0
  br i1 %3, label %then0__1, label %continue__1

then0__1:                                         ; preds = %condContinue__1
  %4 = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 0)
  call void @__quantum__rt__array_update_alias_count(%Array* %left, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %right, i32 -1)
  ret %Array* %4

continue__1:                                      ; preds = %condContinue__1
  %5 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %left, i64 0)
  %6 = bitcast i8* %5 to double*
  %7 = load double, double* %6, align 8
  %8 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %right, i64 0)
  %9 = bitcast i8* %8 to %Qubit**
  %10 = load %Qubit*, %Qubit** %9, align 8
  %11 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %12 = bitcast %Tuple* %11 to { double, %Qubit* }*
  %13 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %12, i32 0, i32 0
  %14 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %12, i32 0, i32 1
  store double %7, double* %13, align 8
  store %Qubit* %10, %Qubit** %14, align 8
  %15 = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 %nElements)
  %16 = sub i64 %nElements, 1
  br label %header__1

header__1:                                        ; preds = %exiting__1, %continue__1
  %17 = phi i64 [ 0, %continue__1 ], [ %21, %exiting__1 ]
  %18 = icmp sle i64 %17, %16
  br i1 %18, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %19 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %15, i64 %17)
  %20 = bitcast i8* %19 to { double, %Qubit* }**
  store { double, %Qubit* }* %12, { double, %Qubit* }** %20, align 8
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %11, i32 1)
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %21 = add i64 %17, 1
  br label %header__1

exit__1:                                          ; preds = %header__1
  store %Array* %15, %Array** %output, align 8
  %22 = sub i64 %nElements, 1
  br label %header__2

header__2:                                        ; preds = %exiting__2, %exit__1
  %23 = phi i64 [ 0, %exit__1 ], [ %29, %exiting__2 ]
  %24 = icmp sle i64 %23, %22
  br i1 %24, label %body__2, label %exit__2

body__2:                                          ; preds = %header__2
  %25 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %15, i64 %23)
  %26 = bitcast i8* %25 to { double, %Qubit* }**
  %27 = load { double, %Qubit* }*, { double, %Qubit* }** %26, align 8
  %28 = bitcast { double, %Qubit* }* %27 to %Tuple*
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %28, i32 1)
  br label %exiting__2

exiting__2:                                       ; preds = %body__2
  %29 = add i64 %23, 1
  br label %header__2

exit__2:                                          ; preds = %header__2
  call void @__quantum__rt__array_update_alias_count(%Array* %15, i32 1)
  %30 = sub i64 %nElements, 1
  br label %header__3

header__3:                                        ; preds = %exiting__3, %exit__2
  %idxElement = phi i64 [ 1, %exit__2 ], [ %48, %exiting__3 ]
  %31 = icmp sle i64 %idxElement, %30
  br i1 %31, label %body__3, label %exit__3

body__3:                                          ; preds = %header__3
  %32 = load %Array*, %Array** %output, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %32, i32 -1)
  %33 = call %Array* @__quantum__rt__array_copy(%Array* %32, i1 false)
  %34 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %left, i64 %idxElement)
  %35 = bitcast i8* %34 to double*
  %36 = load double, double* %35, align 8
  %37 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %right, i64 %idxElement)
  %38 = bitcast i8* %37 to %Qubit**
  %39 = load %Qubit*, %Qubit** %38, align 8
  %40 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, %Qubit* }* getelementptr ({ double, %Qubit* }, { double, %Qubit* }* null, i32 1) to i64))
  %41 = bitcast %Tuple* %40 to { double, %Qubit* }*
  %42 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %41, i32 0, i32 0
  %43 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %41, i32 0, i32 1
  store double %36, double* %42, align 8
  store %Qubit* %39, %Qubit** %43, align 8
  %44 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %33, i64 %idxElement)
  %45 = bitcast i8* %44 to { double, %Qubit* }**
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %40, i32 1)
  %46 = load { double, %Qubit* }*, { double, %Qubit* }** %45, align 8
  %47 = bitcast { double, %Qubit* }* %46 to %Tuple*
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %47, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %47, i32 -1)
  store { double, %Qubit* }* %41, { double, %Qubit* }** %45, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %33, i32 1)
  store %Array* %33, %Array** %output, align 8
  call void @__quantum__rt__array_update_reference_count(%Array* %32, i32 -1)
  br label %exiting__3

exiting__3:                                       ; preds = %body__3
  %48 = add i64 %idxElement, 1
  br label %header__3

exit__3:                                          ; preds = %header__3
  %49 = load %Array*, %Array** %output, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %left, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %right, i32 -1)
  %50 = call i64 @__quantum__rt__array_get_size_1d(%Array* %49)
  %51 = sub i64 %50, 1
  br label %header__4

header__4:                                        ; preds = %exiting__4, %exit__3
  %52 = phi i64 [ 0, %exit__3 ], [ %58, %exiting__4 ]
  %53 = icmp sle i64 %52, %51
  br i1 %53, label %body__4, label %exit__4

body__4:                                          ; preds = %header__4
  %54 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %49, i64 %52)
  %55 = bitcast i8* %54 to { double, %Qubit* }**
  %56 = load { double, %Qubit* }*, { double, %Qubit* }** %55, align 8
  %57 = bitcast { double, %Qubit* }* %56 to %Tuple*
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %57, i32 -1)
  br label %exiting__4

exiting__4:                                       ; preds = %body__4
  %58 = add i64 %52, 1
  br label %header__4

exit__4:                                          ; preds = %header__4
  call void @__quantum__rt__array_update_alias_count(%Array* %49, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %11, i32 -1)
  ret %Array* %49
}

declare i64 @__quantum__rt__array_get_size_1d(%Array*)

declare i8* @__quantum__rt__array_get_element_ptr_1d(%Array*, i64)

define internal void @Microsoft__Quantum__Intrinsic__Rz__body(double %theta, %Qubit* %qubit) {
entry:
  call void @__quantum__qis__r__body(i2 -2, double %theta, %Qubit* %qubit)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__CNOT__body(%Qubit* %control, %Qubit* %target) {
entry:
  %__controlQubits__ = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 1)
  %0 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %__controlQubits__, i64 0)
  %1 = bitcast i8* %0 to %Qubit**
  store %Qubit* %control, %Qubit** %1, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  call void @__quantum__qis__x__ctl(%Array* %__controlQubits__, %Qubit* %target)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__CNOT__adj(%Qubit* %control, %Qubit* %target) {
entry:
  call void @Microsoft__Quantum__Intrinsic__CNOT__body(%Qubit* %control, %Qubit* %target)
  ret void
}

define internal double @Microsoft__Quantum__Samples__QAOA__CalculatedCost__body(%Array* %segmentCosts, %Array* %usedSegments) {
entry:
  %finalCost = alloca double, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %segmentCosts, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %usedSegments, i32 1)
  store double 0.000000e+00, double* %finalCost, align 8
  %0 = call %Array* @Microsoft__Quantum__Arrays___79fcaa5209264c34b71b6465b12b47e0_Zipped__body(%Array* %segmentCosts, %Array* %usedSegments)
  %1 = call i64 @__quantum__rt__array_get_size_1d(%Array* %0)
  %2 = sub i64 %1, 1
  br label %header__1

header__1:                                        ; preds = %exiting__1, %entry
  %3 = phi i64 [ 0, %entry ], [ %13, %exiting__1 ]
  %4 = icmp sle i64 %3, %2
  br i1 %4, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %5 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %0, i64 %3)
  %6 = bitcast i8* %5 to { double, i1 }**
  %7 = load { double, i1 }*, { double, i1 }** %6, align 8
  %8 = getelementptr inbounds { double, i1 }, { double, i1 }* %7, i32 0, i32 0
  %cost = load double, double* %8, align 8
  %9 = getelementptr inbounds { double, i1 }, { double, i1 }* %7, i32 0, i32 1
  %segment = load i1, i1* %9, align 1
  %10 = load double, double* %finalCost, align 8
  %11 = select i1 %segment, double %cost, double 0.000000e+00
  %12 = fadd double %10, %11
  store double %12, double* %finalCost, align 8
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %13 = add i64 %3, 1
  br label %header__1

exit__1:                                          ; preds = %header__1
  %14 = load double, double* %finalCost, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %segmentCosts, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %usedSegments, i32 -1)
  %15 = sub i64 %1, 1
  br label %header__2

header__2:                                        ; preds = %exiting__2, %exit__1
  %16 = phi i64 [ 0, %exit__1 ], [ %22, %exiting__2 ]
  %17 = icmp sle i64 %16, %15
  br i1 %17, label %body__2, label %exit__2

body__2:                                          ; preds = %header__2
  %18 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %0, i64 %16)
  %19 = bitcast i8* %18 to { double, i1 }**
  %20 = load { double, i1 }*, { double, i1 }** %19, align 8
  %21 = bitcast { double, i1 }* %20 to %Tuple*
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %21, i32 -1)
  br label %exiting__2

exiting__2:                                       ; preds = %body__2
  %22 = add i64 %16, 1
  br label %header__2

exit__2:                                          ; preds = %header__2
  call void @__quantum__rt__array_update_reference_count(%Array* %0, i32 -1)
  ret double %14
}

define internal %Array* @Microsoft__Quantum__Arrays___79fcaa5209264c34b71b6465b12b47e0_Zipped__body(%Array* %left, %Array* %right) {
entry:
  %output = alloca %Array*, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %left, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %right, i32 1)
  %0 = call i64 @__quantum__rt__array_get_size_1d(%Array* %left)
  %1 = call i64 @__quantum__rt__array_get_size_1d(%Array* %right)
  %2 = icmp slt i64 %0, %1
  br i1 %2, label %condTrue__1, label %condFalse__1

condTrue__1:                                      ; preds = %entry
  br label %condContinue__1

condFalse__1:                                     ; preds = %entry
  br label %condContinue__1

condContinue__1:                                  ; preds = %condFalse__1, %condTrue__1
  %nElements = phi i64 [ %0, %condTrue__1 ], [ %1, %condFalse__1 ]
  %3 = icmp eq i64 %nElements, 0
  br i1 %3, label %then0__1, label %continue__1

then0__1:                                         ; preds = %condContinue__1
  %4 = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 0)
  call void @__quantum__rt__array_update_alias_count(%Array* %left, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %right, i32 -1)
  ret %Array* %4

continue__1:                                      ; preds = %condContinue__1
  %5 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %left, i64 0)
  %6 = bitcast i8* %5 to double*
  %7 = load double, double* %6, align 8
  %8 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %right, i64 0)
  %9 = bitcast i8* %8 to i1*
  %10 = load i1, i1* %9, align 1
  %11 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, i1 }* getelementptr ({ double, i1 }, { double, i1 }* null, i32 1) to i64))
  %12 = bitcast %Tuple* %11 to { double, i1 }*
  %13 = getelementptr inbounds { double, i1 }, { double, i1 }* %12, i32 0, i32 0
  %14 = getelementptr inbounds { double, i1 }, { double, i1 }* %12, i32 0, i32 1
  store double %7, double* %13, align 8
  store i1 %10, i1* %14, align 1
  %15 = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 %nElements)
  %16 = sub i64 %nElements, 1
  br label %header__1

header__1:                                        ; preds = %exiting__1, %continue__1
  %17 = phi i64 [ 0, %continue__1 ], [ %21, %exiting__1 ]
  %18 = icmp sle i64 %17, %16
  br i1 %18, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %19 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %15, i64 %17)
  %20 = bitcast i8* %19 to { double, i1 }**
  store { double, i1 }* %12, { double, i1 }** %20, align 8
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %11, i32 1)
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %21 = add i64 %17, 1
  br label %header__1

exit__1:                                          ; preds = %header__1
  store %Array* %15, %Array** %output, align 8
  %22 = sub i64 %nElements, 1
  br label %header__2

header__2:                                        ; preds = %exiting__2, %exit__1
  %23 = phi i64 [ 0, %exit__1 ], [ %29, %exiting__2 ]
  %24 = icmp sle i64 %23, %22
  br i1 %24, label %body__2, label %exit__2

body__2:                                          ; preds = %header__2
  %25 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %15, i64 %23)
  %26 = bitcast i8* %25 to { double, i1 }**
  %27 = load { double, i1 }*, { double, i1 }** %26, align 8
  %28 = bitcast { double, i1 }* %27 to %Tuple*
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %28, i32 1)
  br label %exiting__2

exiting__2:                                       ; preds = %body__2
  %29 = add i64 %23, 1
  br label %header__2

exit__2:                                          ; preds = %header__2
  call void @__quantum__rt__array_update_alias_count(%Array* %15, i32 1)
  %30 = sub i64 %nElements, 1
  br label %header__3

header__3:                                        ; preds = %exiting__3, %exit__2
  %idxElement = phi i64 [ 1, %exit__2 ], [ %48, %exiting__3 ]
  %31 = icmp sle i64 %idxElement, %30
  br i1 %31, label %body__3, label %exit__3

body__3:                                          ; preds = %header__3
  %32 = load %Array*, %Array** %output, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %32, i32 -1)
  %33 = call %Array* @__quantum__rt__array_copy(%Array* %32, i1 false)
  %34 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %left, i64 %idxElement)
  %35 = bitcast i8* %34 to double*
  %36 = load double, double* %35, align 8
  %37 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %right, i64 %idxElement)
  %38 = bitcast i8* %37 to i1*
  %39 = load i1, i1* %38, align 1
  %40 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, i1 }* getelementptr ({ double, i1 }, { double, i1 }* null, i32 1) to i64))
  %41 = bitcast %Tuple* %40 to { double, i1 }*
  %42 = getelementptr inbounds { double, i1 }, { double, i1 }* %41, i32 0, i32 0
  %43 = getelementptr inbounds { double, i1 }, { double, i1 }* %41, i32 0, i32 1
  store double %36, double* %42, align 8
  store i1 %39, i1* %43, align 1
  %44 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %33, i64 %idxElement)
  %45 = bitcast i8* %44 to { double, i1 }**
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %40, i32 1)
  %46 = load { double, i1 }*, { double, i1 }** %45, align 8
  %47 = bitcast { double, i1 }* %46 to %Tuple*
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %47, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %47, i32 -1)
  store { double, i1 }* %41, { double, i1 }** %45, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %33, i32 1)
  store %Array* %33, %Array** %output, align 8
  call void @__quantum__rt__array_update_reference_count(%Array* %32, i32 -1)
  br label %exiting__3

exiting__3:                                       ; preds = %body__3
  %48 = add i64 %idxElement, 1
  br label %header__3

exit__3:                                          ; preds = %header__3
  %49 = load %Array*, %Array** %output, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %left, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %right, i32 -1)
  %50 = call i64 @__quantum__rt__array_get_size_1d(%Array* %49)
  %51 = sub i64 %50, 1
  br label %header__4

header__4:                                        ; preds = %exiting__4, %exit__3
  %52 = phi i64 [ 0, %exit__3 ], [ %58, %exiting__4 ]
  %53 = icmp sle i64 %52, %51
  br i1 %53, label %body__4, label %exit__4

body__4:                                          ; preds = %header__4
  %54 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %49, i64 %52)
  %55 = bitcast i8* %54 to { double, i1 }**
  %56 = load { double, i1 }*, { double, i1 }** %55, align 8
  %57 = bitcast { double, i1 }* %56 to %Tuple*
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %57, i32 -1)
  br label %exiting__4

exiting__4:                                       ; preds = %body__4
  %58 = add i64 %52, 1
  br label %header__4

exit__4:                                          ; preds = %header__4
  call void @__quantum__rt__array_update_alias_count(%Array* %49, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %11, i32 -1)
  ret %Array* %49
}

define internal %Array* @Microsoft__Quantum__Samples__QAOA__HamiltonianCouplings__body(double %penalty, i64 %numSegments) {
entry:
  %0 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([80 x i8], [80 x i8]* @0, i32 0, i32 0))
  call void @Microsoft__Quantum__Diagnostics__EqualityFactI__body(i64 %numSegments, i64 6, %String* %0)
  %1 = mul i64 %numSegments, %numSegments
  %2 = fmul double 2.000000e+00, %penalty
  %3 = call %Array* @Microsoft__Quantum__Arrays___8814cf220af4433ca4db2593be92ac86_ConstantArray__body(i64 %1, double %2)
  %4 = call %Array* @__quantum__rt__array_copy(%Array* %3, i1 false)
  %5 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %4, i64 2)
  %6 = bitcast i8* %5 to double*
  store double %penalty, double* %6, align 8
  call void @__quantum__rt__array_update_reference_count(%Array* %3, i32 -1)
  %7 = call %Array* @__quantum__rt__array_copy(%Array* %4, i1 false)
  %8 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %7, i64 9)
  %9 = bitcast i8* %8 to double*
  store double %penalty, double* %9, align 8
  call void @__quantum__rt__array_update_reference_count(%Array* %4, i32 -1)
  %10 = call %Array* @__quantum__rt__array_copy(%Array* %7, i1 false)
  %11 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %10, i64 29)
  %12 = bitcast i8* %11 to double*
  store double %penalty, double* %12, align 8
  call void @__quantum__rt__string_update_reference_count(%String* %0, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %7, i32 -1)
  ret %Array* %10
}

define internal void @Microsoft__Quantum__Diagnostics__EqualityFactI__body(i64 %actual, i64 %expected, %String* %message) {
entry:
  %0 = icmp ne i64 %actual, %expected
  br i1 %0, label %then0__1, label %continue__1

then0__1:                                         ; preds = %entry
  call void @Microsoft__Quantum__Diagnostics___4a93b78d84c84bab83c4c604247a1c64___QsRef1__FormattedFailure____body(i64 %actual, i64 %expected, %String* %message)
  br label %continue__1

continue__1:                                      ; preds = %then0__1, %entry
  ret void
}

declare %String* @__quantum__rt__string_create(i8*)

define internal %Array* @Microsoft__Quantum__Arrays___8814cf220af4433ca4db2593be92ac86_ConstantArray__body(i64 %length, double %value) {
entry:
  %0 = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 %length)
  %1 = sub i64 %length, 1
  br label %header__1

header__1:                                        ; preds = %exiting__1, %entry
  %2 = phi i64 [ 0, %entry ], [ %6, %exiting__1 ]
  %3 = icmp sle i64 %2, %1
  br i1 %3, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %4 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %0, i64 %2)
  %5 = bitcast i8* %4 to double*
  store double %value, double* %5, align 8
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %6 = add i64 %2, 1
  br label %header__1

exit__1:                                          ; preds = %header__1
  ret %Array* %0
}

declare %Array* @__quantum__rt__array_copy(%Array*, i1)

declare void @__quantum__rt__string_update_reference_count(%String*, i32)

define internal %Array* @Microsoft__Quantum__Samples__QAOA__HamiltonianWeights__body(%Array* %segmentCosts, double %penalty, i64 %numSegments) {
entry:
  %weights = alloca %Array*, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %segmentCosts, i32 1)
  %0 = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 %numSegments)
  %1 = sub i64 %numSegments, 1
  br label %header__1

header__1:                                        ; preds = %exiting__1, %entry
  %2 = phi i64 [ 0, %entry ], [ %6, %exiting__1 ]
  %3 = icmp sle i64 %2, %1
  br i1 %3, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %4 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %0, i64 %2)
  %5 = bitcast i8* %4 to double*
  store double 0.000000e+00, double* %5, align 8
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %6 = add i64 %2, 1
  br label %header__1

exit__1:                                          ; preds = %header__1
  store %Array* %0, %Array** %weights, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %0, i32 1)
  %7 = sub i64 %numSegments, 1
  br label %header__2

header__2:                                        ; preds = %exiting__2, %exit__1
  %i = phi i64 [ 0, %exit__1 ], [ %19, %exiting__2 ]
  %8 = icmp sle i64 %i, %7
  br i1 %8, label %body__2, label %exit__2

body__2:                                          ; preds = %header__2
  %9 = load %Array*, %Array** %weights, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %9, i32 -1)
  %10 = call %Array* @__quantum__rt__array_copy(%Array* %9, i1 false)
  %11 = fmul double 4.000000e+00, %penalty
  %12 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %segmentCosts, i64 %i)
  %13 = bitcast i8* %12 to double*
  %14 = load double, double* %13, align 8
  %15 = fmul double 5.000000e-01, %14
  %16 = fsub double %11, %15
  %17 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %10, i64 %i)
  %18 = bitcast i8* %17 to double*
  store double %16, double* %18, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %10, i32 1)
  store %Array* %10, %Array** %weights, align 8
  call void @__quantum__rt__array_update_reference_count(%Array* %9, i32 -1)
  br label %exiting__2

exiting__2:                                       ; preds = %body__2
  %19 = add i64 %i, 1
  br label %header__2

exit__2:                                          ; preds = %header__2
  %20 = load %Array*, %Array** %weights, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %segmentCosts, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %20, i32 -1)
  ret %Array* %20
}

declare %Array* @__quantum__rt__array_create_1d(i32, i64)

define internal i1 @Microsoft__Quantum__Samples__QAOA__IsSatisfactory__body(i64 %numSegments, %Array* %usedSegments) {
entry:
  %hammingWeight = alloca i64, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %usedSegments, i32 1)
  %0 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([68 x i8], [68 x i8]* @1, i32 0, i32 0))
  call void @Microsoft__Quantum__Diagnostics__EqualityFactI__body(i64 %numSegments, i64 6, %String* %0)
  store i64 0, i64* %hammingWeight, align 4
  %1 = call i64 @__quantum__rt__array_get_size_1d(%Array* %usedSegments)
  %2 = sub i64 %1, 1
  br label %header__1

header__1:                                        ; preds = %exiting__1, %entry
  %3 = phi i64 [ 0, %entry ], [ %10, %exiting__1 ]
  %4 = icmp sle i64 %3, %2
  br i1 %4, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %5 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %usedSegments, i64 %3)
  %6 = bitcast i8* %5 to i1*
  %segment = load i1, i1* %6, align 1
  %7 = load i64, i64* %hammingWeight, align 4
  %8 = select i1 %segment, i64 1, i64 0
  %9 = add i64 %7, %8
  store i64 %9, i64* %hammingWeight, align 4
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %10 = add i64 %3, 1
  br label %header__1

exit__1:                                          ; preds = %header__1
  %11 = load i64, i64* %hammingWeight, align 4
  %12 = icmp ne i64 %11, 4
  br i1 %12, label %condContinue__1, label %condFalse__1

condFalse__1:                                     ; preds = %exit__1
  %13 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %usedSegments, i64 0)
  %14 = bitcast i8* %13 to i1*
  %15 = load i1, i1* %14, align 1
  %16 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %usedSegments, i64 2)
  %17 = bitcast i8* %16 to i1*
  %18 = load i1, i1* %17, align 1
  %19 = icmp ne i1 %15, %18
  br label %condContinue__1

condContinue__1:                                  ; preds = %condFalse__1, %exit__1
  %20 = phi i1 [ %12, %exit__1 ], [ %19, %condFalse__1 ]
  br i1 %20, label %condContinue__2, label %condFalse__2

condFalse__2:                                     ; preds = %condContinue__1
  %21 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %usedSegments, i64 1)
  %22 = bitcast i8* %21 to i1*
  %23 = load i1, i1* %22, align 1
  %24 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %usedSegments, i64 3)
  %25 = bitcast i8* %24 to i1*
  %26 = load i1, i1* %25, align 1
  %27 = icmp ne i1 %23, %26
  br label %condContinue__2

condContinue__2:                                  ; preds = %condFalse__2, %condContinue__1
  %28 = phi i1 [ %20, %condContinue__1 ], [ %27, %condFalse__2 ]
  br i1 %28, label %condContinue__3, label %condFalse__3

condFalse__3:                                     ; preds = %condContinue__2
  %29 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %usedSegments, i64 4)
  %30 = bitcast i8* %29 to i1*
  %31 = load i1, i1* %30, align 1
  %32 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %usedSegments, i64 5)
  %33 = bitcast i8* %32 to i1*
  %34 = load i1, i1* %33, align 1
  %35 = icmp ne i1 %31, %34
  br label %condContinue__3

condContinue__3:                                  ; preds = %condFalse__3, %condContinue__2
  %36 = phi i1 [ %28, %condContinue__2 ], [ %35, %condFalse__3 ]
  br i1 %36, label %then0__1, label %continue__1

then0__1:                                         ; preds = %condContinue__3
  call void @__quantum__rt__array_update_alias_count(%Array* %usedSegments, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %0, i32 -1)
  ret i1 false

continue__1:                                      ; preds = %condContinue__3
  call void @__quantum__rt__array_update_alias_count(%Array* %usedSegments, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %0, i32 -1)
  ret i1 true
}

define internal %Array* @Microsoft__Quantum__Samples__QAOA__PerformQAOA__body(i64 %numSegments, %Array* %weights, %Array* %couplings, %Array* %timeX, %Array* %timeZ) {
entry:
  %result = alloca %Array*, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %weights, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %couplings, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %timeX, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %timeZ, i32 1)
  %0 = call i64 @__quantum__rt__array_get_size_1d(%Array* %timeX)
  %1 = call i64 @__quantum__rt__array_get_size_1d(%Array* %timeZ)
  %2 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([40 x i8], [40 x i8]* @2, i32 0, i32 0))
  call void @Microsoft__Quantum__Diagnostics__EqualityFactI__body(i64 %0, i64 %1, %String* %2)
  %3 = call %Array* @__quantum__rt__array_create_1d(i32 1, i64 %numSegments)
  %4 = sub i64 %numSegments, 1
  br label %header__1

header__1:                                        ; preds = %exiting__1, %entry
  %5 = phi i64 [ 0, %entry ], [ %9, %exiting__1 ]
  %6 = icmp sle i64 %5, %4
  br i1 %6, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %7 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %3, i64 %5)
  %8 = bitcast i8* %7 to i1*
  store i1 false, i1* %8, align 1
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %9 = add i64 %5, 1
  br label %header__1

exit__1:                                          ; preds = %header__1
  store %Array* %3, %Array** %result, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %3, i32 1)
  %x = call %Array* @__quantum__rt__qubit_allocate_array(i64 %numSegments)
  call void @__quantum__rt__array_update_alias_count(%Array* %x, i32 1)
  %10 = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @Microsoft__Quantum__Intrinsic__H__FunctionTable, [2 x void (%Tuple*, i32)*]* null, %Tuple* null)
  call void @Microsoft__Quantum__Canon___b5aeea0ab28a48cdbe28d890f57d1344_ApplyToEach__body(%Callable* %10, %Array* %x)
  %11 = call %Array* @Microsoft__Quantum__Arrays___a54dc56da1c94499a3e1cf538d0da3c7_Zipped__body(%Array* %timeZ, %Array* %timeX)
  %12 = call i64 @__quantum__rt__array_get_size_1d(%Array* %11)
  %13 = sub i64 %12, 1
  br label %header__2

header__2:                                        ; preds = %exiting__2, %exit__1
  %14 = phi i64 [ 0, %exit__1 ], [ %21, %exiting__2 ]
  %15 = icmp sle i64 %14, %13
  br i1 %15, label %body__2, label %exit__2

body__2:                                          ; preds = %header__2
  %16 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %11, i64 %14)
  %17 = bitcast i8* %16 to { double, double }**
  %18 = load { double, double }*, { double, double }** %17, align 8
  %19 = getelementptr inbounds { double, double }, { double, double }* %18, i32 0, i32 0
  %tz = load double, double* %19, align 8
  %20 = getelementptr inbounds { double, double }, { double, double }* %18, i32 0, i32 1
  %tx = load double, double* %20, align 8
  call void @Microsoft__Quantum__Samples__QAOA__ApplyInstanceHamiltonian__body(i64 %numSegments, double %tz, %Array* %weights, %Array* %couplings, %Array* %x)
  call void @Microsoft__Quantum__Samples__QAOA__ApplyDriverHamiltonian__body(double %tx, %Array* %x)
  br label %exiting__2

exiting__2:                                       ; preds = %body__2
  %21 = add i64 %14, 1
  br label %header__2

exit__2:                                          ; preds = %header__2
  %22 = call %Array* @Microsoft__Quantum__Measurement__MultiM__body(%Array* %x)
  %23 = call %Array* @Microsoft__Quantum__Convert__ResultArrayAsBoolArray__body(%Array* %22)
  call void @__quantum__rt__array_update_alias_count(%Array* %x, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %weights, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %couplings, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %timeX, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %timeZ, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %3, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %2, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %10, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %10, i32 -1)
  %24 = sub i64 %12, 1
  br label %header__3

header__3:                                        ; preds = %exiting__3, %exit__2
  %25 = phi i64 [ 0, %exit__2 ], [ %31, %exiting__3 ]
  %26 = icmp sle i64 %25, %24
  br i1 %26, label %body__3, label %exit__3

body__3:                                          ; preds = %header__3
  %27 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %11, i64 %25)
  %28 = bitcast i8* %27 to { double, double }**
  %29 = load { double, double }*, { double, double }** %28, align 8
  %30 = bitcast { double, double }* %29 to %Tuple*
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %30, i32 -1)
  br label %exiting__3

exiting__3:                                       ; preds = %body__3
  %31 = add i64 %25, 1
  br label %header__3

exit__3:                                          ; preds = %header__3
  call void @__quantum__rt__array_update_reference_count(%Array* %11, i32 -1)
  %32 = call i64 @__quantum__rt__array_get_size_1d(%Array* %22)
  %33 = sub i64 %32, 1
  br label %header__4

header__4:                                        ; preds = %exiting__4, %exit__3
  %34 = phi i64 [ 0, %exit__3 ], [ %39, %exiting__4 ]
  %35 = icmp sle i64 %34, %33
  br i1 %35, label %body__4, label %exit__4

body__4:                                          ; preds = %header__4
  %36 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %22, i64 %34)
  %37 = bitcast i8* %36 to %Result**
  %38 = load %Result*, %Result** %37, align 8
  call void @__quantum__rt__result_update_reference_count(%Result* %38, i32 -1)
  br label %exiting__4

exiting__4:                                       ; preds = %body__4
  %39 = add i64 %34, 1
  br label %header__4

exit__4:                                          ; preds = %header__4
  call void @__quantum__rt__array_update_reference_count(%Array* %22, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %3, i32 -1)
  call void @__quantum__rt__qubit_release_array(%Array* %x)
  ret %Array* %23
}

declare void @__quantum__rt__qubit_release_array(%Array*)

define internal void @Microsoft__Quantum__Canon___b5aeea0ab28a48cdbe28d890f57d1344_ApplyToEach__body(%Callable* %singleElementOperation, %Array* %register) {
entry:
  call void @__quantum__rt__capture_update_alias_count(%Callable* %singleElementOperation, i32 1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %singleElementOperation, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %register, i32 1)
  %0 = call %Range @Microsoft__Quantum__Arrays___aff6ba86ed7b447f90c2e700aa07a9b4_IndexRange__body(%Array* %register)
  %1 = extractvalue %Range %0, 0
  %2 = extractvalue %Range %0, 1
  %3 = extractvalue %Range %0, 2
  br label %preheader__1

preheader__1:                                     ; preds = %entry
  %4 = icmp sgt i64 %2, 0
  br label %header__1

header__1:                                        ; preds = %exiting__1, %preheader__1
  %idxQubit = phi i64 [ %1, %preheader__1 ], [ %14, %exiting__1 ]
  %5 = icmp sle i64 %idxQubit, %3
  %6 = icmp sge i64 %idxQubit, %3
  %7 = select i1 %4, i1 %5, i1 %6
  br i1 %7, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %8 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %register, i64 %idxQubit)
  %9 = bitcast i8* %8 to %Qubit**
  %10 = load %Qubit*, %Qubit** %9, align 8
  %11 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Qubit* }* getelementptr ({ %Qubit* }, { %Qubit* }* null, i32 1) to i64))
  %12 = bitcast %Tuple* %11 to { %Qubit* }*
  %13 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %12, i32 0, i32 0
  store %Qubit* %10, %Qubit** %13, align 8
  call void @__quantum__rt__callable_invoke(%Callable* %singleElementOperation, %Tuple* %11, %Tuple* null)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %11, i32 -1)
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %14 = add i64 %idxQubit, %2
  br label %header__1

exit__1:                                          ; preds = %header__1
  call void @__quantum__rt__capture_update_alias_count(%Callable* %singleElementOperation, i32 -1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %singleElementOperation, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %register, i32 -1)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__H__body__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Qubit* }*
  %1 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %0, i32 0, i32 0
  %2 = load %Qubit*, %Qubit** %1, align 8
  call void @Microsoft__Quantum__Intrinsic__H__body(%Qubit* %2)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__H__adj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Qubit* }*
  %1 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %0, i32 0, i32 0
  %2 = load %Qubit*, %Qubit** %1, align 8
  call void @Microsoft__Quantum__Intrinsic__H__adj(%Qubit* %2)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__H__ctl__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Array*, %Qubit* }*
  %1 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 1
  %3 = load %Array*, %Array** %1, align 8
  %4 = load %Qubit*, %Qubit** %2, align 8
  call void @Microsoft__Quantum__Intrinsic__H__ctl(%Array* %3, %Qubit* %4)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__H__ctladj__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Array*, %Qubit* }*
  %1 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 0
  %2 = getelementptr inbounds { %Array*, %Qubit* }, { %Array*, %Qubit* }* %0, i32 0, i32 1
  %3 = load %Array*, %Array** %1, align 8
  %4 = load %Qubit*, %Qubit** %2, align 8
  call void @Microsoft__Quantum__Intrinsic__H__ctladj(%Array* %3, %Qubit* %4)
  ret void
}

define internal %Array* @Microsoft__Quantum__Arrays___a54dc56da1c94499a3e1cf538d0da3c7_Zipped__body(%Array* %left, %Array* %right) {
entry:
  %output = alloca %Array*, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %left, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %right, i32 1)
  %0 = call i64 @__quantum__rt__array_get_size_1d(%Array* %left)
  %1 = call i64 @__quantum__rt__array_get_size_1d(%Array* %right)
  %2 = icmp slt i64 %0, %1
  br i1 %2, label %condTrue__1, label %condFalse__1

condTrue__1:                                      ; preds = %entry
  br label %condContinue__1

condFalse__1:                                     ; preds = %entry
  br label %condContinue__1

condContinue__1:                                  ; preds = %condFalse__1, %condTrue__1
  %nElements = phi i64 [ %0, %condTrue__1 ], [ %1, %condFalse__1 ]
  %3 = icmp eq i64 %nElements, 0
  br i1 %3, label %then0__1, label %continue__1

then0__1:                                         ; preds = %condContinue__1
  %4 = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 0)
  call void @__quantum__rt__array_update_alias_count(%Array* %left, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %right, i32 -1)
  ret %Array* %4

continue__1:                                      ; preds = %condContinue__1
  %5 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %left, i64 0)
  %6 = bitcast i8* %5 to double*
  %7 = load double, double* %6, align 8
  %8 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %right, i64 0)
  %9 = bitcast i8* %8 to double*
  %10 = load double, double* %9, align 8
  %11 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, double }* getelementptr ({ double, double }, { double, double }* null, i32 1) to i64))
  %12 = bitcast %Tuple* %11 to { double, double }*
  %13 = getelementptr inbounds { double, double }, { double, double }* %12, i32 0, i32 0
  %14 = getelementptr inbounds { double, double }, { double, double }* %12, i32 0, i32 1
  store double %7, double* %13, align 8
  store double %10, double* %14, align 8
  %15 = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 %nElements)
  %16 = sub i64 %nElements, 1
  br label %header__1

header__1:                                        ; preds = %exiting__1, %continue__1
  %17 = phi i64 [ 0, %continue__1 ], [ %21, %exiting__1 ]
  %18 = icmp sle i64 %17, %16
  br i1 %18, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %19 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %15, i64 %17)
  %20 = bitcast i8* %19 to { double, double }**
  store { double, double }* %12, { double, double }** %20, align 8
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %11, i32 1)
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %21 = add i64 %17, 1
  br label %header__1

exit__1:                                          ; preds = %header__1
  store %Array* %15, %Array** %output, align 8
  %22 = sub i64 %nElements, 1
  br label %header__2

header__2:                                        ; preds = %exiting__2, %exit__1
  %23 = phi i64 [ 0, %exit__1 ], [ %29, %exiting__2 ]
  %24 = icmp sle i64 %23, %22
  br i1 %24, label %body__2, label %exit__2

body__2:                                          ; preds = %header__2
  %25 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %15, i64 %23)
  %26 = bitcast i8* %25 to { double, double }**
  %27 = load { double, double }*, { double, double }** %26, align 8
  %28 = bitcast { double, double }* %27 to %Tuple*
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %28, i32 1)
  br label %exiting__2

exiting__2:                                       ; preds = %body__2
  %29 = add i64 %23, 1
  br label %header__2

exit__2:                                          ; preds = %header__2
  call void @__quantum__rt__array_update_alias_count(%Array* %15, i32 1)
  %30 = sub i64 %nElements, 1
  br label %header__3

header__3:                                        ; preds = %exiting__3, %exit__2
  %idxElement = phi i64 [ 1, %exit__2 ], [ %48, %exiting__3 ]
  %31 = icmp sle i64 %idxElement, %30
  br i1 %31, label %body__3, label %exit__3

body__3:                                          ; preds = %header__3
  %32 = load %Array*, %Array** %output, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %32, i32 -1)
  %33 = call %Array* @__quantum__rt__array_copy(%Array* %32, i1 false)
  %34 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %left, i64 %idxElement)
  %35 = bitcast i8* %34 to double*
  %36 = load double, double* %35, align 8
  %37 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %right, i64 %idxElement)
  %38 = bitcast i8* %37 to double*
  %39 = load double, double* %38, align 8
  %40 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ double, double }* getelementptr ({ double, double }, { double, double }* null, i32 1) to i64))
  %41 = bitcast %Tuple* %40 to { double, double }*
  %42 = getelementptr inbounds { double, double }, { double, double }* %41, i32 0, i32 0
  %43 = getelementptr inbounds { double, double }, { double, double }* %41, i32 0, i32 1
  store double %36, double* %42, align 8
  store double %39, double* %43, align 8
  %44 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %33, i64 %idxElement)
  %45 = bitcast i8* %44 to { double, double }**
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %40, i32 1)
  %46 = load { double, double }*, { double, double }** %45, align 8
  %47 = bitcast { double, double }* %46 to %Tuple*
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %47, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %47, i32 -1)
  store { double, double }* %41, { double, double }** %45, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %33, i32 1)
  store %Array* %33, %Array** %output, align 8
  call void @__quantum__rt__array_update_reference_count(%Array* %32, i32 -1)
  br label %exiting__3

exiting__3:                                       ; preds = %body__3
  %48 = add i64 %idxElement, 1
  br label %header__3

exit__3:                                          ; preds = %header__3
  %49 = load %Array*, %Array** %output, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %left, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %right, i32 -1)
  %50 = call i64 @__quantum__rt__array_get_size_1d(%Array* %49)
  %51 = sub i64 %50, 1
  br label %header__4

header__4:                                        ; preds = %exiting__4, %exit__3
  %52 = phi i64 [ 0, %exit__3 ], [ %58, %exiting__4 ]
  %53 = icmp sle i64 %52, %51
  br i1 %53, label %body__4, label %exit__4

body__4:                                          ; preds = %header__4
  %54 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %49, i64 %52)
  %55 = bitcast i8* %54 to { double, double }**
  %56 = load { double, double }*, { double, double }** %55, align 8
  %57 = bitcast { double, double }* %56 to %Tuple*
  call void @__quantum__rt__tuple_update_alias_count(%Tuple* %57, i32 -1)
  br label %exiting__4

exiting__4:                                       ; preds = %body__4
  %58 = add i64 %52, 1
  br label %header__4

exit__4:                                          ; preds = %header__4
  call void @__quantum__rt__array_update_alias_count(%Array* %49, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %11, i32 -1)
  ret %Array* %49
}

define internal %Array* @Microsoft__Quantum__Convert__ResultArrayAsBoolArray__body(%Array* %input) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %input, i32 1)
  %0 = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @Microsoft__Quantum__Convert__ResultAsBool__FunctionTable, [2 x void (%Tuple*, i32)*]* null, %Tuple* null)
  %1 = call %Array* @Microsoft__Quantum__Arrays___daadaddd4a3b4ffaa0d2be8ed0fc58a3_Mapped__body(%Callable* %0, %Array* %input)
  call void @__quantum__rt__array_update_alias_count(%Array* %input, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %0, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %0, i32 -1)
  ret %Array* %1
}

define internal %Array* @Microsoft__Quantum__Measurement__MultiM__body(%Array* %targets) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %targets, i32 1)
  %0 = call %Callable* @__quantum__rt__callable_create([4 x void (%Tuple*, %Tuple*, %Tuple*)*]* @Microsoft__Quantum__Intrinsic__M__FunctionTable, [2 x void (%Tuple*, i32)*]* null, %Tuple* null)
  %1 = call %Array* @Microsoft__Quantum__Arrays___8d0853e3111f49a6ba2c2f7b8e24156f_ForEach__body(%Callable* %0, %Array* %targets)
  call void @__quantum__rt__array_update_alias_count(%Array* %targets, i32 -1)
  call void @__quantum__rt__capture_update_reference_count(%Callable* %0, i32 -1)
  call void @__quantum__rt__callable_update_reference_count(%Callable* %0, i32 -1)
  ret %Array* %1
}

declare void @__quantum__rt__result_update_reference_count(%Result*, i32)

define internal void @Microsoft__Quantum__Intrinsic__H__body(%Qubit* %qubit) {
entry:
  call void @__quantum__qis__h__body(%Qubit* %qubit)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__H__adj(%Qubit* %qubit) {
entry:
  call void @__quantum__qis__h__body(%Qubit* %qubit)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__H__ctl(%Array* %__controlQubits__, %Qubit* %qubit) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  call void @__quantum__qis__h__ctl(%Array* %__controlQubits__, %Qubit* %qubit)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__H__ctladj(%Array* %__controlQubits__, %Qubit* %qubit) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  call void @__quantum__qis__h__ctl(%Array* %__controlQubits__, %Qubit* %qubit)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

define internal void @Microsoft__Quantum__Samples__QAOA__RunQAOATrials__body() {
entry:
  %successNumber = alloca i64, align 8
  %bestItinerary = alloca %Array*, align 8
  %bestCost = alloca double, align 8
  %segmentCosts = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 6)
  %0 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %segmentCosts, i64 0)
  %1 = bitcast i8* %0 to double*
  %2 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %segmentCosts, i64 1)
  %3 = bitcast i8* %2 to double*
  %4 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %segmentCosts, i64 2)
  %5 = bitcast i8* %4 to double*
  %6 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %segmentCosts, i64 3)
  %7 = bitcast i8* %6 to double*
  %8 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %segmentCosts, i64 4)
  %9 = bitcast i8* %8 to double*
  %10 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %segmentCosts, i64 5)
  %11 = bitcast i8* %10 to double*
  store double 4.700000e+00, double* %1, align 8
  store double 0x40222E147AE147AE, double* %3, align 8
  store double 9.030000e+00, double* %5, align 8
  store double 5.700000e+00, double* %7, align 8
  store double 0x40200A3D70A3D70A, double* %9, align 8
  store double 1.710000e+00, double* %11, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %segmentCosts, i32 1)
  %timeX = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 5)
  %12 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %timeX, i64 0)
  %13 = bitcast i8* %12 to double*
  %14 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %timeX, i64 1)
  %15 = bitcast i8* %14 to double*
  %16 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %timeX, i64 2)
  %17 = bitcast i8* %16 to double*
  %18 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %timeX, i64 3)
  %19 = bitcast i8* %18 to double*
  %20 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %timeX, i64 4)
  %21 = bitcast i8* %20 to double*
  store double 6.191930e-01, double* %13, align 8
  store double 7.425660e-01, double* %15, align 8
  store double 6.003500e-02, double* %17, align 8
  store double 0xBFF91A708EDE54B5, double* %19, align 8
  store double 4.549000e-02, double* %21, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %timeX, i32 1)
  %timeZ = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 5)
  %22 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %timeZ, i64 0)
  %23 = bitcast i8* %22 to double*
  %24 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %timeZ, i64 1)
  %25 = bitcast i8* %24 to double*
  %26 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %timeZ, i64 2)
  %27 = bitcast i8* %26 to double*
  %28 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %timeZ, i64 3)
  %29 = bitcast i8* %28 to double*
  %30 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %timeZ, i64 4)
  %31 = bitcast i8* %30 to double*
  store double 0x40097526D8B1DD5D, double* %23, align 8
  store double 0xBFF239873FFAC1D3, double* %25, align 8
  store double 2.210820e-01, double* %27, align 8
  store double 5.377530e-01, double* %29, align 8
  store double -4.172220e-01, double* %31, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %timeZ, i32 1)
  store double 2.000000e+03, double* %bestCost, align 8
  %32 = call %Array* @__quantum__rt__array_create_1d(i32 1, i64 5)
  %33 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %32, i64 0)
  %34 = bitcast i8* %33 to i1*
  %35 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %32, i64 1)
  %36 = bitcast i8* %35 to i1*
  %37 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %32, i64 2)
  %38 = bitcast i8* %37 to i1*
  %39 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %32, i64 3)
  %40 = bitcast i8* %39 to i1*
  %41 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %32, i64 4)
  %42 = bitcast i8* %41 to i1*
  store i1 false, i1* %34, align 1
  store i1 false, i1* %36, align 1
  store i1 false, i1* %38, align 1
  store i1 false, i1* %40, align 1
  store i1 false, i1* %42, align 1
  store %Array* %32, %Array** %bestItinerary, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %32, i32 1)
  store i64 0, i64* %successNumber, align 4
  %weights = call %Array* @Microsoft__Quantum__Samples__QAOA__HamiltonianWeights__body(%Array* %segmentCosts, double 2.000000e+01, i64 6)
  call void @__quantum__rt__array_update_alias_count(%Array* %weights, i32 1)
  %couplings = call %Array* @Microsoft__Quantum__Samples__QAOA__HamiltonianCouplings__body(double 2.000000e+01, i64 6)
  call void @__quantum__rt__array_update_alias_count(%Array* %couplings, i32 1)
  br label %header__1

header__1:                                        ; preds = %exiting__1, %entry
  %trial = phi i64 [ 1, %entry ], [ %49, %exiting__1 ]
  %43 = icmp sle i64 %trial, 20
  br i1 %43, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %result = call %Array* @Microsoft__Quantum__Samples__QAOA__PerformQAOA__body(i64 6, %Array* %weights, %Array* %couplings, %Array* %timeX, %Array* %timeZ)
  call void @__quantum__rt__array_update_alias_count(%Array* %result, i32 1)
  %cost = call double @Microsoft__Quantum__Samples__QAOA__CalculatedCost__body(%Array* %segmentCosts, %Array* %result)
  %sat = call i1 @Microsoft__Quantum__Samples__QAOA__IsSatisfactory__body(i64 6, %Array* %result)
  %44 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @3, i32 0, i32 0))
  %45 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @4, i32 0, i32 0))
  %46 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @5, i32 0, i32 0))
  call void @__quantum__rt__string_update_reference_count(%String* %46, i32 1)
  %47 = call i64 @__quantum__rt__array_get_size_1d(%Array* %result)
  %48 = sub i64 %47, 1
  br label %header__2

exiting__1:                                       ; preds = %continue__1
  %49 = add i64 %trial, 1
  br label %header__1

exit__1:                                          ; preds = %header__1
  %50 = load i64, i64* %successNumber, align 4
  %51 = sitofp i64 %50 to double
  %52 = fmul double %51, 1.000000e+02
  %runPercentage = fdiv double %52, 2.000000e+01
  %53 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([24 x i8], [24 x i8]* @11, i32 0, i32 0))
  call void @__quantum__rt__message(%String* %53)
  %54 = load %Array*, %Array** %bestItinerary, align 8
  %55 = load double, double* %bestCost, align 8
  %56 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([23 x i8], [23 x i8]* @12, i32 0, i32 0))
  %57 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @4, i32 0, i32 0))
  %58 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @5, i32 0, i32 0))
  call void @__quantum__rt__string_update_reference_count(%String* %58, i32 1)
  %59 = call i64 @__quantum__rt__array_get_size_1d(%Array* %54)
  %60 = sub i64 %59, 1
  br label %header__3

header__2:                                        ; preds = %exiting__2, %body__1
  %61 = phi %String* [ %46, %body__1 ], [ %73, %exiting__2 ]
  %62 = phi i64 [ 0, %body__1 ], [ %74, %exiting__2 ]
  %63 = icmp sle i64 %62, %48
  br i1 %63, label %body__2, label %exit__2

body__2:                                          ; preds = %header__2
  %64 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %result, i64 %62)
  %65 = bitcast i8* %64 to i1*
  %66 = load i1, i1* %65, align 1
  %67 = icmp ne %String* %61, %46
  br i1 %67, label %condTrue__1, label %condContinue__1

condTrue__1:                                      ; preds = %body__2
  %68 = call %String* @__quantum__rt__string_concatenate(%String* %61, %String* %45)
  call void @__quantum__rt__string_update_reference_count(%String* %61, i32 -1)
  br label %condContinue__1

condContinue__1:                                  ; preds = %condTrue__1, %body__2
  %69 = phi %String* [ %68, %condTrue__1 ], [ %61, %body__2 ]
  br i1 %66, label %condTrue__2, label %condFalse__1

condTrue__2:                                      ; preds = %condContinue__1
  %70 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @6, i32 0, i32 0))
  br label %condContinue__2

condFalse__1:                                     ; preds = %condContinue__1
  %71 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @7, i32 0, i32 0))
  br label %condContinue__2

condContinue__2:                                  ; preds = %condFalse__1, %condTrue__2
  %72 = phi %String* [ %70, %condTrue__2 ], [ %71, %condFalse__1 ]
  %73 = call %String* @__quantum__rt__string_concatenate(%String* %69, %String* %72)
  call void @__quantum__rt__string_update_reference_count(%String* %69, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %72, i32 -1)
  br label %exiting__2

exiting__2:                                       ; preds = %condContinue__2
  %74 = add i64 %62, 1
  br label %header__2

exit__2:                                          ; preds = %header__2
  %75 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @8, i32 0, i32 0))
  %76 = call %String* @__quantum__rt__string_concatenate(%String* %61, %String* %75)
  call void @__quantum__rt__string_update_reference_count(%String* %61, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %75, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %45, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %46, i32 -1)
  %77 = call %String* @__quantum__rt__string_concatenate(%String* %44, %String* %76)
  call void @__quantum__rt__string_update_reference_count(%String* %44, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %76, i32 -1)
  %78 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @9, i32 0, i32 0))
  %79 = call %String* @__quantum__rt__string_concatenate(%String* %77, %String* %78)
  call void @__quantum__rt__string_update_reference_count(%String* %77, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %78, i32 -1)
  %80 = call %String* @__quantum__rt__double_to_string(double %cost)
  %81 = call %String* @__quantum__rt__string_concatenate(%String* %79, %String* %80)
  call void @__quantum__rt__string_update_reference_count(%String* %79, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %80, i32 -1)
  %82 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([18 x i8], [18 x i8]* @10, i32 0, i32 0))
  %83 = call %String* @__quantum__rt__string_concatenate(%String* %81, %String* %82)
  call void @__quantum__rt__string_update_reference_count(%String* %81, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %82, i32 -1)
  br i1 %sat, label %condTrue__3, label %condFalse__2

condTrue__3:                                      ; preds = %exit__2
  %84 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @6, i32 0, i32 0))
  br label %condContinue__3

condFalse__2:                                     ; preds = %exit__2
  %85 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @7, i32 0, i32 0))
  br label %condContinue__3

condContinue__3:                                  ; preds = %condFalse__2, %condTrue__3
  %86 = phi %String* [ %84, %condTrue__3 ], [ %85, %condFalse__2 ]
  %87 = call %String* @__quantum__rt__string_concatenate(%String* %83, %String* %86)
  call void @__quantum__rt__string_update_reference_count(%String* %83, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %86, i32 -1)
  call void @__quantum__rt__message(%String* %87)
  br i1 %sat, label %then0__1, label %continue__1

then0__1:                                         ; preds = %condContinue__3
  %88 = load double, double* %bestCost, align 8
  %89 = fsub double %88, 0x3EB0C6F7A0B5ED8D
  %90 = fcmp olt double %cost, %89
  br i1 %90, label %then0__2, label %test1__1

then0__2:                                         ; preds = %then0__1
  store double %cost, double* %bestCost, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %result, i32 1)
  call void @__quantum__rt__array_update_reference_count(%Array* %result, i32 1)
  %91 = load %Array*, %Array** %bestItinerary, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %91, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %91, i32 -1)
  store %Array* %result, %Array** %bestItinerary, align 8
  store i64 1, i64* %successNumber, align 4
  br label %continue__2

test1__1:                                         ; preds = %then0__1
  %92 = load double, double* %bestCost, align 8
  %93 = fsub double %cost, %92
  %94 = call double @Microsoft__Quantum__Math__AbsD__body(double %93)
  %95 = fcmp olt double %94, 0x3EB0C6F7A0B5ED8D
  br i1 %95, label %then1__1, label %continue__2

then1__1:                                         ; preds = %test1__1
  %96 = load i64, i64* %successNumber, align 4
  %97 = add i64 %96, 1
  store i64 %97, i64* %successNumber, align 4
  br label %continue__2

continue__2:                                      ; preds = %then1__1, %test1__1, %then0__2
  br label %continue__1

continue__1:                                      ; preds = %continue__2, %condContinue__3
  call void @__quantum__rt__array_update_alias_count(%Array* %result, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %result, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %87, i32 -1)
  br label %exiting__1

header__3:                                        ; preds = %exiting__3, %exit__1
  %98 = phi %String* [ %58, %exit__1 ], [ %110, %exiting__3 ]
  %99 = phi i64 [ 0, %exit__1 ], [ %111, %exiting__3 ]
  %100 = icmp sle i64 %99, %60
  br i1 %100, label %body__3, label %exit__3

body__3:                                          ; preds = %header__3
  %101 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %54, i64 %99)
  %102 = bitcast i8* %101 to i1*
  %103 = load i1, i1* %102, align 1
  %104 = icmp ne %String* %98, %58
  br i1 %104, label %condTrue__4, label %condContinue__4

condTrue__4:                                      ; preds = %body__3
  %105 = call %String* @__quantum__rt__string_concatenate(%String* %98, %String* %57)
  call void @__quantum__rt__string_update_reference_count(%String* %98, i32 -1)
  br label %condContinue__4

condContinue__4:                                  ; preds = %condTrue__4, %body__3
  %106 = phi %String* [ %105, %condTrue__4 ], [ %98, %body__3 ]
  br i1 %103, label %condTrue__5, label %condFalse__3

condTrue__5:                                      ; preds = %condContinue__4
  %107 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([5 x i8], [5 x i8]* @6, i32 0, i32 0))
  br label %condContinue__5

condFalse__3:                                     ; preds = %condContinue__4
  %108 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([6 x i8], [6 x i8]* @7, i32 0, i32 0))
  br label %condContinue__5

condContinue__5:                                  ; preds = %condFalse__3, %condTrue__5
  %109 = phi %String* [ %107, %condTrue__5 ], [ %108, %condFalse__3 ]
  %110 = call %String* @__quantum__rt__string_concatenate(%String* %106, %String* %109)
  call void @__quantum__rt__string_update_reference_count(%String* %106, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %109, i32 -1)
  br label %exiting__3

exiting__3:                                       ; preds = %condContinue__5
  %111 = add i64 %99, 1
  br label %header__3

exit__3:                                          ; preds = %header__3
  %112 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @8, i32 0, i32 0))
  %113 = call %String* @__quantum__rt__string_concatenate(%String* %98, %String* %112)
  call void @__quantum__rt__string_update_reference_count(%String* %98, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %112, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %57, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %58, i32 -1)
  %114 = call %String* @__quantum__rt__string_concatenate(%String* %56, %String* %113)
  call void @__quantum__rt__string_update_reference_count(%String* %56, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %113, i32 -1)
  %115 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([10 x i8], [10 x i8]* @9, i32 0, i32 0))
  %116 = call %String* @__quantum__rt__string_concatenate(%String* %114, %String* %115)
  call void @__quantum__rt__string_update_reference_count(%String* %114, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %115, i32 -1)
  %117 = call %String* @__quantum__rt__double_to_string(double %55)
  %118 = call %String* @__quantum__rt__string_concatenate(%String* %116, %String* %117)
  call void @__quantum__rt__string_update_reference_count(%String* %116, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %117, i32 -1)
  call void @__quantum__rt__message(%String* %118)
  %119 = call %String* @__quantum__rt__double_to_string(double %runPercentage)
  %120 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([36 x i8], [36 x i8]* @13, i32 0, i32 0))
  %121 = call %String* @__quantum__rt__string_concatenate(%String* %119, %String* %120)
  call void @__quantum__rt__string_update_reference_count(%String* %119, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %120, i32 -1)
  call void @__quantum__rt__message(%String* %121)
  call void @__quantum__rt__array_update_alias_count(%Array* %segmentCosts, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %timeX, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %timeZ, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %54, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %weights, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %couplings, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %segmentCosts, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %timeX, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %timeZ, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %weights, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %couplings, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %53, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %118, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %121, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %54, i32 -1)
  ret void
}

declare %String* @__quantum__rt__string_concatenate(%String*, %String*)

declare %String* @__quantum__rt__double_to_string(double)

declare void @__quantum__rt__message(%String*)

define internal double @Microsoft__Quantum__Math__AbsD__body(double %a) {
entry:
  %0 = fcmp olt double %a, 0.000000e+00
  br i1 %0, label %condTrue__1, label %condFalse__1

condTrue__1:                                      ; preds = %entry
  %1 = fneg double %a
  br label %condContinue__1

condFalse__1:                                     ; preds = %entry
  br label %condContinue__1

condContinue__1:                                  ; preds = %condFalse__1, %condTrue__1
  %2 = phi double [ %1, %condTrue__1 ], [ %a, %condFalse__1 ]
  ret double %2
}

define internal void @Microsoft__Quantum__Diagnostics___4a93b78d84c84bab83c4c604247a1c64___QsRef1__FormattedFailure____body(i64 %actual, i64 %expected, %String* %message) {
entry:
  %0 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @14, i32 0, i32 0))
  %1 = call %String* @__quantum__rt__string_concatenate(%String* %0, %String* %message)
  %2 = call %String* @__quantum__rt__string_concatenate(%String* %1, %String* %0)
  call void @__quantum__rt__string_update_reference_count(%String* %1, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %0, i32 -1)
  %3 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @15, i32 0, i32 0))
  %4 = call %String* @__quantum__rt__string_concatenate(%String* %2, %String* %3)
  call void @__quantum__rt__string_update_reference_count(%String* %2, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %3, i32 -1)
  %5 = call %String* @__quantum__rt__int_to_string(i64 %expected)
  %6 = call %String* @__quantum__rt__string_concatenate(%String* %4, %String* %5)
  call void @__quantum__rt__string_update_reference_count(%String* %4, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %5, i32 -1)
  %7 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([11 x i8], [11 x i8]* @16, i32 0, i32 0))
  %8 = call %String* @__quantum__rt__string_concatenate(%String* %6, %String* %7)
  call void @__quantum__rt__string_update_reference_count(%String* %6, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %7, i32 -1)
  %9 = call %String* @__quantum__rt__int_to_string(i64 %actual)
  %10 = call %String* @__quantum__rt__string_concatenate(%String* %8, %String* %9)
  call void @__quantum__rt__string_update_reference_count(%String* %8, i32 -1)
  call void @__quantum__rt__string_update_reference_count(%String* %9, i32 -1)
  call void @__quantum__rt__fail(%String* %10)
  unreachable
}

declare %String* @__quantum__rt__int_to_string(i64)

declare void @__quantum__rt__fail(%String*)

define internal %Array* @Microsoft__Quantum__Arrays___daadaddd4a3b4ffaa0d2be8ed0fc58a3_Mapped__body(%Callable* %mapper, %Array* %array) {
entry:
  %retval = alloca %Array*, align 8
  call void @__quantum__rt__capture_update_alias_count(%Callable* %mapper, i32 1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %mapper, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 1)
  %length = call i64 @__quantum__rt__array_get_size_1d(%Array* %array)
  %0 = icmp eq i64 %length, 0
  br i1 %0, label %then0__1, label %continue__1

then0__1:                                         ; preds = %entry
  %1 = call %Array* @__quantum__rt__array_create_1d(i32 1, i64 0)
  call void @__quantum__rt__capture_update_alias_count(%Callable* %mapper, i32 -1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %mapper, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 -1)
  ret %Array* %1

continue__1:                                      ; preds = %entry
  %2 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 0)
  %3 = bitcast i8* %2 to %Result**
  %4 = load %Result*, %Result** %3, align 8
  %5 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Result* }* getelementptr ({ %Result* }, { %Result* }* null, i32 1) to i64))
  %6 = bitcast %Tuple* %5 to { %Result* }*
  %7 = getelementptr inbounds { %Result* }, { %Result* }* %6, i32 0, i32 0
  store %Result* %4, %Result** %7, align 8
  %8 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ i1 }* getelementptr ({ i1 }, { i1 }* null, i32 1) to i64))
  call void @__quantum__rt__callable_invoke(%Callable* %mapper, %Tuple* %5, %Tuple* %8)
  %9 = bitcast %Tuple* %8 to { i1 }*
  %10 = getelementptr inbounds { i1 }, { i1 }* %9, i32 0, i32 0
  %first = load i1, i1* %10, align 1
  %11 = call %Array* @__quantum__rt__array_create_1d(i32 1, i64 %length)
  %12 = sub i64 %length, 1
  br label %header__1

header__1:                                        ; preds = %exiting__1, %continue__1
  %13 = phi i64 [ 0, %continue__1 ], [ %17, %exiting__1 ]
  %14 = icmp sle i64 %13, %12
  br i1 %14, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %15 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %11, i64 %13)
  %16 = bitcast i8* %15 to i1*
  store i1 %first, i1* %16, align 1
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %17 = add i64 %13, 1
  br label %header__1

exit__1:                                          ; preds = %header__1
  store %Array* %11, %Array** %retval, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %11, i32 1)
  %18 = sub i64 %length, 1
  br label %header__2

header__2:                                        ; preds = %exiting__2, %exit__1
  %idx = phi i64 [ 1, %exit__1 ], [ %35, %exiting__2 ]
  %19 = icmp sle i64 %idx, %18
  br i1 %19, label %body__2, label %exit__2

body__2:                                          ; preds = %header__2
  %20 = load %Array*, %Array** %retval, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %20, i32 -1)
  %21 = call %Array* @__quantum__rt__array_copy(%Array* %20, i1 false)
  %22 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 %idx)
  %23 = bitcast i8* %22 to %Result**
  %24 = load %Result*, %Result** %23, align 8
  %25 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Result* }* getelementptr ({ %Result* }, { %Result* }* null, i32 1) to i64))
  %26 = bitcast %Tuple* %25 to { %Result* }*
  %27 = getelementptr inbounds { %Result* }, { %Result* }* %26, i32 0, i32 0
  store %Result* %24, %Result** %27, align 8
  %28 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ i1 }* getelementptr ({ i1 }, { i1 }* null, i32 1) to i64))
  call void @__quantum__rt__callable_invoke(%Callable* %mapper, %Tuple* %25, %Tuple* %28)
  %29 = bitcast %Tuple* %28 to { i1 }*
  %30 = getelementptr inbounds { i1 }, { i1 }* %29, i32 0, i32 0
  %31 = load i1, i1* %30, align 1
  %32 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %21, i64 %idx)
  %33 = bitcast i8* %32 to i1*
  %34 = load i1, i1* %33, align 1
  store i1 %31, i1* %33, align 1
  call void @__quantum__rt__array_update_alias_count(%Array* %21, i32 1)
  store %Array* %21, %Array** %retval, align 8
  call void @__quantum__rt__array_update_reference_count(%Array* %20, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %25, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %28, i32 -1)
  br label %exiting__2

exiting__2:                                       ; preds = %body__2
  %35 = add i64 %idx, 1
  br label %header__2

exit__2:                                          ; preds = %header__2
  %36 = load %Array*, %Array** %retval, align 8
  call void @__quantum__rt__capture_update_alias_count(%Callable* %mapper, i32 -1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %mapper, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %36, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %5, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %8, i32 -1)
  ret %Array* %36
}

define internal void @Microsoft__Quantum__Convert__ResultAsBool__body__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Result* }*
  %1 = getelementptr inbounds { %Result* }, { %Result* }* %0, i32 0, i32 0
  %2 = load %Result*, %Result** %1, align 8
  %3 = call i1 @Microsoft__Quantum__Convert__ResultAsBool__body(%Result* %2)
  %4 = bitcast %Tuple* %result-tuple to { i1 }*
  %5 = getelementptr inbounds { i1 }, { i1 }* %4, i32 0, i32 0
  store i1 %3, i1* %5, align 1
  ret void
}

define internal i1 @Microsoft__Quantum__Convert__ResultAsBool__body(%Result* %input) {
entry:
  %0 = call %Result* @__quantum__rt__result_get_zero()
  %1 = call i1 @__quantum__rt__result_equal(%Result* %input, %Result* %0)
  %2 = select i1 %1, i1 false, i1 true
  ret i1 %2
}

declare %Result* @__quantum__rt__result_get_zero()

declare i1 @__quantum__rt__result_equal(%Result*, %Result*)

declare void @__quantum__qis__x__ctl(%Array*, %Qubit*)

define internal void @Microsoft__Quantum__Intrinsic__CNOT__ctl(%Array* %__controlQubits__, { %Qubit*, %Qubit* }* %0) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %1 = getelementptr inbounds { %Qubit*, %Qubit* }, { %Qubit*, %Qubit* }* %0, i32 0, i32 0
  %control = load %Qubit*, %Qubit** %1, align 8
  %2 = getelementptr inbounds { %Qubit*, %Qubit* }, { %Qubit*, %Qubit* }* %0, i32 0, i32 1
  %target = load %Qubit*, %Qubit** %2, align 8
  %3 = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 1)
  %4 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %3, i64 0)
  %5 = bitcast i8* %4 to %Qubit**
  store %Qubit* %control, %Qubit** %5, align 8
  %__controlQubits__1 = call %Array* @__quantum__rt__array_concatenate(%Array* %__controlQubits__, %Array* %3)
  call void @__quantum__rt__array_update_reference_count(%Array* %__controlQubits__1, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__1, i32 1)
  call void @__quantum__qis__x__ctl(%Array* %__controlQubits__1, %Qubit* %target)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__1, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %3, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %__controlQubits__1, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %__controlQubits__1, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

declare %Array* @__quantum__rt__array_concatenate(%Array*, %Array*)

define internal void @Microsoft__Quantum__Intrinsic__CNOT__ctladj(%Array* %__controlQubits__, { %Qubit*, %Qubit* }* %0) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %1 = getelementptr inbounds { %Qubit*, %Qubit* }, { %Qubit*, %Qubit* }* %0, i32 0, i32 0
  %control = load %Qubit*, %Qubit** %1, align 8
  %2 = getelementptr inbounds { %Qubit*, %Qubit* }, { %Qubit*, %Qubit* }* %0, i32 0, i32 1
  %target = load %Qubit*, %Qubit** %2, align 8
  %3 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Qubit*, %Qubit* }* getelementptr ({ %Qubit*, %Qubit* }, { %Qubit*, %Qubit* }* null, i32 1) to i64))
  %4 = bitcast %Tuple* %3 to { %Qubit*, %Qubit* }*
  %5 = getelementptr inbounds { %Qubit*, %Qubit* }, { %Qubit*, %Qubit* }* %4, i32 0, i32 0
  %6 = getelementptr inbounds { %Qubit*, %Qubit* }, { %Qubit*, %Qubit* }* %4, i32 0, i32 1
  store %Qubit* %control, %Qubit** %5, align 8
  store %Qubit* %target, %Qubit** %6, align 8
  call void @Microsoft__Quantum__Intrinsic__CNOT__ctl(%Array* %__controlQubits__, { %Qubit*, %Qubit* }* %4)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %3, i32 -1)
  ret void
}

declare void @__quantum__qis__h__body(%Qubit*)

declare void @__quantum__qis__h__ctl(%Array*, %Qubit*)

define internal %Result* @Microsoft__Quantum__Intrinsic__M__body(%Qubit* %qubit) {
entry:
  %bases = call %Array* @__quantum__rt__array_create_1d(i32 1, i64 1)
  %0 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %bases, i64 0)
  %1 = bitcast i8* %0 to i2*
  store i2 -2, i2* %1, align 1
  call void @__quantum__rt__array_update_alias_count(%Array* %bases, i32 1)
  %qubits = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 1)
  %2 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %qubits, i64 0)
  %3 = bitcast i8* %2 to %Qubit**
  store %Qubit* %qubit, %Qubit** %3, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %qubits, i32 1)
  %4 = call %Result* @__quantum__qis__measure__body(%Array* %bases, %Array* %qubits)
  call void @__quantum__rt__array_update_alias_count(%Array* %bases, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %qubits, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %bases, i32 -1)
  call void @__quantum__rt__array_update_reference_count(%Array* %qubits, i32 -1)
  ret %Result* %4
}

declare %Result* @__quantum__qis__measure__body(%Array*, %Array*)

define internal %Result* @Microsoft__Quantum__Intrinsic__Measure__body(%Array* %bases, %Array* %qubits) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %bases, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %qubits, i32 1)
  %0 = call %Result* @__quantum__qis__measure__body(%Array* %bases, %Array* %qubits)
  call void @__quantum__rt__array_update_alias_count(%Array* %bases, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %qubits, i32 -1)
  ret %Result* %0
}

define internal void @Microsoft__Quantum__Intrinsic__R__body(i2 %pauli, double %theta, %Qubit* %qubit) {
entry:
  call void @__quantum__qis__r__body(i2 %pauli, double %theta, %Qubit* %qubit)
  ret void
}

declare void @__quantum__qis__r__body(i2, double, %Qubit*)

define internal void @Microsoft__Quantum__Intrinsic__R__adj(i2 %pauli, double %theta, %Qubit* %qubit) {
entry:
  call void @__quantum__qis__r__adj(i2 %pauli, double %theta, %Qubit* %qubit)
  ret void
}

declare void @__quantum__qis__r__adj(i2, double, %Qubit*)

define internal void @Microsoft__Quantum__Intrinsic__R__ctl(%Array* %__controlQubits__, { i2, double, %Qubit* }* %0) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %1 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %0, i32 0, i32 0
  %pauli = load i2, i2* %1, align 1
  %2 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %0, i32 0, i32 1
  %theta = load double, double* %2, align 8
  %3 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %0, i32 0, i32 2
  %qubit = load %Qubit*, %Qubit** %3, align 8
  %4 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ i2, double, %Qubit* }* getelementptr ({ i2, double, %Qubit* }, { i2, double, %Qubit* }* null, i32 1) to i64))
  %5 = bitcast %Tuple* %4 to { i2, double, %Qubit* }*
  %6 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %5, i32 0, i32 0
  %7 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %5, i32 0, i32 1
  %8 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %5, i32 0, i32 2
  store i2 %pauli, i2* %6, align 1
  store double %theta, double* %7, align 8
  store %Qubit* %qubit, %Qubit** %8, align 8
  call void @__quantum__qis__r__ctl(%Array* %__controlQubits__, { i2, double, %Qubit* }* %5)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %4, i32 -1)
  ret void
}

declare void @__quantum__qis__r__ctl(%Array*, { i2, double, %Qubit* }*)

define internal void @Microsoft__Quantum__Intrinsic__R__ctladj(%Array* %__controlQubits__, { i2, double, %Qubit* }* %0) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %1 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %0, i32 0, i32 0
  %pauli = load i2, i2* %1, align 1
  %2 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %0, i32 0, i32 1
  %theta = load double, double* %2, align 8
  %3 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %0, i32 0, i32 2
  %qubit = load %Qubit*, %Qubit** %3, align 8
  %4 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ i2, double, %Qubit* }* getelementptr ({ i2, double, %Qubit* }, { i2, double, %Qubit* }* null, i32 1) to i64))
  %5 = bitcast %Tuple* %4 to { i2, double, %Qubit* }*
  %6 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %5, i32 0, i32 0
  %7 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %5, i32 0, i32 1
  %8 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %5, i32 0, i32 2
  store i2 %pauli, i2* %6, align 1
  store double %theta, double* %7, align 8
  store %Qubit* %qubit, %Qubit** %8, align 8
  call void @__quantum__qis__r__ctladj(%Array* %__controlQubits__, { i2, double, %Qubit* }* %5)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %4, i32 -1)
  ret void
}

declare void @__quantum__qis__r__ctladj(%Array*, { i2, double, %Qubit* }*)

define internal void @Microsoft__Quantum__Intrinsic__Rz__adj(double %theta, %Qubit* %qubit) {
entry:
  %theta__1 = fneg double %theta
  call void @__quantum__qis__r__body(i2 -2, double %theta__1, %Qubit* %qubit)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__Rz__ctl(%Array* %__controlQubits__, { double, %Qubit* }* %0) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %1 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %0, i32 0, i32 0
  %theta = load double, double* %1, align 8
  %2 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %0, i32 0, i32 1
  %qubit = load %Qubit*, %Qubit** %2, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %3 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ i2, double, %Qubit* }* getelementptr ({ i2, double, %Qubit* }, { i2, double, %Qubit* }* null, i32 1) to i64))
  %4 = bitcast %Tuple* %3 to { i2, double, %Qubit* }*
  %5 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %4, i32 0, i32 0
  %6 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %4, i32 0, i32 1
  %7 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %4, i32 0, i32 2
  store i2 -2, i2* %5, align 1
  store double %theta, double* %6, align 8
  store %Qubit* %qubit, %Qubit** %7, align 8
  call void @__quantum__qis__r__ctl(%Array* %__controlQubits__, { i2, double, %Qubit* }* %4)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %3, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__Rz__ctladj(%Array* %__controlQubits__, { double, %Qubit* }* %0) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %1 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %0, i32 0, i32 0
  %theta = load double, double* %1, align 8
  %2 = getelementptr inbounds { double, %Qubit* }, { double, %Qubit* }* %0, i32 0, i32 1
  %qubit = load %Qubit*, %Qubit** %2, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  %theta__1 = fneg double %theta
  %3 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ i2, double, %Qubit* }* getelementptr ({ i2, double, %Qubit* }, { i2, double, %Qubit* }* null, i32 1) to i64))
  %4 = bitcast %Tuple* %3 to { i2, double, %Qubit* }*
  %5 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %4, i32 0, i32 0
  %6 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %4, i32 0, i32 1
  %7 = getelementptr inbounds { i2, double, %Qubit* }, { i2, double, %Qubit* }* %4, i32 0, i32 2
  store i2 -2, i2* %5, align 1
  store double %theta__1, double* %6, align 8
  store %Qubit* %qubit, %Qubit** %7, align 8
  call void @__quantum__qis__r__ctl(%Array* %__controlQubits__, { i2, double, %Qubit* }* %4)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %3, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__X__body(%Qubit* %qubit) {
entry:
  call void @__quantum__qis__x__body(%Qubit* %qubit)
  ret void
}

declare void @__quantum__qis__x__body(%Qubit*)

define internal void @Microsoft__Quantum__Intrinsic__X__adj(%Qubit* %qubit) {
entry:
  call void @__quantum__qis__x__body(%Qubit* %qubit)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__X__ctl(%Array* %__controlQubits__, %Qubit* %qubit) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  call void @__quantum__qis__x__ctl(%Array* %__controlQubits__, %Qubit* %qubit)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

define internal void @Microsoft__Quantum__Intrinsic__X__ctladj(%Array* %__controlQubits__, %Qubit* %qubit) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 1)
  call void @__quantum__qis__x__ctl(%Array* %__controlQubits__, %Qubit* %qubit)
  call void @__quantum__rt__array_update_alias_count(%Array* %__controlQubits__, i32 -1)
  ret void
}

define internal %Array* @Microsoft__Quantum__Arrays___8d0853e3111f49a6ba2c2f7b8e24156f_ForEach__body(%Callable* %action, %Array* %array) {
entry:
  %retval = alloca %Array*, align 8
  call void @__quantum__rt__capture_update_alias_count(%Callable* %action, i32 1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %action, i32 1)
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 1)
  %length = call i64 @__quantum__rt__array_get_size_1d(%Array* %array)
  %0 = icmp eq i64 %length, 0
  br i1 %0, label %then0__1, label %continue__1

then0__1:                                         ; preds = %entry
  %1 = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 0)
  call void @__quantum__rt__capture_update_alias_count(%Callable* %action, i32 -1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %action, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 -1)
  ret %Array* %1

continue__1:                                      ; preds = %entry
  %2 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 0)
  %3 = bitcast i8* %2 to %Qubit**
  %4 = load %Qubit*, %Qubit** %3, align 8
  %5 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Qubit* }* getelementptr ({ %Qubit* }, { %Qubit* }* null, i32 1) to i64))
  %6 = bitcast %Tuple* %5 to { %Qubit* }*
  %7 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %6, i32 0, i32 0
  store %Qubit* %4, %Qubit** %7, align 8
  %8 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Result* }* getelementptr ({ %Result* }, { %Result* }* null, i32 1) to i64))
  call void @__quantum__rt__callable_invoke(%Callable* %action, %Tuple* %5, %Tuple* %8)
  %9 = bitcast %Tuple* %8 to { %Result* }*
  %10 = getelementptr inbounds { %Result* }, { %Result* }* %9, i32 0, i32 0
  %first = load %Result*, %Result** %10, align 8
  %11 = call %Array* @__quantum__rt__array_create_1d(i32 8, i64 %length)
  %12 = sub i64 %length, 1
  br label %header__1

header__1:                                        ; preds = %exiting__1, %continue__1
  %13 = phi i64 [ 0, %continue__1 ], [ %17, %exiting__1 ]
  %14 = icmp sle i64 %13, %12
  br i1 %14, label %body__1, label %exit__1

body__1:                                          ; preds = %header__1
  %15 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %11, i64 %13)
  %16 = bitcast i8* %15 to %Result**
  store %Result* %first, %Result** %16, align 8
  call void @__quantum__rt__result_update_reference_count(%Result* %first, i32 1)
  br label %exiting__1

exiting__1:                                       ; preds = %body__1
  %17 = add i64 %13, 1
  br label %header__1

exit__1:                                          ; preds = %header__1
  store %Array* %11, %Array** %retval, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %11, i32 1)
  %18 = sub i64 %length, 1
  br label %header__2

header__2:                                        ; preds = %exiting__2, %exit__1
  %idx = phi i64 [ 1, %exit__1 ], [ %35, %exiting__2 ]
  %19 = icmp sle i64 %idx, %18
  br i1 %19, label %body__2, label %exit__2

body__2:                                          ; preds = %header__2
  %20 = load %Array*, %Array** %retval, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %20, i32 -1)
  %21 = call %Array* @__quantum__rt__array_copy(%Array* %20, i1 false)
  %22 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %array, i64 %idx)
  %23 = bitcast i8* %22 to %Qubit**
  %24 = load %Qubit*, %Qubit** %23, align 8
  %25 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Qubit* }* getelementptr ({ %Qubit* }, { %Qubit* }* null, i32 1) to i64))
  %26 = bitcast %Tuple* %25 to { %Qubit* }*
  %27 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %26, i32 0, i32 0
  store %Qubit* %24, %Qubit** %27, align 8
  %28 = call %Tuple* @__quantum__rt__tuple_create(i64 ptrtoint ({ %Result* }* getelementptr ({ %Result* }, { %Result* }* null, i32 1) to i64))
  call void @__quantum__rt__callable_invoke(%Callable* %action, %Tuple* %25, %Tuple* %28)
  %29 = bitcast %Tuple* %28 to { %Result* }*
  %30 = getelementptr inbounds { %Result* }, { %Result* }* %29, i32 0, i32 0
  %31 = load %Result*, %Result** %30, align 8
  %32 = call i8* @__quantum__rt__array_get_element_ptr_1d(%Array* %21, i64 %idx)
  %33 = bitcast i8* %32 to %Result**
  call void @__quantum__rt__result_update_reference_count(%Result* %31, i32 1)
  %34 = load %Result*, %Result** %33, align 8
  call void @__quantum__rt__result_update_reference_count(%Result* %34, i32 -1)
  store %Result* %31, %Result** %33, align 8
  call void @__quantum__rt__array_update_alias_count(%Array* %21, i32 1)
  store %Array* %21, %Array** %retval, align 8
  call void @__quantum__rt__array_update_reference_count(%Array* %20, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %25, i32 -1)
  call void @__quantum__rt__result_update_reference_count(%Result* %31, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %28, i32 -1)
  br label %exiting__2

exiting__2:                                       ; preds = %body__2
  %35 = add i64 %idx, 1
  br label %header__2

exit__2:                                          ; preds = %header__2
  %36 = load %Array*, %Array** %retval, align 8
  call void @__quantum__rt__capture_update_alias_count(%Callable* %action, i32 -1)
  call void @__quantum__rt__callable_update_alias_count(%Callable* %action, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 -1)
  call void @__quantum__rt__array_update_alias_count(%Array* %36, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %5, i32 -1)
  call void @__quantum__rt__result_update_reference_count(%Result* %first, i32 -1)
  call void @__quantum__rt__tuple_update_reference_count(%Tuple* %8, i32 -1)
  ret %Array* %36
}

define internal %Range @Microsoft__Quantum__Arrays___aff6ba86ed7b447f90c2e700aa07a9b4_IndexRange__body(%Array* %array) {
entry:
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 1)
  %0 = call i64 @__quantum__rt__array_get_size_1d(%Array* %array)
  %1 = sub i64 %0, 1
  %2 = insertvalue %Range { i64 0, i64 1, i64 0 }, i64 %1, 2
  call void @__quantum__rt__array_update_alias_count(%Array* %array, i32 -1)
  ret %Range %2
}

define internal void @Microsoft__Quantum__Intrinsic__M__body__wrapper(%Tuple* %capture-tuple, %Tuple* %arg-tuple, %Tuple* %result-tuple) {
entry:
  %0 = bitcast %Tuple* %arg-tuple to { %Qubit* }*
  %1 = getelementptr inbounds { %Qubit* }, { %Qubit* }* %0, i32 0, i32 0
  %2 = load %Qubit*, %Qubit** %1, align 8
  %3 = call %Result* @Microsoft__Quantum__Intrinsic__M__body(%Qubit* %2)
  %4 = bitcast %Tuple* %result-tuple to { %Result* }*
  %5 = getelementptr inbounds { %Result* }, { %Result* }* %4, i32 0, i32 0
  store %Result* %3, %Result** %5, align 8
  ret void
}

define void @Microsoft__Quantum__Samples__QAOA__RunQAOATrials__Interop() #0 {
entry:
  call void @Microsoft__Quantum__Samples__QAOA__RunQAOATrials__body()
  ret void
}

define void @Microsoft__Quantum__Samples__QAOA__RunQAOATrials() #1 {
entry:
  call void @Microsoft__Quantum__Samples__QAOA__RunQAOATrials__body()
  %0 = call %String* @__quantum__rt__string_create(i8* getelementptr inbounds ([3 x i8], [3 x i8]* @17, i32 0, i32 0))
  call void @__quantum__rt__message(%String* %0)
  call void @__quantum__rt__string_update_reference_count(%String* %0, i32 -1)
  ret void
}

attributes #0 = { "InteropFriendly" }
attributes #1 = { "EntryPoint" }
