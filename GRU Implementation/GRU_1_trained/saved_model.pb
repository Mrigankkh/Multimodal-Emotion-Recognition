??0
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
:
Less
x"T
y"T
z
"
Ttype:
2	
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
-
Tanh
x"T
y"T"
Ttype:

2
?
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type
output_handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListReserve
element_shape"
shape_type
num_elements

handle"
element_dtypetype"

shape_typetype:
2	
?
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint?????????
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
?
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??.
?
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_namebatch_normalization/gamma
?
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes	
:?*
dtype0
?
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*)
shared_namebatch_normalization/beta
?
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*0
shared_name!batch_normalization/moving_mean
?
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes	
:?*
dtype0
?
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#batch_normalization/moving_variance
?
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes	
:?*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:5*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
_output_shapes
: *
dtype0
?
gru/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*$
shared_namegru/gru_cell/kernel
}
'gru/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/kernel* 
_output_shapes
:
??*
dtype0
?
gru/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	5?*.
shared_namegru/gru_cell/recurrent_kernel
?
1gru/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOpgru/gru_cell/recurrent_kernel*
_output_shapes
:	5?*
dtype0

gru/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*"
shared_namegru/gru_cell/bias
x
%gru/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/gru_cell/bias*
_output_shapes
:	?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
%RMSprop/batch_normalization/gamma/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%RMSprop/batch_normalization/gamma/rms
?
9RMSprop/batch_normalization/gamma/rms/Read/ReadVariableOpReadVariableOp%RMSprop/batch_normalization/gamma/rms*
_output_shapes	
:?*
dtype0
?
$RMSprop/batch_normalization/beta/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*5
shared_name&$RMSprop/batch_normalization/beta/rms
?
8RMSprop/batch_normalization/beta/rms/Read/ReadVariableOpReadVariableOp$RMSprop/batch_normalization/beta/rms*
_output_shapes	
:?*
dtype0
?
RMSprop/dense/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:5*)
shared_nameRMSprop/dense/kernel/rms
?
,RMSprop/dense/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/rms*
_output_shapes

:5*
dtype0
?
RMSprop/dense/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameRMSprop/dense/bias/rms
}
*RMSprop/dense/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/rms*
_output_shapes
:*
dtype0
?
RMSprop/dense_1/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameRMSprop/dense_1/kernel/rms
?
.RMSprop/dense_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/kernel/rms*
_output_shapes

:*
dtype0
?
RMSprop/dense_1/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_1/bias/rms
?
,RMSprop/dense_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_1/bias/rms*
_output_shapes
:*
dtype0
?
RMSprop/dense_2/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*+
shared_nameRMSprop/dense_2/kernel/rms
?
.RMSprop/dense_2/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/kernel/rms*
_output_shapes

:*
dtype0
?
RMSprop/dense_2/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_nameRMSprop/dense_2/bias/rms
?
,RMSprop/dense_2/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense_2/bias/rms*
_output_shapes
:*
dtype0
?
RMSprop/gru/gru_cell/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*0
shared_name!RMSprop/gru/gru_cell/kernel/rms
?
3RMSprop/gru/gru_cell/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/gru/gru_cell/kernel/rms* 
_output_shapes
:
??*
dtype0
?
)RMSprop/gru/gru_cell/recurrent_kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	5?*:
shared_name+)RMSprop/gru/gru_cell/recurrent_kernel/rms
?
=RMSprop/gru/gru_cell/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp)RMSprop/gru/gru_cell/recurrent_kernel/rms*
_output_shapes
:	5?*
dtype0
?
RMSprop/gru/gru_cell/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*.
shared_nameRMSprop/gru/gru_cell/bias/rms
?
1RMSprop/gru/gru_cell/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/gru/gru_cell/bias/rms*
_output_shapes
:	?*
dtype0

NoOpNoOp
?4
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?4
value?4B?4 B?4
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
?
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
l
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
h

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
h

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
?
-iter
	.decay
/learning_rate
0momentum
1rho	rmsh	rmsi	rmsj	rmsk	!rmsl	"rmsm	'rmsn	(rmso	2rmsp	3rmsq	4rmsr
^
0
1
2
3
24
35
46
7
8
!9
"10
'11
(12
N
0
1
22
33
44
5
6
!7
"8
'9
(10
 
?
5non_trainable_variables
	variables
6metrics
7layer_metrics
8layer_regularization_losses
trainable_variables

9layers
	regularization_losses
 
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

0
1
 
?
:non_trainable_variables
	variables
;metrics
<layer_metrics
=layer_regularization_losses
trainable_variables

>layers
regularization_losses
~

2kernel
3recurrent_kernel
4bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
 

20
31
42

20
31
42
 
?
Cnon_trainable_variables
	variables
Dmetrics
Elayer_metrics
Flayer_regularization_losses
trainable_variables

Glayers
regularization_losses

Hstates
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Inon_trainable_variables
	variables
Jmetrics
Klayer_metrics
Llayer_regularization_losses
trainable_variables

Mlayers
regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

!0
"1

!0
"1
 
?
Nnon_trainable_variables
#	variables
Ometrics
Player_metrics
Qlayer_regularization_losses
$trainable_variables

Rlayers
%regularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

'0
(1

'0
(1
 
?
Snon_trainable_variables
)	variables
Tmetrics
Ulayer_metrics
Vlayer_regularization_losses
*trainable_variables

Wlayers
+regularization_losses
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEgru/gru_cell/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEgru/gru_cell/recurrent_kernel&variables/5/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEgru/gru_cell/bias&variables/6/.ATTRIBUTES/VARIABLE_VALUE

0
1

X0
Y1
 
 
#
0
1
2
3
4

0
1
 
 
 
 

20
31
42

20
31
42
 
?
Znon_trainable_variables
?	variables
[metrics
\layer_metrics
]layer_regularization_losses
@trainable_variables

^layers
Aregularization_losses
 
 
 
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	_total
	`count
a	variables
b	keras_api
D
	ctotal
	dcount
e
_fn_kwargs
f	variables
g	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

_0
`1

a	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

c0
d1

f	variables
??
VARIABLE_VALUE%RMSprop/batch_normalization/gamma/rmsSlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE$RMSprop/batch_normalization/beta/rmsRlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/dense/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense_1/kernel/rmsTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUERMSprop/dense_1/bias/rmsRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUERMSprop/dense_2/kernel/rmsTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?~
VARIABLE_VALUERMSprop/dense_2/bias/rmsRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUERMSprop/gru/gru_cell/kernel/rmsDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE)RMSprop/gru/gru_cell/recurrent_kernel/rmsDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUERMSprop/gru/gru_cell/bias/rmsDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
?
)serving_default_batch_normalization_inputPlaceholder*,
_output_shapes
:?????????5?*
dtype0*!
shape:?????????5?
?
StatefulPartitionedCallStatefulPartitionedCall)serving_default_batch_normalization_input#batch_normalization/moving_variancebatch_normalization/gammabatch_normalization/moving_meanbatch_normalization/betagru/gru_cell/biasgru/gru_cell/kernelgru/gru_cell/recurrent_kerneldense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_21132
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOp'gru/gru_cell/kernel/Read/ReadVariableOp1gru/gru_cell/recurrent_kernel/Read/ReadVariableOp%gru/gru_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp9RMSprop/batch_normalization/gamma/rms/Read/ReadVariableOp8RMSprop/batch_normalization/beta/rms/Read/ReadVariableOp,RMSprop/dense/kernel/rms/Read/ReadVariableOp*RMSprop/dense/bias/rms/Read/ReadVariableOp.RMSprop/dense_1/kernel/rms/Read/ReadVariableOp,RMSprop/dense_1/bias/rms/Read/ReadVariableOp.RMSprop/dense_2/kernel/rms/Read/ReadVariableOp,RMSprop/dense_2/bias/rms/Read/ReadVariableOp3RMSprop/gru/gru_cell/kernel/rms/Read/ReadVariableOp=RMSprop/gru/gru_cell/recurrent_kernel/rms/Read/ReadVariableOp1RMSprop/gru/gru_cell/bias/rms/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_23940
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variancedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhogru/gru_cell/kernelgru/gru_cell/recurrent_kernelgru/gru_cell/biastotalcounttotal_1count_1%RMSprop/batch_normalization/gamma/rms$RMSprop/batch_normalization/beta/rmsRMSprop/dense/kernel/rmsRMSprop/dense/bias/rmsRMSprop/dense_1/kernel/rmsRMSprop/dense_1/bias/rmsRMSprop/dense_2/kernel/rmsRMSprop/dense_2/bias/rmsRMSprop/gru/gru_cell/kernel/rms)RMSprop/gru/gru_cell/recurrent_kernel/rmsRMSprop/gru/gru_cell/bias/rms*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_24049??-
??
?
>__inference_gru_layer_call_and_return_conditional_losses_22489

inputs$
 gru_cell_readvariableop_resource&
"gru_cell_readvariableop_1_resource&
"gru_cell_readvariableop_4_resource
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :52
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :52
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????52
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:5??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2|
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like/Const?
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
gru_cell/dropout/Const?
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shape?
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ީ2/
-gru_cell/dropout/random_uniform/RandomUniform?
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2!
gru_cell/dropout/GreaterEqual/y?
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/GreaterEqual?
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout/Cast?
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
gru_cell/dropout_1/Const?
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shape?
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??n21
/gru_cell/dropout_1/random_uniform/RandomUniform?
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!gru_cell/dropout_1/GreaterEqual/y?
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell/dropout_1/GreaterEqual?
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout_1/Cast?
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
gru_cell/dropout_2/Const?
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shape?
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2أ)21
/gru_cell/dropout_2/random_uniform/RandomUniform?
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!gru_cell/dropout_2/GreaterEqual/y?
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell/dropout_2/GreaterEqual?
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout_2/Cast?
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_2/Mul_1v
gru_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like_1/Shape}
gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like_1/Const?
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/ones_like_1y
gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/dropout_3/Const?
gru_cell/dropout_3/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_3/Mul?
gru_cell/dropout_3/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_3/Shape?
/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2??21
/gru_cell/dropout_3/random_uniform/RandomUniform?
!gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!gru_cell/dropout_3/GreaterEqual/y?
gru_cell/dropout_3/GreaterEqualGreaterEqual8gru_cell/dropout_3/random_uniform/RandomUniform:output:0*gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52!
gru_cell/dropout_3/GreaterEqual?
gru_cell/dropout_3/CastCast#gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
gru_cell/dropout_3/Cast?
gru_cell/dropout_3/Mul_1Mulgru_cell/dropout_3/Mul:z:0gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_3/Mul_1y
gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/dropout_4/Const?
gru_cell/dropout_4/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_4/Mul?
gru_cell/dropout_4/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_4/Shape?
/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???21
/gru_cell/dropout_4/random_uniform/RandomUniform?
!gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!gru_cell/dropout_4/GreaterEqual/y?
gru_cell/dropout_4/GreaterEqualGreaterEqual8gru_cell/dropout_4/random_uniform/RandomUniform:output:0*gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52!
gru_cell/dropout_4/GreaterEqual?
gru_cell/dropout_4/CastCast#gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
gru_cell/dropout_4/Cast?
gru_cell/dropout_4/Mul_1Mulgru_cell/dropout_4/Mul:z:0gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_4/Mul_1y
gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/dropout_5/Const?
gru_cell/dropout_5/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_5/Mul?
gru_cell/dropout_5/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_5/Shape?
/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2˥?21
/gru_cell/dropout_5/random_uniform/RandomUniform?
!gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!gru_cell/dropout_5/GreaterEqual/y?
gru_cell/dropout_5/GreaterEqualGreaterEqual8gru_cell/dropout_5/random_uniform/RandomUniform:output:0*gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52!
gru_cell/dropout_5/GreaterEqual?
gru_cell/dropout_5/CastCast#gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
gru_cell/dropout_5/Cast?
gru_cell/dropout_5/Mul_1Mulgru_cell/dropout_5/Mul:z:0gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_5/Mul_1?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/mulMulstrided_slice_2:output:0gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1?
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul?
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_2?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_1?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_2?
gru_cell/mul_3Mulzeros:output:0gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_3?
gru_cell/mul_4Mulzeros:output:0gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_4?
gru_cell/mul_5Mulzeros:output:0gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_5?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_4?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru_cell/strided_slice_8?
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_3?
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52 
gru_cell/strided_slice_9/stack?
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2"
 gru_cell/strided_slice_9/stack_1?
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2?
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru_cell/strided_slice_9?
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Sigmoid_1?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2!
gru_cell/strided_slice_10/stack?
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1?
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2?
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_10?
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_5?
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2!
gru_cell/strided_slice_11/stack?
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1?
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2?
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru_cell/strided_slice_11?
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_5?
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_6?
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_8?
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????5: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_22289*
condR
while_cond_22288*8
output_shapes'
%: : : : :?????????5: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:5?????????5*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????5*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????552
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*'
_output_shapes
:?????????52

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????5?:::22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
?
?
#__inference_gru_layer_call_fn_23478
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_200462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????52

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?
|
'__inference_dense_2_layer_call_fn_23538

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_209062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
C__inference_gru_cell_layer_call_and_return_conditional_losses_23688

inputs
states_0
readvariableop_resource
readvariableop_1_resource
readvariableop_4_resource
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const?
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?Ź2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_2/Mul_1^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
ones_like_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_3/Const?
dropout_3/MulMulones_like_1:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????52
dropout_3/Mulf
dropout_3/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????52
dropout_3/Mul_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_4/Const?
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????52
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shape?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2?ױ2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_4/GreaterEqual/y?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52
dropout_4/GreaterEqual?
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
dropout_4/Cast?
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????52
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_5/Const?
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????52
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shape?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2?آ2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_5/GreaterEqual/y?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52
dropout_5/GreaterEqual?
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
dropout_5/Cast?
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????52
dropout_5/Mul_1y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack_
mulMulinputsdropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mule
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_1e
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_2?
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????52
MatMul?
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52

MatMul_1?
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52

MatMul_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceunstack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52	
BiasAddx
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_1x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_2f
mul_3Mulstates_0dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
mul_3f
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
mul_4f
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
mul_5
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:	5?*
dtype02
ReadVariableOp_4
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_4:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
strided_slice_6u
MatMul_3MatMul	mul_3:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52

MatMul_3
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource*
_output_shapes
:	5?*
dtype02
ReadVariableOp_5
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_5:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
strided_slice_7u
MatMul_4MatMul	mul_4:z:0strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52

MatMul_4x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
strided_slice_8?
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_3x
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_9/stack|
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2?
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
strided_slice_9?
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????52	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????52
	Sigmoid_1
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes
:	5?*
dtype02
ReadVariableOp_6?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_6:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
strided_slice_10v
MatMul_5MatMul	mul_5:z:0strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52

MatMul_5z
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2?
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
strided_slice_11?
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:?????????52
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????52
Tanh^
mul_7MulSigmoid:y:0states_0*
T0*'
_output_shapes
:?????????52
mul_7S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????52
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:?????????52
add_3?
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*'
_output_shapes
:?????????52

Identity?

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*'
_output_shapes
:?????????52

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????5:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_6:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????5
"
_user_specified_name
states/0
??
?

E__inference_sequential_layer_call_and_return_conditional_losses_21884

inputs9
5batch_normalization_batchnorm_readvariableop_resource=
9batch_normalization_batchnorm_mul_readvariableop_resource;
7batch_normalization_batchnorm_readvariableop_1_resource;
7batch_normalization_batchnorm_readvariableop_2_resource(
$gru_gru_cell_readvariableop_resource*
&gru_gru_cell_readvariableop_1_resource*
&gru_gru_cell_readvariableop_4_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??,batch_normalization/batchnorm/ReadVariableOp?.batch_normalization/batchnorm/ReadVariableOp_1?.batch_normalization/batchnorm/ReadVariableOp_2?0batch_normalization/batchnorm/mul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?gru/gru_cell/ReadVariableOp?gru/gru_cell/ReadVariableOp_1?gru/gru_cell/ReadVariableOp_2?gru/gru_cell/ReadVariableOp_3?gru/gru_cell/ReadVariableOp_4?gru/gru_cell/ReadVariableOp_5?gru/gru_cell/ReadVariableOp_6?	gru/while?
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp?
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2%
#batch_normalization/batchnorm/add/y?
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/add?
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/Rsqrt?
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/mul?
#batch_normalization/batchnorm/mul_1Mulinputs%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????5?2%
#batch_normalization/batchnorm/mul_1?
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1?
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/mul_2?
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2?
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/sub?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????5?2%
#batch_normalization/batchnorm/add_1m
	gru/ShapeShape'batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack?
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1?
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2?
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_sliced
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :52
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessj
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :52
gru/zeros/packed/1?
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const?
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:?????????52
	gru/zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm?
gru/transpose	Transpose'batch_normalization/batchnorm/add_1:z:0gru/transpose/perm:output:0*
T0*,
_output_shapes
:5??????????2
gru/transpose[
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:2
gru/Shape_1?
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack?
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1?
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2?
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice_1?
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru/TensorArrayV2/element_shape?
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2?
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor?
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack?
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_1?
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2?
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru/strided_slice_2?
gru/gru_cell/ones_like/ShapeShapegru/strided_slice_2:output:0*
T0*
_output_shapes
:2
gru/gru_cell/ones_like/Shape?
gru/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/ones_like/Const?
gru/gru_cell/ones_likeFill%gru/gru_cell/ones_like/Shape:output:0%gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/ones_like?
gru/gru_cell/ones_like_1/ShapeShapegru/zeros:output:0*
T0*
_output_shapes
:2 
gru/gru_cell/ones_like_1/Shape?
gru/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru/gru_cell/ones_like_1/Const?
gru/gru_cell/ones_like_1Fill'gru/gru_cell/ones_like_1/Shape:output:0'gru/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/ones_like_1?
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru/gru_cell/ReadVariableOp?
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/gru_cell/unstack?
gru/gru_cell/mulMulgru/strided_slice_2:output:0gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul?
gru/gru_cell/mul_1Mulgru/strided_slice_2:output:0gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_1?
gru/gru_cell/mul_2Mulgru/strided_slice_2:output:0gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_2?
gru/gru_cell/ReadVariableOp_1ReadVariableOp&gru_gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru/gru_cell/ReadVariableOp_1?
 gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru/gru_cell/strided_slice/stack?
"gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2$
"gru/gru_cell/strided_slice/stack_1?
"gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru/gru_cell/strided_slice/stack_2?
gru/gru_cell/strided_sliceStridedSlice%gru/gru_cell/ReadVariableOp_1:value:0)gru/gru_cell/strided_slice/stack:output:0+gru/gru_cell/strided_slice/stack_1:output:0+gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru/gru_cell/strided_slice?
gru/gru_cell/MatMulMatMulgru/gru_cell/mul:z:0#gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/MatMul?
gru/gru_cell/ReadVariableOp_2ReadVariableOp&gru_gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru/gru_cell/ReadVariableOp_2?
"gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2$
"gru/gru_cell/strided_slice_1/stack?
$gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2&
$gru/gru_cell/strided_slice_1/stack_1?
$gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_1/stack_2?
gru/gru_cell/strided_slice_1StridedSlice%gru/gru_cell/ReadVariableOp_2:value:0+gru/gru_cell/strided_slice_1/stack:output:0-gru/gru_cell/strided_slice_1/stack_1:output:0-gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_1?
gru/gru_cell/MatMul_1MatMulgru/gru_cell/mul_1:z:0%gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/MatMul_1?
gru/gru_cell/ReadVariableOp_3ReadVariableOp&gru_gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru/gru_cell/ReadVariableOp_3?
"gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2$
"gru/gru_cell/strided_slice_2/stack?
$gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_2/stack_1?
$gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_2/stack_2?
gru/gru_cell/strided_slice_2StridedSlice%gru/gru_cell/ReadVariableOp_3:value:0+gru/gru_cell/strided_slice_2/stack:output:0-gru/gru_cell/strided_slice_2/stack_1:output:0-gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_2?
gru/gru_cell/MatMul_2MatMulgru/gru_cell/mul_2:z:0%gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/MatMul_2?
"gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_3/stack?
$gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52&
$gru/gru_cell/strided_slice_3/stack_1?
$gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_3/stack_2?
gru/gru_cell/strided_slice_3StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_3/stack:output:0-gru/gru_cell/strided_slice_3/stack_1:output:0-gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru/gru_cell/strided_slice_3?
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0%gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/BiasAdd?
"gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52$
"gru/gru_cell/strided_slice_4/stack?
$gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2&
$gru/gru_cell/strided_slice_4/stack_1?
$gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_4/stack_2?
gru/gru_cell/strided_slice_4StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_4/stack:output:0-gru/gru_cell/strided_slice_4/stack_1:output:0-gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru/gru_cell/strided_slice_4?
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0%gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/BiasAdd_1?
"gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2$
"gru/gru_cell/strided_slice_5/stack?
$gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_5/stack_1?
$gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_5/stack_2?
gru/gru_cell/strided_slice_5StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_5/stack:output:0-gru/gru_cell/strided_slice_5/stack_1:output:0-gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru/gru_cell/strided_slice_5?
gru/gru_cell/BiasAdd_2BiasAddgru/gru_cell/MatMul_2:product:0%gru/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/BiasAdd_2?
gru/gru_cell/mul_3Mulgru/zeros:output:0!gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/mul_3?
gru/gru_cell/mul_4Mulgru/zeros:output:0!gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/mul_4?
gru/gru_cell/mul_5Mulgru/zeros:output:0!gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/mul_5?
gru/gru_cell/ReadVariableOp_4ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru/gru_cell/ReadVariableOp_4?
"gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_6/stack?
$gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2&
$gru/gru_cell/strided_slice_6/stack_1?
$gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_6/stack_2?
gru/gru_cell/strided_slice_6StridedSlice%gru/gru_cell/ReadVariableOp_4:value:0+gru/gru_cell/strided_slice_6/stack:output:0-gru/gru_cell/strided_slice_6/stack_1:output:0-gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_6?
gru/gru_cell/MatMul_3MatMulgru/gru_cell/mul_3:z:0%gru/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/MatMul_3?
gru/gru_cell/ReadVariableOp_5ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru/gru_cell/ReadVariableOp_5?
"gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2$
"gru/gru_cell/strided_slice_7/stack?
$gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2&
$gru/gru_cell/strided_slice_7/stack_1?
$gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_7/stack_2?
gru/gru_cell/strided_slice_7StridedSlice%gru/gru_cell/ReadVariableOp_5:value:0+gru/gru_cell/strided_slice_7/stack:output:0-gru/gru_cell/strided_slice_7/stack_1:output:0-gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_7?
gru/gru_cell/MatMul_4MatMulgru/gru_cell/mul_4:z:0%gru/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/MatMul_4?
"gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_8/stack?
$gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52&
$gru/gru_cell/strided_slice_8/stack_1?
$gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_8/stack_2?
gru/gru_cell/strided_slice_8StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_8/stack:output:0-gru/gru_cell/strided_slice_8/stack_1:output:0-gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru/gru_cell/strided_slice_8?
gru/gru_cell/BiasAdd_3BiasAddgru/gru_cell/MatMul_3:product:0%gru/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/BiasAdd_3?
"gru/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52$
"gru/gru_cell/strided_slice_9/stack?
$gru/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2&
$gru/gru_cell/strided_slice_9/stack_1?
$gru/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_9/stack_2?
gru/gru_cell/strided_slice_9StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_9/stack:output:0-gru/gru_cell/strided_slice_9/stack_1:output:0-gru/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru/gru_cell/strided_slice_9?
gru/gru_cell/BiasAdd_4BiasAddgru/gru_cell/MatMul_4:product:0%gru/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/BiasAdd_4?
gru/gru_cell/addAddV2gru/gru_cell/BiasAdd:output:0gru/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/add
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/Sigmoid?
gru/gru_cell/add_1AddV2gru/gru_cell/BiasAdd_1:output:0gru/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/add_1?
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/Sigmoid_1?
gru/gru_cell/ReadVariableOp_6ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru/gru_cell/ReadVariableOp_6?
#gru/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2%
#gru/gru_cell/strided_slice_10/stack?
%gru/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%gru/gru_cell/strided_slice_10/stack_1?
%gru/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%gru/gru_cell/strided_slice_10/stack_2?
gru/gru_cell/strided_slice_10StridedSlice%gru/gru_cell/ReadVariableOp_6:value:0,gru/gru_cell/strided_slice_10/stack:output:0.gru/gru_cell/strided_slice_10/stack_1:output:0.gru/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_10?
gru/gru_cell/MatMul_5MatMulgru/gru_cell/mul_5:z:0&gru/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/MatMul_5?
#gru/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2%
#gru/gru_cell/strided_slice_11/stack?
%gru/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%gru/gru_cell/strided_slice_11/stack_1?
%gru/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%gru/gru_cell/strided_slice_11/stack_2?
gru/gru_cell/strided_slice_11StridedSlicegru/gru_cell/unstack:output:1,gru/gru_cell/strided_slice_11/stack:output:0.gru/gru_cell/strided_slice_11/stack_1:output:0.gru/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru/gru_cell/strided_slice_11?
gru/gru_cell/BiasAdd_5BiasAddgru/gru_cell/MatMul_5:product:0&gru/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/BiasAdd_5?
gru/gru_cell/mul_6Mulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/mul_6?
gru/gru_cell/add_2AddV2gru/gru_cell/BiasAdd_2:output:0gru/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/add_2x
gru/gru_cell/TanhTanhgru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/Tanh?
gru/gru_cell/mul_7Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/mul_7m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/sub/x?
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/sub?
gru/gru_cell/mul_8Mulgru/gru_cell/sub:z:0gru/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/mul_8?
gru/gru_cell/add_3AddV2gru/gru_cell/mul_7:z:0gru/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/add_3?
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   2#
!gru/TensorArrayV2_1/element_shape?
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time?
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counter?
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource&gru_gru_cell_readvariableop_1_resource&gru_gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????5: : : : : *%
_read_only_resource_inputs
	* 
bodyR
gru_while_body_21711* 
condR
gru_while_cond_21710*8
output_shapes'
%: : : : :?????????5: : : : : *
parallel_iterations 2
	gru/while?
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   26
4gru/TensorArrayV2Stack/TensorListStack/element_shape?
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:5?????????5*
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack?
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru/strided_slice_3/stack?
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_3/stack_1?
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_3/stack_2?
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????5*
shrink_axis_mask2
gru/strided_slice_3?
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm?
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????552
gru/transpose_1n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtime?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:5*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulgru/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense/Sigmoid?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Sigmoid:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoid?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Sigmoid:y:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Sigmoid?
IdentityIdentitydense_2/Sigmoid:y:0-^batch_normalization/batchnorm/ReadVariableOp/^batch_normalization/batchnorm/ReadVariableOp_1/^batch_normalization/batchnorm/ReadVariableOp_21^batch_normalization/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^gru/gru_cell/ReadVariableOp^gru/gru_cell/ReadVariableOp_1^gru/gru_cell/ReadVariableOp_2^gru/gru_cell/ReadVariableOp_3^gru/gru_cell/ReadVariableOp_4^gru/gru_cell/ReadVariableOp_5^gru/gru_cell/ReadVariableOp_6
^gru/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????5?:::::::::::::2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2`
.batch_normalization/batchnorm/ReadVariableOp_1.batch_normalization/batchnorm/ReadVariableOp_12`
.batch_normalization/batchnorm/ReadVariableOp_2.batch_normalization/batchnorm/ReadVariableOp_22d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2>
gru/gru_cell/ReadVariableOp_1gru/gru_cell/ReadVariableOp_12>
gru/gru_cell/ReadVariableOp_2gru/gru_cell/ReadVariableOp_22>
gru/gru_cell/ReadVariableOp_3gru/gru_cell/ReadVariableOp_32>
gru/gru_cell/ReadVariableOp_4gru/gru_cell/ReadVariableOp_42>
gru/gru_cell/ReadVariableOp_5gru/gru_cell/ReadVariableOp_52>
gru/gru_cell/ReadVariableOp_6gru/gru_cell/ReadVariableOp_62
	gru/while	gru/while:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
ÿ
?
gru_while_body_21711$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_00
,gru_while_gru_cell_readvariableop_resource_02
.gru_while_gru_cell_readvariableop_1_resource_02
.gru_while_gru_cell_readvariableop_4_resource_0
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor.
*gru_while_gru_cell_readvariableop_resource0
,gru_while_gru_cell_readvariableop_1_resource0
,gru_while_gru_cell_readvariableop_4_resource??!gru/while/gru_cell/ReadVariableOp?#gru/while/gru_cell/ReadVariableOp_1?#gru/while/gru_cell/ReadVariableOp_2?#gru/while/gru_cell/ReadVariableOp_3?#gru/while/gru_cell/ReadVariableOp_4?#gru/while/gru_cell/ReadVariableOp_5?#gru/while/gru_cell/ReadVariableOp_6?
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItem?
"gru/while/gru_cell/ones_like/ShapeShape4gru/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/ones_like/Shape?
"gru/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"gru/while/gru_cell/ones_like/Const?
gru/while/gru_cell/ones_likeFill+gru/while/gru_cell/ones_like/Shape:output:0+gru/while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/ones_like?
$gru/while/gru_cell/ones_like_1/ShapeShapegru_while_placeholder_2*
T0*
_output_shapes
:2&
$gru/while/gru_cell/ones_like_1/Shape?
$gru/while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$gru/while/gru_cell/ones_like_1/Const?
gru/while/gru_cell/ones_like_1Fill-gru/while/gru_cell/ones_like_1/Shape:output:0-gru/while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52 
gru/while/gru_cell/ones_like_1?
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02#
!gru/while/gru_cell/ReadVariableOp?
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/while/gru_cell/unstack?
gru/while/gru_cell/mulMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul?
gru/while/gru_cell/mul_1Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_1?
gru/while/gru_cell/mul_2Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0%gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_2?
#gru/while/gru_cell/ReadVariableOp_1ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02%
#gru/while/gru_cell/ReadVariableOp_1?
&gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/while/gru_cell/strided_slice/stack?
(gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2*
(gru/while/gru_cell/strided_slice/stack_1?
(gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru/while/gru_cell/strided_slice/stack_2?
 gru/while/gru_cell/strided_sliceStridedSlice+gru/while/gru_cell/ReadVariableOp_1:value:0/gru/while/gru_cell/strided_slice/stack:output:01gru/while/gru_cell/strided_slice/stack_1:output:01gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2"
 gru/while/gru_cell/strided_slice?
gru/while/gru_cell/MatMulMatMulgru/while/gru_cell/mul:z:0)gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/MatMul?
#gru/while/gru_cell/ReadVariableOp_2ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02%
#gru/while/gru_cell/ReadVariableOp_2?
(gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2*
(gru/while/gru_cell/strided_slice_1/stack?
*gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2,
*gru/while/gru_cell/strided_slice_1/stack_1?
*gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_1/stack_2?
"gru/while/gru_cell/strided_slice_1StridedSlice+gru/while/gru_cell/ReadVariableOp_2:value:01gru/while/gru_cell/strided_slice_1/stack:output:03gru/while/gru_cell/strided_slice_1/stack_1:output:03gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_1?
gru/while/gru_cell/MatMul_1MatMulgru/while/gru_cell/mul_1:z:0+gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/MatMul_1?
#gru/while/gru_cell/ReadVariableOp_3ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02%
#gru/while/gru_cell/ReadVariableOp_3?
(gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2*
(gru/while/gru_cell/strided_slice_2/stack?
*gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_2/stack_1?
*gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_2/stack_2?
"gru/while/gru_cell/strided_slice_2StridedSlice+gru/while/gru_cell/ReadVariableOp_3:value:01gru/while/gru_cell/strided_slice_2/stack:output:03gru/while/gru_cell/strided_slice_2/stack_1:output:03gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_2?
gru/while/gru_cell/MatMul_2MatMulgru/while/gru_cell/mul_2:z:0+gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/MatMul_2?
(gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_3/stack?
*gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52,
*gru/while/gru_cell/strided_slice_3/stack_1?
*gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_3/stack_2?
"gru/while/gru_cell/strided_slice_3StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_3/stack:output:03gru/while/gru_cell/strided_slice_3/stack_1:output:03gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2$
"gru/while/gru_cell/strided_slice_3?
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0+gru/while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/BiasAdd?
(gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52*
(gru/while/gru_cell/strided_slice_4/stack?
*gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2,
*gru/while/gru_cell/strided_slice_4/stack_1?
*gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_4/stack_2?
"gru/while/gru_cell/strided_slice_4StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_4/stack:output:03gru/while/gru_cell/strided_slice_4/stack_1:output:03gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52$
"gru/while/gru_cell/strided_slice_4?
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0+gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/BiasAdd_1?
(gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2*
(gru/while/gru_cell/strided_slice_5/stack?
*gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_5/stack_1?
*gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_5/stack_2?
"gru/while/gru_cell/strided_slice_5StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_5/stack:output:03gru/while/gru_cell/strided_slice_5/stack_1:output:03gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2$
"gru/while/gru_cell/strided_slice_5?
gru/while/gru_cell/BiasAdd_2BiasAdd%gru/while/gru_cell/MatMul_2:product:0+gru/while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/BiasAdd_2?
gru/while/gru_cell/mul_3Mulgru_while_placeholder_2'gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/mul_3?
gru/while/gru_cell/mul_4Mulgru_while_placeholder_2'gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/mul_4?
gru/while/gru_cell/mul_5Mulgru_while_placeholder_2'gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/mul_5?
#gru/while/gru_cell/ReadVariableOp_4ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02%
#gru/while/gru_cell/ReadVariableOp_4?
(gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_6/stack?
*gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2,
*gru/while/gru_cell/strided_slice_6/stack_1?
*gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_6/stack_2?
"gru/while/gru_cell/strided_slice_6StridedSlice+gru/while/gru_cell/ReadVariableOp_4:value:01gru/while/gru_cell/strided_slice_6/stack:output:03gru/while/gru_cell/strided_slice_6/stack_1:output:03gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_6?
gru/while/gru_cell/MatMul_3MatMulgru/while/gru_cell/mul_3:z:0+gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/MatMul_3?
#gru/while/gru_cell/ReadVariableOp_5ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02%
#gru/while/gru_cell/ReadVariableOp_5?
(gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2*
(gru/while/gru_cell/strided_slice_7/stack?
*gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2,
*gru/while/gru_cell/strided_slice_7/stack_1?
*gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_7/stack_2?
"gru/while/gru_cell/strided_slice_7StridedSlice+gru/while/gru_cell/ReadVariableOp_5:value:01gru/while/gru_cell/strided_slice_7/stack:output:03gru/while/gru_cell/strided_slice_7/stack_1:output:03gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_7?
gru/while/gru_cell/MatMul_4MatMulgru/while/gru_cell/mul_4:z:0+gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/MatMul_4?
(gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_8/stack?
*gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52,
*gru/while/gru_cell/strided_slice_8/stack_1?
*gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_8/stack_2?
"gru/while/gru_cell/strided_slice_8StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_8/stack:output:03gru/while/gru_cell/strided_slice_8/stack_1:output:03gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2$
"gru/while/gru_cell/strided_slice_8?
gru/while/gru_cell/BiasAdd_3BiasAdd%gru/while/gru_cell/MatMul_3:product:0+gru/while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/BiasAdd_3?
(gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52*
(gru/while/gru_cell/strided_slice_9/stack?
*gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2,
*gru/while/gru_cell/strided_slice_9/stack_1?
*gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_9/stack_2?
"gru/while/gru_cell/strided_slice_9StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_9/stack:output:03gru/while/gru_cell/strided_slice_9/stack_1:output:03gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52$
"gru/while/gru_cell/strided_slice_9?
gru/while/gru_cell/BiasAdd_4BiasAdd%gru/while/gru_cell/MatMul_4:product:0+gru/while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/BiasAdd_4?
gru/while/gru_cell/addAddV2#gru/while/gru_cell/BiasAdd:output:0%gru/while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/add?
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/Sigmoid?
gru/while/gru_cell/add_1AddV2%gru/while/gru_cell/BiasAdd_1:output:0%gru/while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/add_1?
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/Sigmoid_1?
#gru/while/gru_cell/ReadVariableOp_6ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02%
#gru/while/gru_cell/ReadVariableOp_6?
)gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2+
)gru/while/gru_cell/strided_slice_10/stack?
+gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+gru/while/gru_cell/strided_slice_10/stack_1?
+gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+gru/while/gru_cell/strided_slice_10/stack_2?
#gru/while/gru_cell/strided_slice_10StridedSlice+gru/while/gru_cell/ReadVariableOp_6:value:02gru/while/gru_cell/strided_slice_10/stack:output:04gru/while/gru_cell/strided_slice_10/stack_1:output:04gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2%
#gru/while/gru_cell/strided_slice_10?
gru/while/gru_cell/MatMul_5MatMulgru/while/gru_cell/mul_5:z:0,gru/while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/MatMul_5?
)gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2+
)gru/while/gru_cell/strided_slice_11/stack?
+gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+gru/while/gru_cell/strided_slice_11/stack_1?
+gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gru/while/gru_cell/strided_slice_11/stack_2?
#gru/while/gru_cell/strided_slice_11StridedSlice#gru/while/gru_cell/unstack:output:12gru/while/gru_cell/strided_slice_11/stack:output:04gru/while/gru_cell/strided_slice_11/stack_1:output:04gru/while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2%
#gru/while/gru_cell/strided_slice_11?
gru/while/gru_cell/BiasAdd_5BiasAdd%gru/while/gru_cell/MatMul_5:product:0,gru/while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/BiasAdd_5?
gru/while/gru_cell/mul_6Mul gru/while/gru_cell/Sigmoid_1:y:0%gru/while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/mul_6?
gru/while/gru_cell/add_2AddV2%gru/while/gru_cell/BiasAdd_2:output:0gru/while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/add_2?
gru/while/gru_cell/TanhTanhgru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/Tanh?
gru/while/gru_cell/mul_7Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/mul_7y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/while/gru_cell/sub/x?
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/sub?
gru/while/gru_cell/mul_8Mulgru/while/gru_cell/sub:z:0gru/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/mul_8?
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_7:z:0gru/while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/add_3?
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype020
.gru/while/TensorArrayV2Write/TensorListSetItemd
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add/yy
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/while/addh
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add_1/y?
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1?
gru/while/IdentityIdentitygru/while/add_1:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity?
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_1?
gru/while/Identity_2Identitygru/while/add:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_2?
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_3?
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:?????????52
gru/while/Identity_4"^
,gru_while_gru_cell_readvariableop_1_resource.gru_while_gru_cell_readvariableop_1_resource_0"^
,gru_while_gru_cell_readvariableop_4_resource.gru_while_gru_cell_readvariableop_4_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"?
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????5: : :::2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp2J
#gru/while/gru_cell/ReadVariableOp_1#gru/while/gru_cell/ReadVariableOp_12J
#gru/while/gru_cell/ReadVariableOp_2#gru/while/gru_cell/ReadVariableOp_22J
#gru/while/gru_cell/ReadVariableOp_3#gru/while/gru_cell/ReadVariableOp_32J
#gru/while/gru_cell/ReadVariableOp_4#gru/while/gru_cell/ReadVariableOp_42J
#gru/while/gru_cell/ReadVariableOp_5#gru/while/gru_cell/ReadVariableOp_52J
#gru/while/gru_cell/ReadVariableOp_6#gru/while/gru_cell/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
: 
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_23489

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????5::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????5
 
_user_specified_nameinputs
?<
?
>__inference_gru_layer_call_and_return_conditional_losses_19928

inputs
gru_cell_19852
gru_cell_19854
gru_cell_19856
identity?? gru_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :52
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :52
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????52
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_19852gru_cell_19854gru_cell_19856*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????5:?????????5*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_195032"
 gru_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_19852gru_cell_19854gru_cell_19856*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????5: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_19864*
condR
while_cond_19863*8
output_shapes'
%: : : : :?????????5: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????5*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????5*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????52
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0!^gru_cell/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????52

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?	
?
(__inference_gru_cell_layer_call_fn_23804

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????5:?????????5*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_195032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????52

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????52

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????5:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????5
"
_user_specified_name
states/0
?	
?
#__inference_signature_wrapper_21132
batch_normalization_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_191812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????5?:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:g c
,
_output_shapes
:?????????5?
3
_user_specified_namebatch_normalization_input
??
?
gru_while_body_21343$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2#
gru_while_gru_strided_slice_1_0_
[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_00
,gru_while_gru_cell_readvariableop_resource_02
.gru_while_gru_cell_readvariableop_1_resource_02
.gru_while_gru_cell_readvariableop_4_resource_0
gru_while_identity
gru_while_identity_1
gru_while_identity_2
gru_while_identity_3
gru_while_identity_4!
gru_while_gru_strided_slice_1]
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor.
*gru_while_gru_cell_readvariableop_resource0
,gru_while_gru_cell_readvariableop_1_resource0
,gru_while_gru_cell_readvariableop_4_resource??!gru/while/gru_cell/ReadVariableOp?#gru/while/gru_cell/ReadVariableOp_1?#gru/while/gru_cell/ReadVariableOp_2?#gru/while/gru_cell/ReadVariableOp_3?#gru/while/gru_cell/ReadVariableOp_4?#gru/while/gru_cell/ReadVariableOp_5?#gru/while/gru_cell/ReadVariableOp_6?
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2=
;gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0gru_while_placeholderDgru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02/
-gru/while/TensorArrayV2Read/TensorListGetItem?
"gru/while/gru_cell/ones_like/ShapeShape4gru/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/ones_like/Shape?
"gru/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"gru/while/gru_cell/ones_like/Const?
gru/while/gru_cell/ones_likeFill+gru/while/gru_cell/ones_like/Shape:output:0+gru/while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/ones_like?
 gru/while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2"
 gru/while/gru_cell/dropout/Const?
gru/while/gru_cell/dropout/MulMul%gru/while/gru_cell/ones_like:output:0)gru/while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2 
gru/while/gru_cell/dropout/Mul?
 gru/while/gru_cell/dropout/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2"
 gru/while/gru_cell/dropout/Shape?
7gru/while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform)gru/while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???29
7gru/while/gru_cell/dropout/random_uniform/RandomUniform?
)gru/while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2+
)gru/while/gru_cell/dropout/GreaterEqual/y?
'gru/while/gru_cell/dropout/GreaterEqualGreaterEqual@gru/while/gru_cell/dropout/random_uniform/RandomUniform:output:02gru/while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2)
'gru/while/gru_cell/dropout/GreaterEqual?
gru/while/gru_cell/dropout/CastCast+gru/while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2!
gru/while/gru_cell/dropout/Cast?
 gru/while/gru_cell/dropout/Mul_1Mul"gru/while/gru_cell/dropout/Mul:z:0#gru/while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2"
 gru/while/gru_cell/dropout/Mul_1?
"gru/while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2$
"gru/while/gru_cell/dropout_1/Const?
 gru/while/gru_cell/dropout_1/MulMul%gru/while/gru_cell/ones_like:output:0+gru/while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2"
 gru/while/gru_cell/dropout_1/Mul?
"gru/while/gru_cell/dropout_1/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_1/Shape?
9gru/while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?ѥ2;
9gru/while/gru_cell/dropout_1/random_uniform/RandomUniform?
+gru/while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2-
+gru/while/gru_cell/dropout_1/GreaterEqual/y?
)gru/while/gru_cell/dropout_1/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_1/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2+
)gru/while/gru_cell/dropout_1/GreaterEqual?
!gru/while/gru_cell/dropout_1/CastCast-gru/while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2#
!gru/while/gru_cell/dropout_1/Cast?
"gru/while/gru_cell/dropout_1/Mul_1Mul$gru/while/gru_cell/dropout_1/Mul:z:0%gru/while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2$
"gru/while/gru_cell/dropout_1/Mul_1?
"gru/while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2$
"gru/while/gru_cell/dropout_2/Const?
 gru/while/gru_cell/dropout_2/MulMul%gru/while/gru_cell/ones_like:output:0+gru/while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2"
 gru/while/gru_cell/dropout_2/Mul?
"gru/while/gru_cell/dropout_2/ShapeShape%gru/while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_2/Shape?
9gru/while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2흉2;
9gru/while/gru_cell/dropout_2/random_uniform/RandomUniform?
+gru/while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2-
+gru/while/gru_cell/dropout_2/GreaterEqual/y?
)gru/while/gru_cell/dropout_2/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_2/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2+
)gru/while/gru_cell/dropout_2/GreaterEqual?
!gru/while/gru_cell/dropout_2/CastCast-gru/while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2#
!gru/while/gru_cell/dropout_2/Cast?
"gru/while/gru_cell/dropout_2/Mul_1Mul$gru/while/gru_cell/dropout_2/Mul:z:0%gru/while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2$
"gru/while/gru_cell/dropout_2/Mul_1?
$gru/while/gru_cell/ones_like_1/ShapeShapegru_while_placeholder_2*
T0*
_output_shapes
:2&
$gru/while/gru_cell/ones_like_1/Shape?
$gru/while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$gru/while/gru_cell/ones_like_1/Const?
gru/while/gru_cell/ones_like_1Fill-gru/while/gru_cell/ones_like_1/Shape:output:0-gru/while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52 
gru/while/gru_cell/ones_like_1?
"gru/while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"gru/while/gru_cell/dropout_3/Const?
 gru/while/gru_cell/dropout_3/MulMul'gru/while/gru_cell/ones_like_1:output:0+gru/while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????52"
 gru/while/gru_cell/dropout_3/Mul?
"gru/while/gru_cell/dropout_3/ShapeShape'gru/while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_3/Shape?
9gru/while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???2;
9gru/while/gru_cell/dropout_3/random_uniform/RandomUniform?
+gru/while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+gru/while/gru_cell/dropout_3/GreaterEqual/y?
)gru/while/gru_cell/dropout_3/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_3/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52+
)gru/while/gru_cell/dropout_3/GreaterEqual?
!gru/while/gru_cell/dropout_3/CastCast-gru/while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52#
!gru/while/gru_cell/dropout_3/Cast?
"gru/while/gru_cell/dropout_3/Mul_1Mul$gru/while/gru_cell/dropout_3/Mul:z:0%gru/while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????52$
"gru/while/gru_cell/dropout_3/Mul_1?
"gru/while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"gru/while/gru_cell/dropout_4/Const?
 gru/while/gru_cell/dropout_4/MulMul'gru/while/gru_cell/ones_like_1:output:0+gru/while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????52"
 gru/while/gru_cell/dropout_4/Mul?
"gru/while/gru_cell/dropout_4/ShapeShape'gru/while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_4/Shape?
9gru/while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???2;
9gru/while/gru_cell/dropout_4/random_uniform/RandomUniform?
+gru/while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+gru/while/gru_cell/dropout_4/GreaterEqual/y?
)gru/while/gru_cell/dropout_4/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_4/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52+
)gru/while/gru_cell/dropout_4/GreaterEqual?
!gru/while/gru_cell/dropout_4/CastCast-gru/while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52#
!gru/while/gru_cell/dropout_4/Cast?
"gru/while/gru_cell/dropout_4/Mul_1Mul$gru/while/gru_cell/dropout_4/Mul:z:0%gru/while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????52$
"gru/while/gru_cell/dropout_4/Mul_1?
"gru/while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2$
"gru/while/gru_cell/dropout_5/Const?
 gru/while/gru_cell/dropout_5/MulMul'gru/while/gru_cell/ones_like_1:output:0+gru/while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????52"
 gru/while/gru_cell/dropout_5/Mul?
"gru/while/gru_cell/dropout_5/ShapeShape'gru/while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2$
"gru/while/gru_cell/dropout_5/Shape?
9gru/while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform+gru/while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???2;
9gru/while/gru_cell/dropout_5/random_uniform/RandomUniform?
+gru/while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2-
+gru/while/gru_cell/dropout_5/GreaterEqual/y?
)gru/while/gru_cell/dropout_5/GreaterEqualGreaterEqualBgru/while/gru_cell/dropout_5/random_uniform/RandomUniform:output:04gru/while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52+
)gru/while/gru_cell/dropout_5/GreaterEqual?
!gru/while/gru_cell/dropout_5/CastCast-gru/while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52#
!gru/while/gru_cell/dropout_5/Cast?
"gru/while/gru_cell/dropout_5/Mul_1Mul$gru/while/gru_cell/dropout_5/Mul:z:0%gru/while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????52$
"gru/while/gru_cell/dropout_5/Mul_1?
!gru/while/gru_cell/ReadVariableOpReadVariableOp,gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02#
!gru/while/gru_cell/ReadVariableOp?
gru/while/gru_cell/unstackUnpack)gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/while/gru_cell/unstack?
gru/while/gru_cell/mulMul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0$gru/while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul?
gru/while/gru_cell/mul_1Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0&gru/while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_1?
gru/while/gru_cell/mul_2Mul4gru/while/TensorArrayV2Read/TensorListGetItem:item:0&gru/while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru/while/gru_cell/mul_2?
#gru/while/gru_cell/ReadVariableOp_1ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02%
#gru/while/gru_cell/ReadVariableOp_1?
&gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2(
&gru/while/gru_cell/strided_slice/stack?
(gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2*
(gru/while/gru_cell/strided_slice/stack_1?
(gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2*
(gru/while/gru_cell/strided_slice/stack_2?
 gru/while/gru_cell/strided_sliceStridedSlice+gru/while/gru_cell/ReadVariableOp_1:value:0/gru/while/gru_cell/strided_slice/stack:output:01gru/while/gru_cell/strided_slice/stack_1:output:01gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2"
 gru/while/gru_cell/strided_slice?
gru/while/gru_cell/MatMulMatMulgru/while/gru_cell/mul:z:0)gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/MatMul?
#gru/while/gru_cell/ReadVariableOp_2ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02%
#gru/while/gru_cell/ReadVariableOp_2?
(gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2*
(gru/while/gru_cell/strided_slice_1/stack?
*gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2,
*gru/while/gru_cell/strided_slice_1/stack_1?
*gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_1/stack_2?
"gru/while/gru_cell/strided_slice_1StridedSlice+gru/while/gru_cell/ReadVariableOp_2:value:01gru/while/gru_cell/strided_slice_1/stack:output:03gru/while/gru_cell/strided_slice_1/stack_1:output:03gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_1?
gru/while/gru_cell/MatMul_1MatMulgru/while/gru_cell/mul_1:z:0+gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/MatMul_1?
#gru/while/gru_cell/ReadVariableOp_3ReadVariableOp.gru_while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02%
#gru/while/gru_cell/ReadVariableOp_3?
(gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2*
(gru/while/gru_cell/strided_slice_2/stack?
*gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2,
*gru/while/gru_cell/strided_slice_2/stack_1?
*gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_2/stack_2?
"gru/while/gru_cell/strided_slice_2StridedSlice+gru/while/gru_cell/ReadVariableOp_3:value:01gru/while/gru_cell/strided_slice_2/stack:output:03gru/while/gru_cell/strided_slice_2/stack_1:output:03gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_2?
gru/while/gru_cell/MatMul_2MatMulgru/while/gru_cell/mul_2:z:0+gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/MatMul_2?
(gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_3/stack?
*gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52,
*gru/while/gru_cell/strided_slice_3/stack_1?
*gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_3/stack_2?
"gru/while/gru_cell/strided_slice_3StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_3/stack:output:03gru/while/gru_cell/strided_slice_3/stack_1:output:03gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2$
"gru/while/gru_cell/strided_slice_3?
gru/while/gru_cell/BiasAddBiasAdd#gru/while/gru_cell/MatMul:product:0+gru/while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/BiasAdd?
(gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52*
(gru/while/gru_cell/strided_slice_4/stack?
*gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2,
*gru/while/gru_cell/strided_slice_4/stack_1?
*gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_4/stack_2?
"gru/while/gru_cell/strided_slice_4StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_4/stack:output:03gru/while/gru_cell/strided_slice_4/stack_1:output:03gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52$
"gru/while/gru_cell/strided_slice_4?
gru/while/gru_cell/BiasAdd_1BiasAdd%gru/while/gru_cell/MatMul_1:product:0+gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/BiasAdd_1?
(gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2*
(gru/while/gru_cell/strided_slice_5/stack?
*gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2,
*gru/while/gru_cell/strided_slice_5/stack_1?
*gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_5/stack_2?
"gru/while/gru_cell/strided_slice_5StridedSlice#gru/while/gru_cell/unstack:output:01gru/while/gru_cell/strided_slice_5/stack:output:03gru/while/gru_cell/strided_slice_5/stack_1:output:03gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2$
"gru/while/gru_cell/strided_slice_5?
gru/while/gru_cell/BiasAdd_2BiasAdd%gru/while/gru_cell/MatMul_2:product:0+gru/while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/BiasAdd_2?
gru/while/gru_cell/mul_3Mulgru_while_placeholder_2&gru/while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/mul_3?
gru/while/gru_cell/mul_4Mulgru_while_placeholder_2&gru/while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/mul_4?
gru/while/gru_cell/mul_5Mulgru_while_placeholder_2&gru/while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/mul_5?
#gru/while/gru_cell/ReadVariableOp_4ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02%
#gru/while/gru_cell/ReadVariableOp_4?
(gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(gru/while/gru_cell/strided_slice_6/stack?
*gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2,
*gru/while/gru_cell/strided_slice_6/stack_1?
*gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_6/stack_2?
"gru/while/gru_cell/strided_slice_6StridedSlice+gru/while/gru_cell/ReadVariableOp_4:value:01gru/while/gru_cell/strided_slice_6/stack:output:03gru/while/gru_cell/strided_slice_6/stack_1:output:03gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_6?
gru/while/gru_cell/MatMul_3MatMulgru/while/gru_cell/mul_3:z:0+gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/MatMul_3?
#gru/while/gru_cell/ReadVariableOp_5ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02%
#gru/while/gru_cell/ReadVariableOp_5?
(gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2*
(gru/while/gru_cell/strided_slice_7/stack?
*gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2,
*gru/while/gru_cell/strided_slice_7/stack_1?
*gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*gru/while/gru_cell/strided_slice_7/stack_2?
"gru/while/gru_cell/strided_slice_7StridedSlice+gru/while/gru_cell/ReadVariableOp_5:value:01gru/while/gru_cell/strided_slice_7/stack:output:03gru/while/gru_cell/strided_slice_7/stack_1:output:03gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2$
"gru/while/gru_cell/strided_slice_7?
gru/while/gru_cell/MatMul_4MatMulgru/while/gru_cell/mul_4:z:0+gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/MatMul_4?
(gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gru/while/gru_cell/strided_slice_8/stack?
*gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52,
*gru/while/gru_cell/strided_slice_8/stack_1?
*gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_8/stack_2?
"gru/while/gru_cell/strided_slice_8StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_8/stack:output:03gru/while/gru_cell/strided_slice_8/stack_1:output:03gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2$
"gru/while/gru_cell/strided_slice_8?
gru/while/gru_cell/BiasAdd_3BiasAdd%gru/while/gru_cell/MatMul_3:product:0+gru/while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/BiasAdd_3?
(gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52*
(gru/while/gru_cell/strided_slice_9/stack?
*gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2,
*gru/while/gru_cell/strided_slice_9/stack_1?
*gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gru/while/gru_cell/strided_slice_9/stack_2?
"gru/while/gru_cell/strided_slice_9StridedSlice#gru/while/gru_cell/unstack:output:11gru/while/gru_cell/strided_slice_9/stack:output:03gru/while/gru_cell/strided_slice_9/stack_1:output:03gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52$
"gru/while/gru_cell/strided_slice_9?
gru/while/gru_cell/BiasAdd_4BiasAdd%gru/while/gru_cell/MatMul_4:product:0+gru/while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/BiasAdd_4?
gru/while/gru_cell/addAddV2#gru/while/gru_cell/BiasAdd:output:0%gru/while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/add?
gru/while/gru_cell/SigmoidSigmoidgru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/Sigmoid?
gru/while/gru_cell/add_1AddV2%gru/while/gru_cell/BiasAdd_1:output:0%gru/while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/add_1?
gru/while/gru_cell/Sigmoid_1Sigmoidgru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/Sigmoid_1?
#gru/while/gru_cell/ReadVariableOp_6ReadVariableOp.gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02%
#gru/while/gru_cell/ReadVariableOp_6?
)gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2+
)gru/while/gru_cell/strided_slice_10/stack?
+gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2-
+gru/while/gru_cell/strided_slice_10/stack_1?
+gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2-
+gru/while/gru_cell/strided_slice_10/stack_2?
#gru/while/gru_cell/strided_slice_10StridedSlice+gru/while/gru_cell/ReadVariableOp_6:value:02gru/while/gru_cell/strided_slice_10/stack:output:04gru/while/gru_cell/strided_slice_10/stack_1:output:04gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2%
#gru/while/gru_cell/strided_slice_10?
gru/while/gru_cell/MatMul_5MatMulgru/while/gru_cell/mul_5:z:0,gru/while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/MatMul_5?
)gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2+
)gru/while/gru_cell/strided_slice_11/stack?
+gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2-
+gru/while/gru_cell/strided_slice_11/stack_1?
+gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+gru/while/gru_cell/strided_slice_11/stack_2?
#gru/while/gru_cell/strided_slice_11StridedSlice#gru/while/gru_cell/unstack:output:12gru/while/gru_cell/strided_slice_11/stack:output:04gru/while/gru_cell/strided_slice_11/stack_1:output:04gru/while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2%
#gru/while/gru_cell/strided_slice_11?
gru/while/gru_cell/BiasAdd_5BiasAdd%gru/while/gru_cell/MatMul_5:product:0,gru/while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/BiasAdd_5?
gru/while/gru_cell/mul_6Mul gru/while/gru_cell/Sigmoid_1:y:0%gru/while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/mul_6?
gru/while/gru_cell/add_2AddV2%gru/while/gru_cell/BiasAdd_2:output:0gru/while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/add_2?
gru/while/gru_cell/TanhTanhgru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/Tanh?
gru/while/gru_cell/mul_7Mulgru/while/gru_cell/Sigmoid:y:0gru_while_placeholder_2*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/mul_7y
gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/while/gru_cell/sub/x?
gru/while/gru_cell/subSub!gru/while/gru_cell/sub/x:output:0gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/sub?
gru/while/gru_cell/mul_8Mulgru/while/gru_cell/sub:z:0gru/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/mul_8?
gru/while/gru_cell/add_3AddV2gru/while/gru_cell/mul_7:z:0gru/while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
gru/while/gru_cell/add_3?
.gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_while_placeholder_1gru_while_placeholdergru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype020
.gru/while/TensorArrayV2Write/TensorListSetItemd
gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add/yy
gru/while/addAddV2gru_while_placeholdergru/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/while/addh
gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/while/add_1/y?
gru/while/add_1AddV2 gru_while_gru_while_loop_countergru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/while/add_1?
gru/while/IdentityIdentitygru/while/add_1:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity?
gru/while/Identity_1Identity&gru_while_gru_while_maximum_iterations"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_1?
gru/while/Identity_2Identitygru/while/add:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_2?
gru/while/Identity_3Identity>gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
gru/while/Identity_3?
gru/while/Identity_4Identitygru/while/gru_cell/add_3:z:0"^gru/while/gru_cell/ReadVariableOp$^gru/while/gru_cell/ReadVariableOp_1$^gru/while/gru_cell/ReadVariableOp_2$^gru/while/gru_cell/ReadVariableOp_3$^gru/while/gru_cell/ReadVariableOp_4$^gru/while/gru_cell/ReadVariableOp_5$^gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:?????????52
gru/while/Identity_4"^
,gru_while_gru_cell_readvariableop_1_resource.gru_while_gru_cell_readvariableop_1_resource_0"^
,gru_while_gru_cell_readvariableop_4_resource.gru_while_gru_cell_readvariableop_4_resource_0"Z
*gru_while_gru_cell_readvariableop_resource,gru_while_gru_cell_readvariableop_resource_0"@
gru_while_gru_strided_slice_1gru_while_gru_strided_slice_1_0"1
gru_while_identitygru/while/Identity:output:0"5
gru_while_identity_1gru/while/Identity_1:output:0"5
gru_while_identity_2gru/while/Identity_2:output:0"5
gru_while_identity_3gru/while/Identity_3:output:0"5
gru_while_identity_4gru/while/Identity_4:output:0"?
Ygru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor[gru_while_tensorarrayv2read_tensorlistgetitem_gru_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????5: : :::2F
!gru/while/gru_cell/ReadVariableOp!gru/while/gru_cell/ReadVariableOp2J
#gru/while/gru_cell/ReadVariableOp_1#gru/while/gru_cell/ReadVariableOp_12J
#gru/while/gru_cell/ReadVariableOp_2#gru/while/gru_cell/ReadVariableOp_22J
#gru/while/gru_cell/ReadVariableOp_3#gru/while/gru_cell/ReadVariableOp_32J
#gru/while/gru_cell/ReadVariableOp_4#gru/while/gru_cell/ReadVariableOp_42J
#gru/while/gru_cell/ReadVariableOp_5#gru/while/gru_cell/ReadVariableOp_52J
#gru/while/gru_cell/ReadVariableOp_6#gru/while/gru_cell/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_19981
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_19981___redundant_placeholder03
/while_while_cond_19981___redundant_placeholder13
/while_while_cond_19981___redundant_placeholder23
/while_while_cond_19981___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????5: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
:
?
?
while_cond_22972
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_22972___redundant_placeholder03
/while_while_cond_22972___redundant_placeholder13
/while_while_cond_22972___redundant_placeholder23
/while_while_cond_22972___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????5: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
:
?0
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19277

inputs
assignmovingavg_19252
assignmovingavg_1_19258)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/19252*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_19252*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/19252*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/19252*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_19252AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/19252*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/19258*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_19258*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/19258*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/19258*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_19258AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/19258*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
3__inference_batch_normalization_layer_call_fn_22015

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_192772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?m
?
C__inference_gru_cell_layer_call_and_return_conditional_losses_19605

inputs

states
readvariableop_resource
readvariableop_1_resource
readvariableop_4_resource
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
	ones_like\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
ones_like_1y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack`
mulMulinputsones_like:output:0*
T0*(
_output_shapes
:??????????2
muld
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_1d
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_2?
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????52
MatMul?
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52

MatMul_1?
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52

MatMul_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceunstack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52	
BiasAddx
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_1x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_2e
mul_3Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:?????????52
mul_3e
mul_4Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:?????????52
mul_4e
mul_5Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:?????????52
mul_5
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:	5?*
dtype02
ReadVariableOp_4
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_4:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
strided_slice_6u
MatMul_3MatMul	mul_3:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52

MatMul_3
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource*
_output_shapes
:	5?*
dtype02
ReadVariableOp_5
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_5:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
strided_slice_7u
MatMul_4MatMul	mul_4:z:0strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52

MatMul_4x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
strided_slice_8?
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_3x
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_9/stack|
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2?
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
strided_slice_9?
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????52	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????52
	Sigmoid_1
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes
:	5?*
dtype02
ReadVariableOp_6?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_6:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
strided_slice_10v
MatMul_5MatMul	mul_5:z:0strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52

MatMul_5z
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2?
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
strided_slice_11?
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:?????????52
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????52
Tanh\
mul_7MulSigmoid:y:0states*
T0*'
_output_shapes
:?????????52
mul_7S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????52
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:?????????52
add_3?
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*'
_output_shapes
:?????????52

Identity?

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*'
_output_shapes
:?????????52

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????5:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_6:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????5
 
_user_specified_namestates
ٶ
?
>__inference_gru_layer_call_and_return_conditional_losses_23456
inputs_0$
 gru_cell_readvariableop_resource&
"gru_cell_readvariableop_1_resource&
"gru_cell_readvariableop_4_resource
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :52
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :52
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????52
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2|
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like/Const?
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/ones_likev
gru_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like_1/Shape}
gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like_1/Const?
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/ones_like_1?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/mulMulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1?
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul?
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_2?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_1?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_2?
gru_cell/mul_3Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_3?
gru_cell/mul_4Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_4?
gru_cell/mul_5Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_5?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_4?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru_cell/strided_slice_8?
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_3?
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52 
gru_cell/strided_slice_9/stack?
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2"
 gru_cell/strided_slice_9/stack_1?
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2?
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru_cell/strided_slice_9?
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Sigmoid_1?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2!
gru_cell/strided_slice_10/stack?
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1?
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2?
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_10?
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_5?
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2!
gru_cell/strided_slice_11/stack?
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1?
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2?
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru_cell/strided_slice_11?
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_5?
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_6?
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_8?
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????5: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_23304*
condR
while_cond_23303*8
output_shapes'
%: : : : :?????????5: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????5*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????5*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????52
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*'
_output_shapes
:?????????52

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?!
?
while_body_19982
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_20004_0
while_gru_cell_20006_0
while_gru_cell_20008_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_20004
while_gru_cell_20006
while_gru_cell_20008??&while/gru_cell/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_20004_0while_gru_cell_20006_0while_gru_cell_20008_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????5:?????????5*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_196052(
&while/gru_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????52
while/Identity_4".
while_gru_cell_20004while_gru_cell_20004_0".
while_gru_cell_20006while_gru_cell_20006_0".
while_gru_cell_20008while_gru_cell_20008_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????5: : :::2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
: 
?
?
3__inference_batch_normalization_layer_call_fn_22097

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????5?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_200942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????5?2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????5?::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_20879

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
@__inference_dense_layer_call_and_return_conditional_losses_20852

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:5*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????5::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????5
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_19310

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?0
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22064

inputs
assignmovingavg_22039
assignmovingavg_1_22045)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????5?2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/22039*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_22039*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/22039*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/22039*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_22039AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/22039*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/22045*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_22045*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/22045*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/22045*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_22045AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/22045*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????5?2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????5?2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:?????????5?2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????5?::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
?<
?
>__inference_gru_layer_call_and_return_conditional_losses_20046

inputs
gru_cell_19970
gru_cell_19972
gru_cell_19974
identity?? gru_cell/StatefulPartitionedCall?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :52
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :52
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????52
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2?
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0gru_cell_19970gru_cell_19972gru_cell_19974*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????5:?????????5*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_196052"
 gru_cell/StatefulPartitionedCall?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_19970gru_cell_19972gru_cell_19974*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????5: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_19982*
condR
while_cond_19981*8
output_shapes'
%: : : : :?????????5: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????5*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????5*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????52
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0!^gru_cell/StatefulPartitionedCall^while*
T0*'
_output_shapes
:?????????52

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
??
?
>__inference_gru_layer_call_and_return_conditional_losses_20811

inputs$
 gru_cell_readvariableop_resource&
"gru_cell_readvariableop_1_resource&
"gru_cell_readvariableop_4_resource
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :52
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :52
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????52
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:5??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2|
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like/Const?
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/ones_likev
gru_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like_1/Shape}
gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like_1/Const?
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/ones_like_1?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/mulMulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1?
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul?
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_2?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_1?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_2?
gru_cell/mul_3Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_3?
gru_cell/mul_4Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_4?
gru_cell/mul_5Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_5?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_4?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru_cell/strided_slice_8?
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_3?
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52 
gru_cell/strided_slice_9/stack?
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2"
 gru_cell/strided_slice_9/stack_1?
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2?
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru_cell/strided_slice_9?
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Sigmoid_1?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2!
gru_cell/strided_slice_10/stack?
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1?
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2?
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_10?
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_5?
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2!
gru_cell/strided_slice_11/stack?
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1?
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2?
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru_cell/strided_slice_11?
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_5?
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_6?
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_8?
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????5: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_20659*
condR
while_cond_20658*8
output_shapes'
%: : : : :?????????5: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:5?????????5*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????5*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????552
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*'
_output_shapes
:?????????52

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????5?:::22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
?
?
while_cond_22619
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_22619___redundant_placeholder03
/while_while_cond_22619___redundant_placeholder13
/while_while_cond_22619___redundant_placeholder23
/while_while_cond_22619___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????5: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
:
??
?
>__inference_gru_layer_call_and_return_conditional_losses_22772

inputs$
 gru_cell_readvariableop_resource&
"gru_cell_readvariableop_1_resource&
"gru_cell_readvariableop_4_resource
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :52
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :52
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????52
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:5??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2|
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like/Const?
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/ones_likev
gru_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like_1/Shape}
gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like_1/Const?
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/ones_like_1?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/mulMulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1?
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul?
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_2?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_1?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_2?
gru_cell/mul_3Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_3?
gru_cell/mul_4Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_4?
gru_cell/mul_5Mulzeros:output:0gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_5?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_4?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru_cell/strided_slice_8?
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_3?
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52 
gru_cell/strided_slice_9/stack?
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2"
 gru_cell/strided_slice_9/stack_1?
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2?
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru_cell/strided_slice_9?
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Sigmoid_1?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2!
gru_cell/strided_slice_10/stack?
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1?
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2?
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_10?
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_5?
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2!
gru_cell/strided_slice_11/stack?
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1?
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2?
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru_cell/strided_slice_11?
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_5?
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_6?
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_8?
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????5: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_22620*
condR
while_cond_22619*8
output_shapes'
%: : : : :?????????5: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:5?????????5*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????5*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????552
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*'
_output_shapes
:?????????52

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????5?:::22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_20114

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????5?2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????5?2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:?????????5?2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????5?::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
?	
?
*__inference_sequential_layer_call_fn_21915

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_209962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????5?:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
??
?
E__inference_sequential_layer_call_and_return_conditional_losses_21564

inputs-
)batch_normalization_assignmovingavg_21143/
+batch_normalization_assignmovingavg_1_21149=
9batch_normalization_batchnorm_mul_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource(
$gru_gru_cell_readvariableop_resource*
&gru_gru_cell_readvariableop_1_resource*
&gru_gru_cell_readvariableop_4_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identity??7batch_normalization/AssignMovingAvg/AssignSubVariableOp?2batch_normalization/AssignMovingAvg/ReadVariableOp?9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp?4batch_normalization/AssignMovingAvg_1/ReadVariableOp?,batch_normalization/batchnorm/ReadVariableOp?0batch_normalization/batchnorm/mul/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?gru/gru_cell/ReadVariableOp?gru/gru_cell/ReadVariableOp_1?gru/gru_cell/ReadVariableOp_2?gru/gru_cell/ReadVariableOp_3?gru/gru_cell/ReadVariableOp_4?gru/gru_cell/ReadVariableOp_5?gru/gru_cell/ReadVariableOp_6?	gru/while?
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indices?
 batch_normalization/moments/meanMeaninputs;batch_normalization/moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2"
 batch_normalization/moments/mean?
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*#
_output_shapes
:?2*
(batch_normalization/moments/StopGradient?
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceinputs1batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:?????????5?2/
-batch_normalization/moments/SquaredDifference?
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       28
6batch_normalization/moments/variance/reduction_indices?
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2&
$batch_normalization/moments/variance?
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze?
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1?
)batch_normalization/AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/21143*
_output_shapes
: *
dtype0*
valueB
 *
?#<2+
)batch_normalization/AssignMovingAvg/decay?
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp)batch_normalization_assignmovingavg_21143*
_output_shapes	
:?*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp?
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/21143*
_output_shapes	
:?2)
'batch_normalization/AssignMovingAvg/sub?
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/21143*
_output_shapes	
:?2)
'batch_normalization/AssignMovingAvg/mul?
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp)batch_normalization_assignmovingavg_21143+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/21143*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOp?
+batch_normalization/AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/21149*
_output_shapes
: *
dtype0*
valueB
 *
?#<2-
+batch_normalization/AssignMovingAvg_1/decay?
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp+batch_normalization_assignmovingavg_1_21149*
_output_shapes	
:?*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp?
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/21149*
_output_shapes	
:?2+
)batch_normalization/AssignMovingAvg_1/sub?
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/21149*
_output_shapes	
:?2+
)batch_normalization/AssignMovingAvg_1/mul?
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp+batch_normalization_assignmovingavg_1_21149-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/21149*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp?
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2%
#batch_normalization/batchnorm/add/y?
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/add?
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/Rsqrt?
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp?
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/mul?
#batch_normalization/batchnorm/mul_1Mulinputs%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????5?2%
#batch_normalization/batchnorm/mul_1?
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?2%
#batch_normalization/batchnorm/mul_2?
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp?
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2#
!batch_normalization/batchnorm/sub?
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????5?2%
#batch_normalization/batchnorm/add_1m
	gru/ShapeShape'batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
	gru/Shape|
gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice/stack?
gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_1?
gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice/stack_2?
gru/strided_sliceStridedSlicegru/Shape:output:0 gru/strided_slice/stack:output:0"gru/strided_slice/stack_1:output:0"gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_sliced
gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :52
gru/zeros/mul/y|
gru/zeros/mulMulgru/strided_slice:output:0gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/mulg
gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
gru/zeros/Less/yw
gru/zeros/LessLessgru/zeros/mul:z:0gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
gru/zeros/Lessj
gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :52
gru/zeros/packed/1?
gru/zeros/packedPackgru/strided_slice:output:0gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
gru/zeros/packedg
gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/zeros/Const?
	gru/zerosFillgru/zeros/packed:output:0gru/zeros/Const:output:0*
T0*'
_output_shapes
:?????????52
	gru/zeros}
gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose/perm?
gru/transpose	Transpose'batch_normalization/batchnorm/add_1:z:0gru/transpose/perm:output:0*
T0*,
_output_shapes
:5??????????2
gru/transpose[
gru/Shape_1Shapegru/transpose:y:0*
T0*
_output_shapes
:2
gru/Shape_1?
gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_1/stack?
gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_1?
gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_1/stack_2?
gru/strided_slice_1StridedSlicegru/Shape_1:output:0"gru/strided_slice_1/stack:output:0$gru/strided_slice_1/stack_1:output:0$gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/strided_slice_1?
gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
gru/TensorArrayV2/element_shape?
gru/TensorArrayV2TensorListReserve(gru/TensorArrayV2/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2?
9gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2;
9gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
+gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/transpose:y:0Bgru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+gru/TensorArrayUnstack/TensorListFromTensor?
gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_2/stack?
gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_1?
gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_2/stack_2?
gru/strided_slice_2StridedSlicegru/transpose:y:0"gru/strided_slice_2/stack:output:0$gru/strided_slice_2/stack_1:output:0$gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
gru/strided_slice_2?
gru/gru_cell/ones_like/ShapeShapegru/strided_slice_2:output:0*
T0*
_output_shapes
:2
gru/gru_cell/ones_like/Shape?
gru/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/ones_like/Const?
gru/gru_cell/ones_likeFill%gru/gru_cell/ones_like/Shape:output:0%gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/ones_like}
gru/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
gru/gru_cell/dropout/Const?
gru/gru_cell/dropout/MulMulgru/gru_cell/ones_like:output:0#gru/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/dropout/Mul?
gru/gru_cell/dropout/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout/Shape?
1gru/gru_cell/dropout/random_uniform/RandomUniformRandomUniform#gru/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???23
1gru/gru_cell/dropout/random_uniform/RandomUniform?
#gru/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2%
#gru/gru_cell/dropout/GreaterEqual/y?
!gru/gru_cell/dropout/GreaterEqualGreaterEqual:gru/gru_cell/dropout/random_uniform/RandomUniform:output:0,gru/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2#
!gru/gru_cell/dropout/GreaterEqual?
gru/gru_cell/dropout/CastCast%gru/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru/gru_cell/dropout/Cast?
gru/gru_cell/dropout/Mul_1Mulgru/gru_cell/dropout/Mul:z:0gru/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/dropout/Mul_1?
gru/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
gru/gru_cell/dropout_1/Const?
gru/gru_cell/dropout_1/MulMulgru/gru_cell/ones_like:output:0%gru/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/dropout_1/Mul?
gru/gru_cell/dropout_1/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_1/Shape?
3gru/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???25
3gru/gru_cell/dropout_1/random_uniform/RandomUniform?
%gru/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%gru/gru_cell/dropout_1/GreaterEqual/y?
#gru/gru_cell/dropout_1/GreaterEqualGreaterEqual<gru/gru_cell/dropout_1/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2%
#gru/gru_cell/dropout_1/GreaterEqual?
gru/gru_cell/dropout_1/CastCast'gru/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru/gru_cell/dropout_1/Cast?
gru/gru_cell/dropout_1/Mul_1Mulgru/gru_cell/dropout_1/Mul:z:0gru/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/dropout_1/Mul_1?
gru/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
gru/gru_cell/dropout_2/Const?
gru/gru_cell/dropout_2/MulMulgru/gru_cell/ones_like:output:0%gru/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/dropout_2/Mul?
gru/gru_cell/dropout_2/ShapeShapegru/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_2/Shape?
3gru/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ߏ?25
3gru/gru_cell/dropout_2/random_uniform/RandomUniform?
%gru/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%gru/gru_cell/dropout_2/GreaterEqual/y?
#gru/gru_cell/dropout_2/GreaterEqualGreaterEqual<gru/gru_cell/dropout_2/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2%
#gru/gru_cell/dropout_2/GreaterEqual?
gru/gru_cell/dropout_2/CastCast'gru/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru/gru_cell/dropout_2/Cast?
gru/gru_cell/dropout_2/Mul_1Mulgru/gru_cell/dropout_2/Mul:z:0gru/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/dropout_2/Mul_1?
gru/gru_cell/ones_like_1/ShapeShapegru/zeros:output:0*
T0*
_output_shapes
:2 
gru/gru_cell/ones_like_1/Shape?
gru/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
gru/gru_cell/ones_like_1/Const?
gru/gru_cell/ones_like_1Fill'gru/gru_cell/ones_like_1/Shape:output:0'gru/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/ones_like_1?
gru/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/dropout_3/Const?
gru/gru_cell/dropout_3/MulMul!gru/gru_cell/ones_like_1:output:0%gru/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/dropout_3/Mul?
gru/gru_cell/dropout_3/ShapeShape!gru/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_3/Shape?
3gru/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???25
3gru/gru_cell/dropout_3/random_uniform/RandomUniform?
%gru/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%gru/gru_cell/dropout_3/GreaterEqual/y?
#gru/gru_cell/dropout_3/GreaterEqualGreaterEqual<gru/gru_cell/dropout_3/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52%
#gru/gru_cell/dropout_3/GreaterEqual?
gru/gru_cell/dropout_3/CastCast'gru/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
gru/gru_cell/dropout_3/Cast?
gru/gru_cell/dropout_3/Mul_1Mulgru/gru_cell/dropout_3/Mul:z:0gru/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/dropout_3/Mul_1?
gru/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/dropout_4/Const?
gru/gru_cell/dropout_4/MulMul!gru/gru_cell/ones_like_1:output:0%gru/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/dropout_4/Mul?
gru/gru_cell/dropout_4/ShapeShape!gru/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_4/Shape?
3gru/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???25
3gru/gru_cell/dropout_4/random_uniform/RandomUniform?
%gru/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%gru/gru_cell/dropout_4/GreaterEqual/y?
#gru/gru_cell/dropout_4/GreaterEqualGreaterEqual<gru/gru_cell/dropout_4/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52%
#gru/gru_cell/dropout_4/GreaterEqual?
gru/gru_cell/dropout_4/CastCast'gru/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
gru/gru_cell/dropout_4/Cast?
gru/gru_cell/dropout_4/Mul_1Mulgru/gru_cell/dropout_4/Mul:z:0gru/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/dropout_4/Mul_1?
gru/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/dropout_5/Const?
gru/gru_cell/dropout_5/MulMul!gru/gru_cell/ones_like_1:output:0%gru/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/dropout_5/Mul?
gru/gru_cell/dropout_5/ShapeShape!gru/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru/gru_cell/dropout_5/Shape?
3gru/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform%gru/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???25
3gru/gru_cell/dropout_5/random_uniform/RandomUniform?
%gru/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2'
%gru/gru_cell/dropout_5/GreaterEqual/y?
#gru/gru_cell/dropout_5/GreaterEqualGreaterEqual<gru/gru_cell/dropout_5/random_uniform/RandomUniform:output:0.gru/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52%
#gru/gru_cell/dropout_5/GreaterEqual?
gru/gru_cell/dropout_5/CastCast'gru/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
gru/gru_cell/dropout_5/Cast?
gru/gru_cell/dropout_5/Mul_1Mulgru/gru_cell/dropout_5/Mul:z:0gru/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/dropout_5/Mul_1?
gru/gru_cell/ReadVariableOpReadVariableOp$gru_gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru/gru_cell/ReadVariableOp?
gru/gru_cell/unstackUnpack#gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru/gru_cell/unstack?
gru/gru_cell/mulMulgru/strided_slice_2:output:0gru/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul?
gru/gru_cell/mul_1Mulgru/strided_slice_2:output:0 gru/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_1?
gru/gru_cell/mul_2Mulgru/strided_slice_2:output:0 gru/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru/gru_cell/mul_2?
gru/gru_cell/ReadVariableOp_1ReadVariableOp&gru_gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru/gru_cell/ReadVariableOp_1?
 gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2"
 gru/gru_cell/strided_slice/stack?
"gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2$
"gru/gru_cell/strided_slice/stack_1?
"gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2$
"gru/gru_cell/strided_slice/stack_2?
gru/gru_cell/strided_sliceStridedSlice%gru/gru_cell/ReadVariableOp_1:value:0)gru/gru_cell/strided_slice/stack:output:0+gru/gru_cell/strided_slice/stack_1:output:0+gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru/gru_cell/strided_slice?
gru/gru_cell/MatMulMatMulgru/gru_cell/mul:z:0#gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/MatMul?
gru/gru_cell/ReadVariableOp_2ReadVariableOp&gru_gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru/gru_cell/ReadVariableOp_2?
"gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2$
"gru/gru_cell/strided_slice_1/stack?
$gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2&
$gru/gru_cell/strided_slice_1/stack_1?
$gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_1/stack_2?
gru/gru_cell/strided_slice_1StridedSlice%gru/gru_cell/ReadVariableOp_2:value:0+gru/gru_cell/strided_slice_1/stack:output:0-gru/gru_cell/strided_slice_1/stack_1:output:0-gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_1?
gru/gru_cell/MatMul_1MatMulgru/gru_cell/mul_1:z:0%gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/MatMul_1?
gru/gru_cell/ReadVariableOp_3ReadVariableOp&gru_gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru/gru_cell/ReadVariableOp_3?
"gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2$
"gru/gru_cell/strided_slice_2/stack?
$gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2&
$gru/gru_cell/strided_slice_2/stack_1?
$gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_2/stack_2?
gru/gru_cell/strided_slice_2StridedSlice%gru/gru_cell/ReadVariableOp_3:value:0+gru/gru_cell/strided_slice_2/stack:output:0-gru/gru_cell/strided_slice_2/stack_1:output:0-gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_2?
gru/gru_cell/MatMul_2MatMulgru/gru_cell/mul_2:z:0%gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/MatMul_2?
"gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_3/stack?
$gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52&
$gru/gru_cell/strided_slice_3/stack_1?
$gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_3/stack_2?
gru/gru_cell/strided_slice_3StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_3/stack:output:0-gru/gru_cell/strided_slice_3/stack_1:output:0-gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru/gru_cell/strided_slice_3?
gru/gru_cell/BiasAddBiasAddgru/gru_cell/MatMul:product:0%gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/BiasAdd?
"gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52$
"gru/gru_cell/strided_slice_4/stack?
$gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2&
$gru/gru_cell/strided_slice_4/stack_1?
$gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_4/stack_2?
gru/gru_cell/strided_slice_4StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_4/stack:output:0-gru/gru_cell/strided_slice_4/stack_1:output:0-gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru/gru_cell/strided_slice_4?
gru/gru_cell/BiasAdd_1BiasAddgru/gru_cell/MatMul_1:product:0%gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/BiasAdd_1?
"gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2$
"gru/gru_cell/strided_slice_5/stack?
$gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$gru/gru_cell/strided_slice_5/stack_1?
$gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_5/stack_2?
gru/gru_cell/strided_slice_5StridedSlicegru/gru_cell/unstack:output:0+gru/gru_cell/strided_slice_5/stack:output:0-gru/gru_cell/strided_slice_5/stack_1:output:0-gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru/gru_cell/strided_slice_5?
gru/gru_cell/BiasAdd_2BiasAddgru/gru_cell/MatMul_2:product:0%gru/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/BiasAdd_2?
gru/gru_cell/mul_3Mulgru/zeros:output:0 gru/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/mul_3?
gru/gru_cell/mul_4Mulgru/zeros:output:0 gru/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/mul_4?
gru/gru_cell/mul_5Mulgru/zeros:output:0 gru/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/mul_5?
gru/gru_cell/ReadVariableOp_4ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru/gru_cell/ReadVariableOp_4?
"gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"gru/gru_cell/strided_slice_6/stack?
$gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2&
$gru/gru_cell/strided_slice_6/stack_1?
$gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_6/stack_2?
gru/gru_cell/strided_slice_6StridedSlice%gru/gru_cell/ReadVariableOp_4:value:0+gru/gru_cell/strided_slice_6/stack:output:0-gru/gru_cell/strided_slice_6/stack_1:output:0-gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_6?
gru/gru_cell/MatMul_3MatMulgru/gru_cell/mul_3:z:0%gru/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/MatMul_3?
gru/gru_cell/ReadVariableOp_5ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru/gru_cell/ReadVariableOp_5?
"gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2$
"gru/gru_cell/strided_slice_7/stack?
$gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2&
$gru/gru_cell/strided_slice_7/stack_1?
$gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$gru/gru_cell/strided_slice_7/stack_2?
gru/gru_cell/strided_slice_7StridedSlice%gru/gru_cell/ReadVariableOp_5:value:0+gru/gru_cell/strided_slice_7/stack:output:0-gru/gru_cell/strided_slice_7/stack_1:output:0-gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_7?
gru/gru_cell/MatMul_4MatMulgru/gru_cell/mul_4:z:0%gru/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/MatMul_4?
"gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"gru/gru_cell/strided_slice_8/stack?
$gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52&
$gru/gru_cell/strided_slice_8/stack_1?
$gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_8/stack_2?
gru/gru_cell/strided_slice_8StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_8/stack:output:0-gru/gru_cell/strided_slice_8/stack_1:output:0-gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru/gru_cell/strided_slice_8?
gru/gru_cell/BiasAdd_3BiasAddgru/gru_cell/MatMul_3:product:0%gru/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/BiasAdd_3?
"gru/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52$
"gru/gru_cell/strided_slice_9/stack?
$gru/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2&
$gru/gru_cell/strided_slice_9/stack_1?
$gru/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$gru/gru_cell/strided_slice_9/stack_2?
gru/gru_cell/strided_slice_9StridedSlicegru/gru_cell/unstack:output:1+gru/gru_cell/strided_slice_9/stack:output:0-gru/gru_cell/strided_slice_9/stack_1:output:0-gru/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru/gru_cell/strided_slice_9?
gru/gru_cell/BiasAdd_4BiasAddgru/gru_cell/MatMul_4:product:0%gru/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/BiasAdd_4?
gru/gru_cell/addAddV2gru/gru_cell/BiasAdd:output:0gru/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/add
gru/gru_cell/SigmoidSigmoidgru/gru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/Sigmoid?
gru/gru_cell/add_1AddV2gru/gru_cell/BiasAdd_1:output:0gru/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/add_1?
gru/gru_cell/Sigmoid_1Sigmoidgru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/Sigmoid_1?
gru/gru_cell/ReadVariableOp_6ReadVariableOp&gru_gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru/gru_cell/ReadVariableOp_6?
#gru/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2%
#gru/gru_cell/strided_slice_10/stack?
%gru/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%gru/gru_cell/strided_slice_10/stack_1?
%gru/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%gru/gru_cell/strided_slice_10/stack_2?
gru/gru_cell/strided_slice_10StridedSlice%gru/gru_cell/ReadVariableOp_6:value:0,gru/gru_cell/strided_slice_10/stack:output:0.gru/gru_cell/strided_slice_10/stack_1:output:0.gru/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru/gru_cell/strided_slice_10?
gru/gru_cell/MatMul_5MatMulgru/gru_cell/mul_5:z:0&gru/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/MatMul_5?
#gru/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2%
#gru/gru_cell/strided_slice_11/stack?
%gru/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2'
%gru/gru_cell/strided_slice_11/stack_1?
%gru/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%gru/gru_cell/strided_slice_11/stack_2?
gru/gru_cell/strided_slice_11StridedSlicegru/gru_cell/unstack:output:1,gru/gru_cell/strided_slice_11/stack:output:0.gru/gru_cell/strided_slice_11/stack_1:output:0.gru/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru/gru_cell/strided_slice_11?
gru/gru_cell/BiasAdd_5BiasAddgru/gru_cell/MatMul_5:product:0&gru/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/BiasAdd_5?
gru/gru_cell/mul_6Mulgru/gru_cell/Sigmoid_1:y:0gru/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/mul_6?
gru/gru_cell/add_2AddV2gru/gru_cell/BiasAdd_2:output:0gru/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/add_2x
gru/gru_cell/TanhTanhgru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/Tanh?
gru/gru_cell/mul_7Mulgru/gru_cell/Sigmoid:y:0gru/zeros:output:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/mul_7m
gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru/gru_cell/sub/x?
gru/gru_cell/subSubgru/gru_cell/sub/x:output:0gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/sub?
gru/gru_cell/mul_8Mulgru/gru_cell/sub:z:0gru/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/mul_8?
gru/gru_cell/add_3AddV2gru/gru_cell/mul_7:z:0gru/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
gru/gru_cell/add_3?
!gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   2#
!gru/TensorArrayV2_1/element_shape?
gru/TensorArrayV2_1TensorListReserve*gru/TensorArrayV2_1/element_shape:output:0gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/TensorArrayV2_1V
gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2

gru/time?
gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru/while/maximum_iterationsr
gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/while/loop_counter?
	gru/whileWhilegru/while/loop_counter:output:0%gru/while/maximum_iterations:output:0gru/time:output:0gru/TensorArrayV2_1:handle:0gru/zeros:output:0gru/strided_slice_1:output:0;gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0$gru_gru_cell_readvariableop_resource&gru_gru_cell_readvariableop_1_resource&gru_gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????5: : : : : *%
_read_only_resource_inputs
	* 
bodyR
gru_while_body_21343* 
condR
gru_while_cond_21342*8
output_shapes'
%: : : : :?????????5: : : : : *
parallel_iterations 2
	gru/while?
4gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   26
4gru/TensorArrayV2Stack/TensorListStack/element_shape?
&gru/TensorArrayV2Stack/TensorListStackTensorListStackgru/while:output:3=gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:5?????????5*
element_dtype02(
&gru/TensorArrayV2Stack/TensorListStack?
gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
gru/strided_slice_3/stack?
gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
gru/strided_slice_3/stack_1?
gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
gru/strided_slice_3/stack_2?
gru/strided_slice_3StridedSlice/gru/TensorArrayV2Stack/TensorListStack:tensor:0"gru/strided_slice_3/stack:output:0$gru/strided_slice_3/stack_1:output:0$gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????5*
shrink_axis_mask2
gru/strided_slice_3?
gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/transpose_1/perm?
gru/transpose_1	Transpose/gru/TensorArrayV2Stack/TensorListStack:tensor:0gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????552
gru/transpose_1n
gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/runtime?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:5*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulgru/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdds
dense/SigmoidSigmoiddense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense/Sigmoid?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Sigmoid:y:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddy
dense_1/SigmoidSigmoiddense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_1/Sigmoid?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Sigmoid:y:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAddy
dense_2/SigmoidSigmoiddense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_2/Sigmoid?
IdentityIdentitydense_2/Sigmoid:y:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp-^batch_normalization/batchnorm/ReadVariableOp1^batch_normalization/batchnorm/mul/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp^gru/gru_cell/ReadVariableOp^gru/gru_cell/ReadVariableOp_1^gru/gru_cell/ReadVariableOp_2^gru/gru_cell/ReadVariableOp_3^gru/gru_cell/ReadVariableOp_4^gru/gru_cell/ReadVariableOp_5^gru/gru_cell/ReadVariableOp_6
^gru/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????5?:::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2\
,batch_normalization/batchnorm/ReadVariableOp,batch_normalization/batchnorm/ReadVariableOp2d
0batch_normalization/batchnorm/mul/ReadVariableOp0batch_normalization/batchnorm/mul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2:
gru/gru_cell/ReadVariableOpgru/gru_cell/ReadVariableOp2>
gru/gru_cell/ReadVariableOp_1gru/gru_cell/ReadVariableOp_12>
gru/gru_cell/ReadVariableOp_2gru/gru_cell/ReadVariableOp_22>
gru/gru_cell/ReadVariableOp_3gru/gru_cell/ReadVariableOp_32>
gru/gru_cell/ReadVariableOp_4gru/gru_cell/ReadVariableOp_42>
gru/gru_cell/ReadVariableOp_5gru/gru_cell/ReadVariableOp_52>
gru/gru_cell/ReadVariableOp_6gru/gru_cell/ReadVariableOp_62
	gru/while	gru/while:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_23529

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
z
%__inference_dense_layer_call_fn_23498

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_208522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????5::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????5
 
_user_specified_nameinputs
??
?
>__inference_gru_layer_call_and_return_conditional_losses_20528

inputs$
 gru_cell_readvariableop_resource&
"gru_cell_readvariableop_1_resource&
"gru_cell_readvariableop_4_resource
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :52
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :52
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????52
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm{
	transpose	Transposeinputstranspose/perm:output:0*
T0*,
_output_shapes
:5??????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2|
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like/Const?
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
gru_cell/dropout/Const?
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shape?
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2՟?2/
-gru_cell/dropout/random_uniform/RandomUniform?
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2!
gru_cell/dropout/GreaterEqual/y?
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/GreaterEqual?
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout/Cast?
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
gru_cell/dropout_1/Const?
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shape?
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???21
/gru_cell/dropout_1/random_uniform/RandomUniform?
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!gru_cell/dropout_1/GreaterEqual/y?
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell/dropout_1/GreaterEqual?
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout_1/Cast?
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
gru_cell/dropout_2/Const?
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shape?
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2ẝ21
/gru_cell/dropout_2/random_uniform/RandomUniform?
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!gru_cell/dropout_2/GreaterEqual/y?
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell/dropout_2/GreaterEqual?
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout_2/Cast?
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_2/Mul_1v
gru_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like_1/Shape}
gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like_1/Const?
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/ones_like_1y
gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/dropout_3/Const?
gru_cell/dropout_3/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_3/Mul?
gru_cell/dropout_3/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_3/Shape?
/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???21
/gru_cell/dropout_3/random_uniform/RandomUniform?
!gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!gru_cell/dropout_3/GreaterEqual/y?
gru_cell/dropout_3/GreaterEqualGreaterEqual8gru_cell/dropout_3/random_uniform/RandomUniform:output:0*gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52!
gru_cell/dropout_3/GreaterEqual?
gru_cell/dropout_3/CastCast#gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
gru_cell/dropout_3/Cast?
gru_cell/dropout_3/Mul_1Mulgru_cell/dropout_3/Mul:z:0gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_3/Mul_1y
gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/dropout_4/Const?
gru_cell/dropout_4/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_4/Mul?
gru_cell/dropout_4/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_4/Shape?
/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2?Ǚ21
/gru_cell/dropout_4/random_uniform/RandomUniform?
!gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!gru_cell/dropout_4/GreaterEqual/y?
gru_cell/dropout_4/GreaterEqualGreaterEqual8gru_cell/dropout_4/random_uniform/RandomUniform:output:0*gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52!
gru_cell/dropout_4/GreaterEqual?
gru_cell/dropout_4/CastCast#gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
gru_cell/dropout_4/Cast?
gru_cell/dropout_4/Mul_1Mulgru_cell/dropout_4/Mul:z:0gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_4/Mul_1y
gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/dropout_5/Const?
gru_cell/dropout_5/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_5/Mul?
gru_cell/dropout_5/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_5/Shape?
/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2?ؐ21
/gru_cell/dropout_5/random_uniform/RandomUniform?
!gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!gru_cell/dropout_5/GreaterEqual/y?
gru_cell/dropout_5/GreaterEqualGreaterEqual8gru_cell/dropout_5/random_uniform/RandomUniform:output:0*gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52!
gru_cell/dropout_5/GreaterEqual?
gru_cell/dropout_5/CastCast#gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
gru_cell/dropout_5/Cast?
gru_cell/dropout_5/Mul_1Mulgru_cell/dropout_5/Mul:z:0gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_5/Mul_1?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/mulMulstrided_slice_2:output:0gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1?
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul?
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_2?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_1?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_2?
gru_cell/mul_3Mulzeros:output:0gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_3?
gru_cell/mul_4Mulzeros:output:0gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_4?
gru_cell/mul_5Mulzeros:output:0gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_5?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_4?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru_cell/strided_slice_8?
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_3?
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52 
gru_cell/strided_slice_9/stack?
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2"
 gru_cell/strided_slice_9/stack_1?
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2?
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru_cell/strided_slice_9?
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Sigmoid_1?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2!
gru_cell/strided_slice_10/stack?
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1?
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2?
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_10?
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_5?
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2!
gru_cell/strided_slice_11/stack?
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1?
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2?
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru_cell/strided_slice_11?
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_5?
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_6?
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_8?
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????5: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_20328*
condR
while_cond_20327*8
output_shapes'
%: : : : :?????????5: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:5?????????5*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????5*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????552
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*'
_output_shapes
:?????????52

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????5?:::22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
?
?
while_cond_22288
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_22288___redundant_placeholder03
/while_while_cond_22288___redundant_placeholder13
/while_while_cond_22288___redundant_placeholder23
/while_while_cond_22288___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????5: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
:
?
?
3__inference_batch_normalization_layer_call_fn_22110

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????5?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_201142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:?????????5?2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????5?::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
?
?
#__inference_gru_layer_call_fn_23467
inputs_0
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_199282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????52

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?	
?
B__inference_dense_1_layer_call_and_return_conditional_losses_23509

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
while_body_22973
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_0.
*while_gru_cell_readvariableop_1_resource_0.
*while_gru_cell_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource,
(while_gru_cell_readvariableop_1_resource,
(while_gru_cell_readvariableop_4_resource??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape?
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/ones_like/Const?
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/ones_like?
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
while/gru_cell/dropout/Const?
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout/Mul?
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shape?
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???25
3while/gru_cell/dropout/random_uniform/RandomUniform?
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%while/gru_cell/dropout/GreaterEqual/y?
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2%
#while/gru_cell/dropout/GreaterEqual?
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout/Cast?
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout/Mul_1?
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2 
while/gru_cell/dropout_1/Const?
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout_1/Mul?
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/Shape?
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_1/random_uniform/RandomUniform?
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2)
'while/gru_cell/dropout_1/GreaterEqual/y?
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell/dropout_1/GreaterEqual?
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout_1/Cast?
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell/dropout_1/Mul_1?
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2 
while/gru_cell/dropout_2/Const?
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout_2/Mul?
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/Shape?
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_2/random_uniform/RandomUniform?
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2)
'while/gru_cell/dropout_2/GreaterEqual/y?
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell/dropout_2/GreaterEqual?
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout_2/Cast?
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell/dropout_2/Mul_1?
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/Shape?
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell/ones_like_1/Const?
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/ones_like_1?
while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/dropout_3/Const?
while/gru_cell/dropout_3/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/dropout_3/Mul?
while/gru_cell/dropout_3/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_3/Shape?
5while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_3/random_uniform/RandomUniform?
'while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2)
'while/gru_cell/dropout_3/GreaterEqual/y?
%while/gru_cell/dropout_3/GreaterEqualGreaterEqual>while/gru_cell/dropout_3/random_uniform/RandomUniform:output:00while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52'
%while/gru_cell/dropout_3/GreaterEqual?
while/gru_cell/dropout_3/CastCast)while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
while/gru_cell/dropout_3/Cast?
while/gru_cell/dropout_3/Mul_1Mul while/gru_cell/dropout_3/Mul:z:0!while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????52 
while/gru_cell/dropout_3/Mul_1?
while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/dropout_4/Const?
while/gru_cell/dropout_4/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/dropout_4/Mul?
while/gru_cell/dropout_4/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_4/Shape?
5while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_4/random_uniform/RandomUniform?
'while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2)
'while/gru_cell/dropout_4/GreaterEqual/y?
%while/gru_cell/dropout_4/GreaterEqualGreaterEqual>while/gru_cell/dropout_4/random_uniform/RandomUniform:output:00while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52'
%while/gru_cell/dropout_4/GreaterEqual?
while/gru_cell/dropout_4/CastCast)while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
while/gru_cell/dropout_4/Cast?
while/gru_cell/dropout_4/Mul_1Mul while/gru_cell/dropout_4/Mul:z:0!while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????52 
while/gru_cell/dropout_4/Mul_1?
while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/dropout_5/Const?
while/gru_cell/dropout_5/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/dropout_5/Mul?
while/gru_cell/dropout_5/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_5/Shape?
5while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_5/random_uniform/RandomUniform?
'while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2)
'while/gru_cell/dropout_5/GreaterEqual/y?
%while/gru_cell/dropout_5/GreaterEqualGreaterEqual>while/gru_cell/dropout_5/random_uniform/RandomUniform:output:00while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52'
%while/gru_cell/dropout_5/GreaterEqual?
while/gru_cell/dropout_5/CastCast)while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
while/gru_cell/dropout_5/Cast?
while/gru_cell/dropout_5/Mul_1Mul while/gru_cell/dropout_5/Mul:z:0!while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????52 
while/gru_cell/dropout_5/Mul_1?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1?
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_1?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_2?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_1?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_2?
while/gru_cell/mul_3Mulwhile_placeholder_2"while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_3?
while/gru_cell/mul_4Mulwhile_placeholder_2"while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_4?
while/gru_cell/mul_5Mulwhile_placeholder_2"while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_5?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_4?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_3?
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52&
$while/gru_cell/strided_slice_9/stack?
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2(
&while/gru_cell/strided_slice_9/stack_1?
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2?
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52 
while/gru_cell/strided_slice_9?
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Sigmoid_1?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_6?
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2'
%while/gru_cell/strided_slice_10/stack?
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1?
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2?
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_5?
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2'
%while/gru_cell/strided_slice_11/stack?
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1?
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2?
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2!
while/gru_cell/strided_slice_11?
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_5?
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_6?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Tanh?
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/sub?
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_8?
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:?????????52
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????5: : :::2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
: 
?
|
'__inference_dense_1_layer_call_fn_23518

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_208792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
*__inference_sequential_layer_call_fn_21091
batch_normalization_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_210622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????5?:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:g c
,
_output_shapes
:?????????5?
3
_user_specified_namebatch_normalization_input
?	
?
*__inference_sequential_layer_call_fn_21946

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*/
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_210622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????5?:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
??
?
 __inference__wrapped_model_19181
batch_normalization_inputD
@sequential_batch_normalization_batchnorm_readvariableop_resourceH
Dsequential_batch_normalization_batchnorm_mul_readvariableop_resourceF
Bsequential_batch_normalization_batchnorm_readvariableop_1_resourceF
Bsequential_batch_normalization_batchnorm_readvariableop_2_resource3
/sequential_gru_gru_cell_readvariableop_resource5
1sequential_gru_gru_cell_readvariableop_1_resource5
1sequential_gru_gru_cell_readvariableop_4_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource5
1sequential_dense_2_matmul_readvariableop_resource6
2sequential_dense_2_biasadd_readvariableop_resource
identity??7sequential/batch_normalization/batchnorm/ReadVariableOp?9sequential/batch_normalization/batchnorm/ReadVariableOp_1?9sequential/batch_normalization/batchnorm/ReadVariableOp_2?;sequential/batch_normalization/batchnorm/mul/ReadVariableOp?'sequential/dense/BiasAdd/ReadVariableOp?&sequential/dense/MatMul/ReadVariableOp?)sequential/dense_1/BiasAdd/ReadVariableOp?(sequential/dense_1/MatMul/ReadVariableOp?)sequential/dense_2/BiasAdd/ReadVariableOp?(sequential/dense_2/MatMul/ReadVariableOp?&sequential/gru/gru_cell/ReadVariableOp?(sequential/gru/gru_cell/ReadVariableOp_1?(sequential/gru/gru_cell/ReadVariableOp_2?(sequential/gru/gru_cell/ReadVariableOp_3?(sequential/gru/gru_cell/ReadVariableOp_4?(sequential/gru/gru_cell/ReadVariableOp_5?(sequential/gru/gru_cell/ReadVariableOp_6?sequential/gru/while?
7sequential/batch_normalization/batchnorm/ReadVariableOpReadVariableOp@sequential_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype029
7sequential/batch_normalization/batchnorm/ReadVariableOp?
.sequential/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:20
.sequential/batch_normalization/batchnorm/add/y?
,sequential/batch_normalization/batchnorm/addAddV2?sequential/batch_normalization/batchnorm/ReadVariableOp:value:07sequential/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/add?
.sequential/batch_normalization/batchnorm/RsqrtRsqrt0sequential/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/Rsqrt?
;sequential/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpDsequential_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02=
;sequential/batch_normalization/batchnorm/mul/ReadVariableOp?
,sequential/batch_normalization/batchnorm/mulMul2sequential/batch_normalization/batchnorm/Rsqrt:y:0Csequential/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/mul?
.sequential/batch_normalization/batchnorm/mul_1Mulbatch_normalization_input0sequential/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:?????????5?20
.sequential/batch_normalization/batchnorm/mul_1?
9sequential/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOpBsequential_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02;
9sequential/batch_normalization/batchnorm/ReadVariableOp_1?
.sequential/batch_normalization/batchnorm/mul_2MulAsequential/batch_normalization/batchnorm/ReadVariableOp_1:value:00sequential/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes	
:?20
.sequential/batch_normalization/batchnorm/mul_2?
9sequential/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOpBsequential_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02;
9sequential/batch_normalization/batchnorm/ReadVariableOp_2?
,sequential/batch_normalization/batchnorm/subSubAsequential/batch_normalization/batchnorm/ReadVariableOp_2:value:02sequential/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2.
,sequential/batch_normalization/batchnorm/sub?
.sequential/batch_normalization/batchnorm/add_1AddV22sequential/batch_normalization/batchnorm/mul_1:z:00sequential/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????5?20
.sequential/batch_normalization/batchnorm/add_1?
sequential/gru/ShapeShape2sequential/batch_normalization/batchnorm/add_1:z:0*
T0*
_output_shapes
:2
sequential/gru/Shape?
"sequential/gru/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/gru/strided_slice/stack?
$sequential/gru/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/gru/strided_slice/stack_1?
$sequential/gru/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/gru/strided_slice/stack_2?
sequential/gru/strided_sliceStridedSlicesequential/gru/Shape:output:0+sequential/gru/strided_slice/stack:output:0-sequential/gru/strided_slice/stack_1:output:0-sequential/gru/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sequential/gru/strided_slicez
sequential/gru/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :52
sequential/gru/zeros/mul/y?
sequential/gru/zeros/mulMul%sequential/gru/strided_slice:output:0#sequential/gru/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/zeros/mul}
sequential/gru/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
sequential/gru/zeros/Less/y?
sequential/gru/zeros/LessLesssequential/gru/zeros/mul:z:0$sequential/gru/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/zeros/Less?
sequential/gru/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :52
sequential/gru/zeros/packed/1?
sequential/gru/zeros/packedPack%sequential/gru/strided_slice:output:0&sequential/gru/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/gru/zeros/packed}
sequential/gru/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/gru/zeros/Const?
sequential/gru/zerosFill$sequential/gru/zeros/packed:output:0#sequential/gru/zeros/Const:output:0*
T0*'
_output_shapes
:?????????52
sequential/gru/zeros?
sequential/gru/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
sequential/gru/transpose/perm?
sequential/gru/transpose	Transpose2sequential/batch_normalization/batchnorm/add_1:z:0&sequential/gru/transpose/perm:output:0*
T0*,
_output_shapes
:5??????????2
sequential/gru/transpose|
sequential/gru/Shape_1Shapesequential/gru/transpose:y:0*
T0*
_output_shapes
:2
sequential/gru/Shape_1?
$sequential/gru/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/gru/strided_slice_1/stack?
&sequential/gru/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_1/stack_1?
&sequential/gru/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_1/stack_2?
sequential/gru/strided_slice_1StridedSlicesequential/gru/Shape_1:output:0-sequential/gru/strided_slice_1/stack:output:0/sequential/gru/strided_slice_1/stack_1:output:0/sequential/gru/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
sequential/gru/strided_slice_1?
*sequential/gru/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2,
*sequential/gru/TensorArrayV2/element_shape?
sequential/gru/TensorArrayV2TensorListReserve3sequential/gru/TensorArrayV2/element_shape:output:0'sequential/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
sequential/gru/TensorArrayV2?
Dsequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2F
Dsequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shape?
6sequential/gru/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/gru/transpose:y:0Msequential/gru/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type028
6sequential/gru/TensorArrayUnstack/TensorListFromTensor?
$sequential/gru/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/gru/strided_slice_2/stack?
&sequential/gru/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_2/stack_1?
&sequential/gru/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_2/stack_2?
sequential/gru/strided_slice_2StridedSlicesequential/gru/transpose:y:0-sequential/gru/strided_slice_2/stack:output:0/sequential/gru/strided_slice_2/stack_1:output:0/sequential/gru/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2 
sequential/gru/strided_slice_2?
'sequential/gru/gru_cell/ones_like/ShapeShape'sequential/gru/strided_slice_2:output:0*
T0*
_output_shapes
:2)
'sequential/gru/gru_cell/ones_like/Shape?
'sequential/gru/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2)
'sequential/gru/gru_cell/ones_like/Const?
!sequential/gru/gru_cell/ones_likeFill0sequential/gru/gru_cell/ones_like/Shape:output:00sequential/gru/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2#
!sequential/gru/gru_cell/ones_like?
)sequential/gru/gru_cell/ones_like_1/ShapeShapesequential/gru/zeros:output:0*
T0*
_output_shapes
:2+
)sequential/gru/gru_cell/ones_like_1/Shape?
)sequential/gru/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2+
)sequential/gru/gru_cell/ones_like_1/Const?
#sequential/gru/gru_cell/ones_like_1Fill2sequential/gru/gru_cell/ones_like_1/Shape:output:02sequential/gru/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52%
#sequential/gru/gru_cell/ones_like_1?
&sequential/gru/gru_cell/ReadVariableOpReadVariableOp/sequential_gru_gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&sequential/gru/gru_cell/ReadVariableOp?
sequential/gru/gru_cell/unstackUnpack.sequential/gru/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2!
sequential/gru/gru_cell/unstack?
sequential/gru/gru_cell/mulMul'sequential/gru/strided_slice_2:output:0*sequential/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
sequential/gru/gru_cell/mul?
sequential/gru/gru_cell/mul_1Mul'sequential/gru/strided_slice_2:output:0*sequential/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
sequential/gru/gru_cell/mul_1?
sequential/gru/gru_cell/mul_2Mul'sequential/gru/strided_slice_2:output:0*sequential/gru/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
sequential/gru/gru_cell/mul_2?
(sequential/gru/gru_cell/ReadVariableOp_1ReadVariableOp1sequential_gru_gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02*
(sequential/gru/gru_cell/ReadVariableOp_1?
+sequential/gru/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2-
+sequential/gru/gru_cell/strided_slice/stack?
-sequential/gru/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2/
-sequential/gru/gru_cell/strided_slice/stack_1?
-sequential/gru/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2/
-sequential/gru/gru_cell/strided_slice/stack_2?
%sequential/gru/gru_cell/strided_sliceStridedSlice0sequential/gru/gru_cell/ReadVariableOp_1:value:04sequential/gru/gru_cell/strided_slice/stack:output:06sequential/gru/gru_cell/strided_slice/stack_1:output:06sequential/gru/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2'
%sequential/gru/gru_cell/strided_slice?
sequential/gru/gru_cell/MatMulMatMulsequential/gru/gru_cell/mul:z:0.sequential/gru/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52 
sequential/gru/gru_cell/MatMul?
(sequential/gru/gru_cell/ReadVariableOp_2ReadVariableOp1sequential_gru_gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02*
(sequential/gru/gru_cell/ReadVariableOp_2?
-sequential/gru/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2/
-sequential/gru/gru_cell/strided_slice_1/stack?
/sequential/gru/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   21
/sequential/gru/gru_cell/strided_slice_1/stack_1?
/sequential/gru/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/sequential/gru/gru_cell/strided_slice_1/stack_2?
'sequential/gru/gru_cell/strided_slice_1StridedSlice0sequential/gru/gru_cell/ReadVariableOp_2:value:06sequential/gru/gru_cell/strided_slice_1/stack:output:08sequential/gru/gru_cell/strided_slice_1/stack_1:output:08sequential/gru/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2)
'sequential/gru/gru_cell/strided_slice_1?
 sequential/gru/gru_cell/MatMul_1MatMul!sequential/gru/gru_cell/mul_1:z:00sequential/gru/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52"
 sequential/gru/gru_cell/MatMul_1?
(sequential/gru/gru_cell/ReadVariableOp_3ReadVariableOp1sequential_gru_gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02*
(sequential/gru/gru_cell/ReadVariableOp_3?
-sequential/gru/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2/
-sequential/gru/gru_cell/strided_slice_2/stack?
/sequential/gru/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        21
/sequential/gru/gru_cell/strided_slice_2/stack_1?
/sequential/gru/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/sequential/gru/gru_cell/strided_slice_2/stack_2?
'sequential/gru/gru_cell/strided_slice_2StridedSlice0sequential/gru/gru_cell/ReadVariableOp_3:value:06sequential/gru/gru_cell/strided_slice_2/stack:output:08sequential/gru/gru_cell/strided_slice_2/stack_1:output:08sequential/gru/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2)
'sequential/gru/gru_cell/strided_slice_2?
 sequential/gru/gru_cell/MatMul_2MatMul!sequential/gru/gru_cell/mul_2:z:00sequential/gru/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52"
 sequential/gru/gru_cell/MatMul_2?
-sequential/gru/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/gru/gru_cell/strided_slice_3/stack?
/sequential/gru/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:521
/sequential/gru/gru_cell/strided_slice_3/stack_1?
/sequential/gru/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential/gru/gru_cell/strided_slice_3/stack_2?
'sequential/gru/gru_cell/strided_slice_3StridedSlice(sequential/gru/gru_cell/unstack:output:06sequential/gru/gru_cell/strided_slice_3/stack:output:08sequential/gru/gru_cell/strided_slice_3/stack_1:output:08sequential/gru/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2)
'sequential/gru/gru_cell/strided_slice_3?
sequential/gru/gru_cell/BiasAddBiasAdd(sequential/gru/gru_cell/MatMul:product:00sequential/gru/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52!
sequential/gru/gru_cell/BiasAdd?
-sequential/gru/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52/
-sequential/gru/gru_cell/strided_slice_4/stack?
/sequential/gru/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j21
/sequential/gru/gru_cell/strided_slice_4/stack_1?
/sequential/gru/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential/gru/gru_cell/strided_slice_4/stack_2?
'sequential/gru/gru_cell/strided_slice_4StridedSlice(sequential/gru/gru_cell/unstack:output:06sequential/gru/gru_cell/strided_slice_4/stack:output:08sequential/gru/gru_cell/strided_slice_4/stack_1:output:08sequential/gru/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52)
'sequential/gru/gru_cell/strided_slice_4?
!sequential/gru/gru_cell/BiasAdd_1BiasAdd*sequential/gru/gru_cell/MatMul_1:product:00sequential/gru/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52#
!sequential/gru/gru_cell/BiasAdd_1?
-sequential/gru/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2/
-sequential/gru/gru_cell/strided_slice_5/stack?
/sequential/gru/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 21
/sequential/gru/gru_cell/strided_slice_5/stack_1?
/sequential/gru/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential/gru/gru_cell/strided_slice_5/stack_2?
'sequential/gru/gru_cell/strided_slice_5StridedSlice(sequential/gru/gru_cell/unstack:output:06sequential/gru/gru_cell/strided_slice_5/stack:output:08sequential/gru/gru_cell/strided_slice_5/stack_1:output:08sequential/gru/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2)
'sequential/gru/gru_cell/strided_slice_5?
!sequential/gru/gru_cell/BiasAdd_2BiasAdd*sequential/gru/gru_cell/MatMul_2:product:00sequential/gru/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52#
!sequential/gru/gru_cell/BiasAdd_2?
sequential/gru/gru_cell/mul_3Mulsequential/gru/zeros:output:0,sequential/gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
sequential/gru/gru_cell/mul_3?
sequential/gru/gru_cell/mul_4Mulsequential/gru/zeros:output:0,sequential/gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
sequential/gru/gru_cell/mul_4?
sequential/gru/gru_cell/mul_5Mulsequential/gru/zeros:output:0,sequential/gru/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
sequential/gru/gru_cell/mul_5?
(sequential/gru/gru_cell/ReadVariableOp_4ReadVariableOp1sequential_gru_gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02*
(sequential/gru/gru_cell/ReadVariableOp_4?
-sequential/gru/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-sequential/gru/gru_cell/strided_slice_6/stack?
/sequential/gru/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   21
/sequential/gru/gru_cell/strided_slice_6/stack_1?
/sequential/gru/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/sequential/gru/gru_cell/strided_slice_6/stack_2?
'sequential/gru/gru_cell/strided_slice_6StridedSlice0sequential/gru/gru_cell/ReadVariableOp_4:value:06sequential/gru/gru_cell/strided_slice_6/stack:output:08sequential/gru/gru_cell/strided_slice_6/stack_1:output:08sequential/gru/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2)
'sequential/gru/gru_cell/strided_slice_6?
 sequential/gru/gru_cell/MatMul_3MatMul!sequential/gru/gru_cell/mul_3:z:00sequential/gru/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52"
 sequential/gru/gru_cell/MatMul_3?
(sequential/gru/gru_cell/ReadVariableOp_5ReadVariableOp1sequential_gru_gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02*
(sequential/gru/gru_cell/ReadVariableOp_5?
-sequential/gru/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2/
-sequential/gru/gru_cell/strided_slice_7/stack?
/sequential/gru/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   21
/sequential/gru/gru_cell/strided_slice_7/stack_1?
/sequential/gru/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/sequential/gru/gru_cell/strided_slice_7/stack_2?
'sequential/gru/gru_cell/strided_slice_7StridedSlice0sequential/gru/gru_cell/ReadVariableOp_5:value:06sequential/gru/gru_cell/strided_slice_7/stack:output:08sequential/gru/gru_cell/strided_slice_7/stack_1:output:08sequential/gru/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2)
'sequential/gru/gru_cell/strided_slice_7?
 sequential/gru/gru_cell/MatMul_4MatMul!sequential/gru/gru_cell/mul_4:z:00sequential/gru/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52"
 sequential/gru/gru_cell/MatMul_4?
-sequential/gru/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-sequential/gru/gru_cell/strided_slice_8/stack?
/sequential/gru/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:521
/sequential/gru/gru_cell/strided_slice_8/stack_1?
/sequential/gru/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential/gru/gru_cell/strided_slice_8/stack_2?
'sequential/gru/gru_cell/strided_slice_8StridedSlice(sequential/gru/gru_cell/unstack:output:16sequential/gru/gru_cell/strided_slice_8/stack:output:08sequential/gru/gru_cell/strided_slice_8/stack_1:output:08sequential/gru/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2)
'sequential/gru/gru_cell/strided_slice_8?
!sequential/gru/gru_cell/BiasAdd_3BiasAdd*sequential/gru/gru_cell/MatMul_3:product:00sequential/gru/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52#
!sequential/gru/gru_cell/BiasAdd_3?
-sequential/gru/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52/
-sequential/gru/gru_cell/strided_slice_9/stack?
/sequential/gru/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j21
/sequential/gru/gru_cell/strided_slice_9/stack_1?
/sequential/gru/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/sequential/gru/gru_cell/strided_slice_9/stack_2?
'sequential/gru/gru_cell/strided_slice_9StridedSlice(sequential/gru/gru_cell/unstack:output:16sequential/gru/gru_cell/strided_slice_9/stack:output:08sequential/gru/gru_cell/strided_slice_9/stack_1:output:08sequential/gru/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52)
'sequential/gru/gru_cell/strided_slice_9?
!sequential/gru/gru_cell/BiasAdd_4BiasAdd*sequential/gru/gru_cell/MatMul_4:product:00sequential/gru/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52#
!sequential/gru/gru_cell/BiasAdd_4?
sequential/gru/gru_cell/addAddV2(sequential/gru/gru_cell/BiasAdd:output:0*sequential/gru/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
sequential/gru/gru_cell/add?
sequential/gru/gru_cell/SigmoidSigmoidsequential/gru/gru_cell/add:z:0*
T0*'
_output_shapes
:?????????52!
sequential/gru/gru_cell/Sigmoid?
sequential/gru/gru_cell/add_1AddV2*sequential/gru/gru_cell/BiasAdd_1:output:0*sequential/gru/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
sequential/gru/gru_cell/add_1?
!sequential/gru/gru_cell/Sigmoid_1Sigmoid!sequential/gru/gru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52#
!sequential/gru/gru_cell/Sigmoid_1?
(sequential/gru/gru_cell/ReadVariableOp_6ReadVariableOp1sequential_gru_gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02*
(sequential/gru/gru_cell/ReadVariableOp_6?
.sequential/gru/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   20
.sequential/gru/gru_cell/strided_slice_10/stack?
0sequential/gru/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        22
0sequential/gru/gru_cell/strided_slice_10/stack_1?
0sequential/gru/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      22
0sequential/gru/gru_cell/strided_slice_10/stack_2?
(sequential/gru/gru_cell/strided_slice_10StridedSlice0sequential/gru/gru_cell/ReadVariableOp_6:value:07sequential/gru/gru_cell/strided_slice_10/stack:output:09sequential/gru/gru_cell/strided_slice_10/stack_1:output:09sequential/gru/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2*
(sequential/gru/gru_cell/strided_slice_10?
 sequential/gru/gru_cell/MatMul_5MatMul!sequential/gru/gru_cell/mul_5:z:01sequential/gru/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52"
 sequential/gru/gru_cell/MatMul_5?
.sequential/gru/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j20
.sequential/gru/gru_cell/strided_slice_11/stack?
0sequential/gru/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 22
0sequential/gru/gru_cell/strided_slice_11/stack_1?
0sequential/gru/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0sequential/gru/gru_cell/strided_slice_11/stack_2?
(sequential/gru/gru_cell/strided_slice_11StridedSlice(sequential/gru/gru_cell/unstack:output:17sequential/gru/gru_cell/strided_slice_11/stack:output:09sequential/gru/gru_cell/strided_slice_11/stack_1:output:09sequential/gru/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2*
(sequential/gru/gru_cell/strided_slice_11?
!sequential/gru/gru_cell/BiasAdd_5BiasAdd*sequential/gru/gru_cell/MatMul_5:product:01sequential/gru/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52#
!sequential/gru/gru_cell/BiasAdd_5?
sequential/gru/gru_cell/mul_6Mul%sequential/gru/gru_cell/Sigmoid_1:y:0*sequential/gru/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
sequential/gru/gru_cell/mul_6?
sequential/gru/gru_cell/add_2AddV2*sequential/gru/gru_cell/BiasAdd_2:output:0!sequential/gru/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
sequential/gru/gru_cell/add_2?
sequential/gru/gru_cell/TanhTanh!sequential/gru/gru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
sequential/gru/gru_cell/Tanh?
sequential/gru/gru_cell/mul_7Mul#sequential/gru/gru_cell/Sigmoid:y:0sequential/gru/zeros:output:0*
T0*'
_output_shapes
:?????????52
sequential/gru/gru_cell/mul_7?
sequential/gru/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sequential/gru/gru_cell/sub/x?
sequential/gru/gru_cell/subSub&sequential/gru/gru_cell/sub/x:output:0#sequential/gru/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
sequential/gru/gru_cell/sub?
sequential/gru/gru_cell/mul_8Mulsequential/gru/gru_cell/sub:z:0 sequential/gru/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
sequential/gru/gru_cell/mul_8?
sequential/gru/gru_cell/add_3AddV2!sequential/gru/gru_cell/mul_7:z:0!sequential/gru/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
sequential/gru/gru_cell/add_3?
,sequential/gru/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   2.
,sequential/gru/TensorArrayV2_1/element_shape?
sequential/gru/TensorArrayV2_1TensorListReserve5sequential/gru/TensorArrayV2_1/element_shape:output:0'sequential/gru/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02 
sequential/gru/TensorArrayV2_1l
sequential/gru/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/gru/time?
'sequential/gru/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'sequential/gru/while/maximum_iterations?
!sequential/gru/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2#
!sequential/gru/while/loop_counter?
sequential/gru/whileWhile*sequential/gru/while/loop_counter:output:00sequential/gru/while/maximum_iterations:output:0sequential/gru/time:output:0'sequential/gru/TensorArrayV2_1:handle:0sequential/gru/zeros:output:0'sequential/gru/strided_slice_1:output:0Fsequential/gru/TensorArrayUnstack/TensorListFromTensor:output_handle:0/sequential_gru_gru_cell_readvariableop_resource1sequential_gru_gru_cell_readvariableop_1_resource1sequential_gru_gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????5: : : : : *%
_read_only_resource_inputs
	*+
body#R!
sequential_gru_while_body_19008*+
cond#R!
sequential_gru_while_cond_19007*8
output_shapes'
%: : : : :?????????5: : : : : *
parallel_iterations 2
sequential/gru/while?
?sequential/gru/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   2A
?sequential/gru/TensorArrayV2Stack/TensorListStack/element_shape?
1sequential/gru/TensorArrayV2Stack/TensorListStackTensorListStacksequential/gru/while:output:3Hsequential/gru/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:5?????????5*
element_dtype023
1sequential/gru/TensorArrayV2Stack/TensorListStack?
$sequential/gru/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2&
$sequential/gru/strided_slice_3/stack?
&sequential/gru/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&sequential/gru/strided_slice_3/stack_1?
&sequential/gru/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&sequential/gru/strided_slice_3/stack_2?
sequential/gru/strided_slice_3StridedSlice:sequential/gru/TensorArrayV2Stack/TensorListStack:tensor:0-sequential/gru/strided_slice_3/stack:output:0/sequential/gru/strided_slice_3/stack_1:output:0/sequential/gru/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????5*
shrink_axis_mask2 
sequential/gru/strided_slice_3?
sequential/gru/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2!
sequential/gru/transpose_1/perm?
sequential/gru/transpose_1	Transpose:sequential/gru/TensorArrayV2Stack/TensorListStack:tensor:0(sequential/gru/transpose_1/perm:output:0*
T0*+
_output_shapes
:?????????552
sequential/gru/transpose_1?
sequential/gru/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/gru/runtime?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:5*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul'sequential/gru/strided_slice_3:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense/BiasAdd?
sequential/dense/SigmoidSigmoid!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense/Sigmoid?
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_1/MatMul/ReadVariableOp?
sequential/dense_1/MatMulMatMulsequential/dense/Sigmoid:y:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/MatMul?
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOp?
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/BiasAdd?
sequential/dense_1/SigmoidSigmoid#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_1/Sigmoid?
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOp1sequential_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(sequential/dense_2/MatMul/ReadVariableOp?
sequential/dense_2/MatMulMatMulsequential/dense_1/Sigmoid:y:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/MatMul?
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_2/BiasAdd/ReadVariableOp?
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/BiasAdd?
sequential/dense_2/SigmoidSigmoid#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sequential/dense_2/Sigmoid?
IdentityIdentitysequential/dense_2/Sigmoid:y:08^sequential/batch_normalization/batchnorm/ReadVariableOp:^sequential/batch_normalization/batchnorm/ReadVariableOp_1:^sequential/batch_normalization/batchnorm/ReadVariableOp_2<^sequential/batch_normalization/batchnorm/mul/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp'^sequential/gru/gru_cell/ReadVariableOp)^sequential/gru/gru_cell/ReadVariableOp_1)^sequential/gru/gru_cell/ReadVariableOp_2)^sequential/gru/gru_cell/ReadVariableOp_3)^sequential/gru/gru_cell/ReadVariableOp_4)^sequential/gru/gru_cell/ReadVariableOp_5)^sequential/gru/gru_cell/ReadVariableOp_6^sequential/gru/while*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????5?:::::::::::::2r
7sequential/batch_normalization/batchnorm/ReadVariableOp7sequential/batch_normalization/batchnorm/ReadVariableOp2v
9sequential/batch_normalization/batchnorm/ReadVariableOp_19sequential/batch_normalization/batchnorm/ReadVariableOp_12v
9sequential/batch_normalization/batchnorm/ReadVariableOp_29sequential/batch_normalization/batchnorm/ReadVariableOp_22z
;sequential/batch_normalization/batchnorm/mul/ReadVariableOp;sequential/batch_normalization/batchnorm/mul/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp2P
&sequential/gru/gru_cell/ReadVariableOp&sequential/gru/gru_cell/ReadVariableOp2T
(sequential/gru/gru_cell/ReadVariableOp_1(sequential/gru/gru_cell/ReadVariableOp_12T
(sequential/gru/gru_cell/ReadVariableOp_2(sequential/gru/gru_cell/ReadVariableOp_22T
(sequential/gru/gru_cell/ReadVariableOp_3(sequential/gru/gru_cell/ReadVariableOp_32T
(sequential/gru/gru_cell/ReadVariableOp_4(sequential/gru/gru_cell/ReadVariableOp_42T
(sequential/gru/gru_cell/ReadVariableOp_5(sequential/gru/gru_cell/ReadVariableOp_52T
(sequential/gru/gru_cell/ReadVariableOp_6(sequential/gru/gru_cell/ReadVariableOp_62,
sequential/gru/whilesequential/gru/while:g c
,
_output_shapes
:?????????5?
3
_user_specified_namebatch_normalization_input
?
?
gru_while_cond_21710$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1;
7gru_while_gru_while_cond_21710___redundant_placeholder0;
7gru_while_gru_while_cond_21710___redundant_placeholder1;
7gru_while_gru_while_cond_21710___redundant_placeholder2;
7gru_while_gru_while_cond_21710___redundant_placeholder3
gru_while_identity
?
gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: 2
gru/while/Lessi
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/while/Identity"1
gru_while_identitygru/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????5: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
:
?	
?
B__inference_dense_2_layer_call_and_return_conditional_losses_20906

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoid?
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_20996

inputs
batch_normalization_20964
batch_normalization_20966
batch_normalization_20968
batch_normalization_20970
	gru_20973
	gru_20975
	gru_20977
dense_20980
dense_20982
dense_1_20985
dense_1_20987
dense_2_20990
dense_2_20992
identity??+batch_normalization/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?gru/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_20964batch_normalization_20966batch_normalization_20968batch_normalization_20970*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????5?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_200942-
+batch_normalization/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0	gru_20973	gru_20975	gru_20977*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_205282
gru/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_20980dense_20982*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_208522
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_20985dense_1_20987*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_208792!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_20990dense_2_20992*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_209062!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????5?:::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
?
?
#__inference_gru_layer_call_fn_22794

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_208112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????52

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????5?:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
??
?
while_body_20328
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_0.
*while_gru_cell_readvariableop_1_resource_0.
*while_gru_cell_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource,
(while_gru_cell_readvariableop_1_resource,
(while_gru_cell_readvariableop_4_resource??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape?
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/ones_like/Const?
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/ones_like?
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
while/gru_cell/dropout/Const?
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout/Mul?
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shape?
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2?̩25
3while/gru_cell/dropout/random_uniform/RandomUniform?
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%while/gru_cell/dropout/GreaterEqual/y?
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2%
#while/gru_cell/dropout/GreaterEqual?
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout/Cast?
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout/Mul_1?
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2 
while/gru_cell/dropout_1/Const?
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout_1/Mul?
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/Shape?
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??}27
5while/gru_cell/dropout_1/random_uniform/RandomUniform?
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2)
'while/gru_cell/dropout_1/GreaterEqual/y?
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell/dropout_1/GreaterEqual?
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout_1/Cast?
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell/dropout_1/Mul_1?
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2 
while/gru_cell/dropout_2/Const?
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout_2/Mul?
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/Shape?
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_2/random_uniform/RandomUniform?
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2)
'while/gru_cell/dropout_2/GreaterEqual/y?
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell/dropout_2/GreaterEqual?
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout_2/Cast?
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell/dropout_2/Mul_1?
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/Shape?
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell/ones_like_1/Const?
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/ones_like_1?
while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/dropout_3/Const?
while/gru_cell/dropout_3/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/dropout_3/Mul?
while/gru_cell/dropout_3/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_3/Shape?
5while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_3/random_uniform/RandomUniform?
'while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2)
'while/gru_cell/dropout_3/GreaterEqual/y?
%while/gru_cell/dropout_3/GreaterEqualGreaterEqual>while/gru_cell/dropout_3/random_uniform/RandomUniform:output:00while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52'
%while/gru_cell/dropout_3/GreaterEqual?
while/gru_cell/dropout_3/CastCast)while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
while/gru_cell/dropout_3/Cast?
while/gru_cell/dropout_3/Mul_1Mul while/gru_cell/dropout_3/Mul:z:0!while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????52 
while/gru_cell/dropout_3/Mul_1?
while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/dropout_4/Const?
while/gru_cell/dropout_4/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/dropout_4/Mul?
while/gru_cell/dropout_4/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_4/Shape?
5while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2倠27
5while/gru_cell/dropout_4/random_uniform/RandomUniform?
'while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2)
'while/gru_cell/dropout_4/GreaterEqual/y?
%while/gru_cell/dropout_4/GreaterEqualGreaterEqual>while/gru_cell/dropout_4/random_uniform/RandomUniform:output:00while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52'
%while/gru_cell/dropout_4/GreaterEqual?
while/gru_cell/dropout_4/CastCast)while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
while/gru_cell/dropout_4/Cast?
while/gru_cell/dropout_4/Mul_1Mul while/gru_cell/dropout_4/Mul:z:0!while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????52 
while/gru_cell/dropout_4/Mul_1?
while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/dropout_5/Const?
while/gru_cell/dropout_5/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/dropout_5/Mul?
while/gru_cell/dropout_5/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_5/Shape?
5while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed227
5while/gru_cell/dropout_5/random_uniform/RandomUniform?
'while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2)
'while/gru_cell/dropout_5/GreaterEqual/y?
%while/gru_cell/dropout_5/GreaterEqualGreaterEqual>while/gru_cell/dropout_5/random_uniform/RandomUniform:output:00while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52'
%while/gru_cell/dropout_5/GreaterEqual?
while/gru_cell/dropout_5/CastCast)while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
while/gru_cell/dropout_5/Cast?
while/gru_cell/dropout_5/Mul_1Mul while/gru_cell/dropout_5/Mul:z:0!while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????52 
while/gru_cell/dropout_5/Mul_1?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1?
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_1?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_2?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_1?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_2?
while/gru_cell/mul_3Mulwhile_placeholder_2"while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_3?
while/gru_cell/mul_4Mulwhile_placeholder_2"while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_4?
while/gru_cell/mul_5Mulwhile_placeholder_2"while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_5?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_4?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_3?
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52&
$while/gru_cell/strided_slice_9/stack?
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2(
&while/gru_cell/strided_slice_9/stack_1?
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2?
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52 
while/gru_cell/strided_slice_9?
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Sigmoid_1?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_6?
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2'
%while/gru_cell/strided_slice_10/stack?
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1?
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2?
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_5?
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2'
%while/gru_cell/strided_slice_11/stack?
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1?
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2?
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2!
while/gru_cell/strided_slice_11?
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_5?
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_6?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Tanh?
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/sub?
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_8?
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:?????????52
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????5: : :::2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
: 
??
?
while_body_22620
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_0.
*while_gru_cell_readvariableop_1_resource_0.
*while_gru_cell_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource,
(while_gru_cell_readvariableop_1_resource,
(while_gru_cell_readvariableop_4_resource??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape?
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/ones_like/Const?
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/ones_like?
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/Shape?
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell/ones_like_1/Const?
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/ones_like_1?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1?
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_1?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_2?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_1?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_2?
while/gru_cell/mul_3Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_3?
while/gru_cell/mul_4Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_4?
while/gru_cell/mul_5Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_5?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_4?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_3?
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52&
$while/gru_cell/strided_slice_9/stack?
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2(
&while/gru_cell/strided_slice_9/stack_1?
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2?
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52 
while/gru_cell/strided_slice_9?
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Sigmoid_1?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_6?
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2'
%while/gru_cell/strided_slice_10/stack?
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1?
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2?
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_5?
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2'
%while/gru_cell/strided_slice_11/stack?
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1?
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2?
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2!
while/gru_cell/strided_slice_11?
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_5?
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_6?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Tanh?
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/sub?
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_8?
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:?????????52
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????5: : :::2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_19863
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_19863___redundant_placeholder03
/while_while_cond_19863___redundant_placeholder13
/while_while_cond_19863___redundant_placeholder23
/while_while_cond_19863___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????5: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
:
?
?
gru_while_cond_21342$
 gru_while_gru_while_loop_counter*
&gru_while_gru_while_maximum_iterations
gru_while_placeholder
gru_while_placeholder_1
gru_while_placeholder_2&
"gru_while_less_gru_strided_slice_1;
7gru_while_gru_while_cond_21342___redundant_placeholder0;
7gru_while_gru_while_cond_21342___redundant_placeholder1;
7gru_while_gru_while_cond_21342___redundant_placeholder2;
7gru_while_gru_while_cond_21342___redundant_placeholder3
gru_while_identity
?
gru/while/LessLessgru_while_placeholder"gru_while_less_gru_strided_slice_1*
T0*
_output_shapes
: 2
gru/while/Lessi
gru/while/IdentityIdentitygru/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/while/Identity"1
gru_while_identitygru/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????5: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
:
??
?
while_body_20659
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_0.
*while_gru_cell_readvariableop_1_resource_0.
*while_gru_cell_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource,
(while_gru_cell_readvariableop_1_resource,
(while_gru_cell_readvariableop_4_resource??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape?
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/ones_like/Const?
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/ones_like?
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/Shape?
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell/ones_like_1/Const?
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/ones_like_1?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1?
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_1?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_2?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_1?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_2?
while/gru_cell/mul_3Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_3?
while/gru_cell/mul_4Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_4?
while/gru_cell/mul_5Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_5?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_4?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_3?
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52&
$while/gru_cell/strided_slice_9/stack?
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2(
&while/gru_cell/strided_slice_9/stack_1?
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2?
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52 
while/gru_cell/strided_slice_9?
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Sigmoid_1?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_6?
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2'
%while/gru_cell/strided_slice_10/stack?
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1?
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2?
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_5?
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2'
%while/gru_cell/strided_slice_11/stack?
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1?
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2?
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2!
while/gru_cell/strided_slice_11?
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_5?
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_6?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Tanh?
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/sub?
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_8?
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:?????????52
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????5: : :::2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
: 
?0
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21982

inputs
assignmovingavg_21957
assignmovingavg_1_21963)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*5
_output_shapes#
!:???????????????????2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/21957*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_21957*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/21957*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/21957*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_21957AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/21957*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/21963*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_21963*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/21963*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/21963*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_21963AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/21963*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
while_cond_20327
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_20327___redundant_placeholder03
/while_while_cond_20327___redundant_placeholder13
/while_while_cond_20327___redundant_placeholder23
/while_while_cond_20327___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????5: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
:
??
?
while_body_23304
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_0.
*while_gru_cell_readvariableop_1_resource_0.
*while_gru_cell_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource,
(while_gru_cell_readvariableop_1_resource,
(while_gru_cell_readvariableop_4_resource??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape?
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/ones_like/Const?
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/ones_like?
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/Shape?
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell/ones_like_1/Const?
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/ones_like_1?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1?
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0!while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_1?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_2?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_1?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_2?
while/gru_cell/mul_3Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_3?
while/gru_cell/mul_4Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_4?
while/gru_cell/mul_5Mulwhile_placeholder_2#while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_5?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_4?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_3?
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52&
$while/gru_cell/strided_slice_9/stack?
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2(
&while/gru_cell/strided_slice_9/stack_1?
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2?
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52 
while/gru_cell/strided_slice_9?
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Sigmoid_1?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_6?
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2'
%while/gru_cell/strided_slice_10/stack?
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1?
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2?
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_5?
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2'
%while/gru_cell/strided_slice_11/stack?
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1?
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2?
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2!
while/gru_cell/strided_slice_11?
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_5?
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_6?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Tanh?
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/sub?
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_8?
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:?????????52
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????5: : :::2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
: 
?

?
*__inference_sequential_layer_call_fn_21025
batch_normalization_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*-
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_209962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????5?:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:g c
,
_output_shapes
:?????????5?
3
_user_specified_namebatch_normalization_input
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_20958
batch_normalization_input
batch_normalization_20926
batch_normalization_20928
batch_normalization_20930
batch_normalization_20932
	gru_20935
	gru_20937
	gru_20939
dense_20942
dense_20944
dense_1_20947
dense_1_20949
dense_2_20952
dense_2_20954
identity??+batch_normalization/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?gru/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputbatch_normalization_20926batch_normalization_20928batch_normalization_20930batch_normalization_20932*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????5?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_201142-
+batch_normalization/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0	gru_20935	gru_20937	gru_20939*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_208112
gru/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_20942dense_20944*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_208522
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_20947dense_1_20949*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_208792!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_20952dense_2_20954*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_209062!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????5?:::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:g c
,
_output_shapes
:?????????5?
3
_user_specified_namebatch_normalization_input
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22002

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul?
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*5
_output_shapes#
!:???????????????????2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?H
?
__inference__traced_save_23940
file_prefix8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop2
.savev2_gru_gru_cell_kernel_read_readvariableop<
8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop0
,savev2_gru_gru_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopD
@savev2_rmsprop_batch_normalization_gamma_rms_read_readvariableopC
?savev2_rmsprop_batch_normalization_beta_rms_read_readvariableop7
3savev2_rmsprop_dense_kernel_rms_read_readvariableop5
1savev2_rmsprop_dense_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_1_bias_rms_read_readvariableop9
5savev2_rmsprop_dense_2_kernel_rms_read_readvariableop7
3savev2_rmsprop_dense_2_bias_rms_read_readvariableop>
:savev2_rmsprop_gru_gru_cell_kernel_rms_read_readvariableopH
Dsavev2_rmsprop_gru_gru_cell_recurrent_kernel_rms_read_readvariableop<
8savev2_rmsprop_gru_gru_cell_bias_rms_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop.savev2_gru_gru_cell_kernel_read_readvariableop8savev2_gru_gru_cell_recurrent_kernel_read_readvariableop,savev2_gru_gru_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop@savev2_rmsprop_batch_normalization_gamma_rms_read_readvariableop?savev2_rmsprop_batch_normalization_beta_rms_read_readvariableop3savev2_rmsprop_dense_kernel_rms_read_readvariableop1savev2_rmsprop_dense_bias_rms_read_readvariableop5savev2_rmsprop_dense_1_kernel_rms_read_readvariableop3savev2_rmsprop_dense_1_bias_rms_read_readvariableop5savev2_rmsprop_dense_2_kernel_rms_read_readvariableop3savev2_rmsprop_dense_2_bias_rms_read_readvariableop:savev2_rmsprop_gru_gru_cell_kernel_rms_read_readvariableopDsavev2_rmsprop_gru_gru_cell_recurrent_kernel_rms_read_readvariableop8savev2_rmsprop_gru_gru_cell_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?:?:?:?:5:::::: : : : : :
??:	5?:	?: : : : :?:?:5::::::
??:	5?:	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:$ 

_output_shapes

:5: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
??:%!

_output_shapes
:	5?:%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :!

_output_shapes	
:?:!

_output_shapes	
:?:$ 

_output_shapes

:5: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::&"
 
_output_shapes
:
??:% !

_output_shapes
:	5?:%!!

_output_shapes
:	?:"

_output_shapes
: 
??
?
while_body_22289
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0,
(while_gru_cell_readvariableop_resource_0.
*while_gru_cell_readvariableop_1_resource_0.
*while_gru_cell_readvariableop_4_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor*
&while_gru_cell_readvariableop_resource,
(while_gru_cell_readvariableop_1_resource,
(while_gru_cell_readvariableop_4_resource??while/gru_cell/ReadVariableOp?while/gru_cell/ReadVariableOp_1?while/gru_cell/ReadVariableOp_2?while/gru_cell/ReadVariableOp_3?while/gru_cell/ReadVariableOp_4?while/gru_cell/ReadVariableOp_5?while/gru_cell/ReadVariableOp_6?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
while/gru_cell/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2 
while/gru_cell/ones_like/Shape?
while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/ones_like/Const?
while/gru_cell/ones_likeFill'while/gru_cell/ones_like/Shape:output:0'while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/ones_like?
while/gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
while/gru_cell/dropout/Const?
while/gru_cell/dropout/MulMul!while/gru_cell/ones_like:output:0%while/gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout/Mul?
while/gru_cell/dropout/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2
while/gru_cell/dropout/Shape?
3while/gru_cell/dropout/random_uniform/RandomUniformRandomUniform%while/gru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??%25
3while/gru_cell/dropout/random_uniform/RandomUniform?
%while/gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2'
%while/gru_cell/dropout/GreaterEqual/y?
#while/gru_cell/dropout/GreaterEqualGreaterEqual<while/gru_cell/dropout/random_uniform/RandomUniform:output:0.while/gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2%
#while/gru_cell/dropout/GreaterEqual?
while/gru_cell/dropout/CastCast'while/gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout/Cast?
while/gru_cell/dropout/Mul_1Mulwhile/gru_cell/dropout/Mul:z:0while/gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout/Mul_1?
while/gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2 
while/gru_cell/dropout_1/Const?
while/gru_cell/dropout_1/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout_1/Mul?
while/gru_cell/dropout_1/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_1/Shape?
5while/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2??-27
5while/gru_cell/dropout_1/random_uniform/RandomUniform?
'while/gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2)
'while/gru_cell/dropout_1/GreaterEqual/y?
%while/gru_cell/dropout_1/GreaterEqualGreaterEqual>while/gru_cell/dropout_1/random_uniform/RandomUniform:output:00while/gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell/dropout_1/GreaterEqual?
while/gru_cell/dropout_1/CastCast)while/gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout_1/Cast?
while/gru_cell/dropout_1/Mul_1Mul while/gru_cell/dropout_1/Mul:z:0!while/gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell/dropout_1/Mul_1?
while/gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2 
while/gru_cell/dropout_2/Const?
while/gru_cell/dropout_2/MulMul!while/gru_cell/ones_like:output:0'while/gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/dropout_2/Mul?
while/gru_cell/dropout_2/ShapeShape!while/gru_cell/ones_like:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_2/Shape?
5while/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_2/random_uniform/RandomUniform?
'while/gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2)
'while/gru_cell/dropout_2/GreaterEqual/y?
%while/gru_cell/dropout_2/GreaterEqualGreaterEqual>while/gru_cell/dropout_2/random_uniform/RandomUniform:output:00while/gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2'
%while/gru_cell/dropout_2/GreaterEqual?
while/gru_cell/dropout_2/CastCast)while/gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
while/gru_cell/dropout_2/Cast?
while/gru_cell/dropout_2/Mul_1Mul while/gru_cell/dropout_2/Mul:z:0!while/gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2 
while/gru_cell/dropout_2/Mul_1?
 while/gru_cell/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:2"
 while/gru_cell/ones_like_1/Shape?
 while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 while/gru_cell/ones_like_1/Const?
while/gru_cell/ones_like_1Fill)while/gru_cell/ones_like_1/Shape:output:0)while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/ones_like_1?
while/gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/dropout_3/Const?
while/gru_cell/dropout_3/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/dropout_3/Mul?
while/gru_cell/dropout_3/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_3/Shape?
5while/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_3/random_uniform/RandomUniform?
'while/gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2)
'while/gru_cell/dropout_3/GreaterEqual/y?
%while/gru_cell/dropout_3/GreaterEqualGreaterEqual>while/gru_cell/dropout_3/random_uniform/RandomUniform:output:00while/gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52'
%while/gru_cell/dropout_3/GreaterEqual?
while/gru_cell/dropout_3/CastCast)while/gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
while/gru_cell/dropout_3/Cast?
while/gru_cell/dropout_3/Mul_1Mul while/gru_cell/dropout_3/Mul:z:0!while/gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????52 
while/gru_cell/dropout_3/Mul_1?
while/gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/dropout_4/Const?
while/gru_cell/dropout_4/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/dropout_4/Mul?
while/gru_cell/dropout_4/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_4/Shape?
5while/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_4/random_uniform/RandomUniform?
'while/gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2)
'while/gru_cell/dropout_4/GreaterEqual/y?
%while/gru_cell/dropout_4/GreaterEqualGreaterEqual>while/gru_cell/dropout_4/random_uniform/RandomUniform:output:00while/gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52'
%while/gru_cell/dropout_4/GreaterEqual?
while/gru_cell/dropout_4/CastCast)while/gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
while/gru_cell/dropout_4/Cast?
while/gru_cell/dropout_4/Mul_1Mul while/gru_cell/dropout_4/Mul:z:0!while/gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????52 
while/gru_cell/dropout_4/Mul_1?
while/gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
while/gru_cell/dropout_5/Const?
while/gru_cell/dropout_5/MulMul#while/gru_cell/ones_like_1:output:0'while/gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/dropout_5/Mul?
while/gru_cell/dropout_5/ShapeShape#while/gru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2 
while/gru_cell/dropout_5/Shape?
5while/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform'while/gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???27
5while/gru_cell/dropout_5/random_uniform/RandomUniform?
'while/gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2)
'while/gru_cell/dropout_5/GreaterEqual/y?
%while/gru_cell/dropout_5/GreaterEqualGreaterEqual>while/gru_cell/dropout_5/random_uniform/RandomUniform:output:00while/gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52'
%while/gru_cell/dropout_5/GreaterEqual?
while/gru_cell/dropout_5/CastCast)while/gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
while/gru_cell/dropout_5/Cast?
while/gru_cell/dropout_5/Mul_1Mul while/gru_cell/dropout_5/Mul:z:0!while/gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????52 
while/gru_cell/dropout_5/Mul_1?
while/gru_cell/ReadVariableOpReadVariableOp(while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02
while/gru_cell/ReadVariableOp?
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
while/gru_cell/unstack?
while/gru_cell/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0 while/gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul?
while/gru_cell/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_1?
while/gru_cell/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0"while/gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
while/gru_cell/mul_2?
while/gru_cell/ReadVariableOp_1ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_1?
"while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"while/gru_cell/strided_slice/stack?
$while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice/stack_1?
$while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$while/gru_cell/strided_slice/stack_2?
while/gru_cell/strided_sliceStridedSlice'while/gru_cell/ReadVariableOp_1:value:0+while/gru_cell/strided_slice/stack:output:0-while/gru_cell/strided_slice/stack_1:output:0-while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
while/gru_cell/strided_slice?
while/gru_cell/MatMulMatMulwhile/gru_cell/mul:z:0%while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul?
while/gru_cell/ReadVariableOp_2ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_2?
$while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice_1/stack?
&while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2(
&while/gru_cell/strided_slice_1/stack_1?
&while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_1/stack_2?
while/gru_cell/strided_slice_1StridedSlice'while/gru_cell/ReadVariableOp_2:value:0-while/gru_cell/strided_slice_1/stack:output:0/while/gru_cell/strided_slice_1/stack_1:output:0/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_1?
while/gru_cell/MatMul_1MatMulwhile/gru_cell/mul_1:z:0'while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_1?
while/gru_cell/ReadVariableOp_3ReadVariableOp*while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype02!
while/gru_cell/ReadVariableOp_3?
$while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2&
$while/gru_cell/strided_slice_2/stack?
&while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&while/gru_cell/strided_slice_2/stack_1?
&while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_2/stack_2?
while/gru_cell/strided_slice_2StridedSlice'while/gru_cell/ReadVariableOp_3:value:0-while/gru_cell/strided_slice_2/stack:output:0/while/gru_cell/strided_slice_2/stack_1:output:0/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_2?
while/gru_cell/MatMul_2MatMulwhile/gru_cell/mul_2:z:0'while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_2?
$while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_3/stack?
&while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52(
&while/gru_cell/strided_slice_3/stack_1?
&while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_3/stack_2?
while/gru_cell/strided_slice_3StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_3/stack:output:0/while/gru_cell/strided_slice_3/stack_1:output:0/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2 
while/gru_cell/strided_slice_3?
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0'while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd?
$while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52&
$while/gru_cell/strided_slice_4/stack?
&while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2(
&while/gru_cell/strided_slice_4/stack_1?
&while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_4/stack_2?
while/gru_cell/strided_slice_4StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_4/stack:output:0/while/gru_cell/strided_slice_4/stack_1:output:0/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52 
while/gru_cell/strided_slice_4?
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0'while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_1?
$while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2&
$while/gru_cell/strided_slice_5/stack?
&while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&while/gru_cell/strided_slice_5/stack_1?
&while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_5/stack_2?
while/gru_cell/strided_slice_5StridedSlicewhile/gru_cell/unstack:output:0-while/gru_cell/strided_slice_5/stack:output:0/while/gru_cell/strided_slice_5/stack_1:output:0/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2 
while/gru_cell/strided_slice_5?
while/gru_cell/BiasAdd_2BiasAdd!while/gru_cell/MatMul_2:product:0'while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_2?
while/gru_cell/mul_3Mulwhile_placeholder_2"while/gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_3?
while/gru_cell/mul_4Mulwhile_placeholder_2"while/gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_4?
while/gru_cell/mul_5Mulwhile_placeholder_2"while/gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_5?
while/gru_cell/ReadVariableOp_4ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_4?
$while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$while/gru_cell/strided_slice_6/stack?
&while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2(
&while/gru_cell/strided_slice_6/stack_1?
&while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_6/stack_2?
while/gru_cell/strided_slice_6StridedSlice'while/gru_cell/ReadVariableOp_4:value:0-while/gru_cell/strided_slice_6/stack:output:0/while/gru_cell/strided_slice_6/stack_1:output:0/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_6?
while/gru_cell/MatMul_3MatMulwhile/gru_cell/mul_3:z:0'while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_3?
while/gru_cell/ReadVariableOp_5ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_5?
$while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2&
$while/gru_cell/strided_slice_7/stack?
&while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2(
&while/gru_cell/strided_slice_7/stack_1?
&while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&while/gru_cell/strided_slice_7/stack_2?
while/gru_cell/strided_slice_7StridedSlice'while/gru_cell/ReadVariableOp_5:value:0-while/gru_cell/strided_slice_7/stack:output:0/while/gru_cell/strided_slice_7/stack_1:output:0/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2 
while/gru_cell/strided_slice_7?
while/gru_cell/MatMul_4MatMulwhile/gru_cell/mul_4:z:0'while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_4?
$while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$while/gru_cell/strided_slice_8/stack?
&while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52(
&while/gru_cell/strided_slice_8/stack_1?
&while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_8/stack_2?
while/gru_cell/strided_slice_8StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_8/stack:output:0/while/gru_cell/strided_slice_8/stack_1:output:0/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2 
while/gru_cell/strided_slice_8?
while/gru_cell/BiasAdd_3BiasAdd!while/gru_cell/MatMul_3:product:0'while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_3?
$while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52&
$while/gru_cell/strided_slice_9/stack?
&while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2(
&while/gru_cell/strided_slice_9/stack_1?
&while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&while/gru_cell/strided_slice_9/stack_2?
while/gru_cell/strided_slice_9StridedSlicewhile/gru_cell/unstack:output:1-while/gru_cell/strided_slice_9/stack:output:0/while/gru_cell/strided_slice_9/stack_1:output:0/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52 
while/gru_cell/strided_slice_9?
while/gru_cell/BiasAdd_4BiasAdd!while/gru_cell/MatMul_4:product:0'while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_4?
while/gru_cell/addAddV2while/gru_cell/BiasAdd:output:0!while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add?
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Sigmoid?
while/gru_cell/add_1AddV2!while/gru_cell/BiasAdd_1:output:0!while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_1?
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Sigmoid_1?
while/gru_cell/ReadVariableOp_6ReadVariableOp*while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype02!
while/gru_cell/ReadVariableOp_6?
%while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2'
%while/gru_cell/strided_slice_10/stack?
'while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/gru_cell/strided_slice_10/stack_1?
'while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/gru_cell/strided_slice_10/stack_2?
while/gru_cell/strided_slice_10StridedSlice'while/gru_cell/ReadVariableOp_6:value:0.while/gru_cell/strided_slice_10/stack:output:00while/gru_cell/strided_slice_10/stack_1:output:00while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2!
while/gru_cell/strided_slice_10?
while/gru_cell/MatMul_5MatMulwhile/gru_cell/mul_5:z:0(while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/MatMul_5?
%while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2'
%while/gru_cell/strided_slice_11/stack?
'while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'while/gru_cell/strided_slice_11/stack_1?
'while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'while/gru_cell/strided_slice_11/stack_2?
while/gru_cell/strided_slice_11StridedSlicewhile/gru_cell/unstack:output:1.while/gru_cell/strided_slice_11/stack:output:00while/gru_cell/strided_slice_11/stack_1:output:00while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2!
while/gru_cell/strided_slice_11?
while/gru_cell/BiasAdd_5BiasAdd!while/gru_cell/MatMul_5:product:0(while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/BiasAdd_5?
while/gru_cell/mul_6Mulwhile/gru_cell/Sigmoid_1:y:0!while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_6?
while/gru_cell/add_2AddV2!while/gru_cell/BiasAdd_2:output:0while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_2~
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/Tanh?
while/gru_cell/mul_7Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_7q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
while/gru_cell/sub/x?
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/sub?
while/gru_cell/mul_8Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/mul_8?
while/gru_cell/add_3AddV2while/gru_cell/mul_7:z:0while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
while/gru_cell/add_3?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identitywhile/gru_cell/add_3:z:0^while/gru_cell/ReadVariableOp ^while/gru_cell/ReadVariableOp_1 ^while/gru_cell/ReadVariableOp_2 ^while/gru_cell/ReadVariableOp_3 ^while/gru_cell/ReadVariableOp_4 ^while/gru_cell/ReadVariableOp_5 ^while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:?????????52
while/Identity_4"V
(while_gru_cell_readvariableop_1_resource*while_gru_cell_readvariableop_1_resource_0"V
(while_gru_cell_readvariableop_4_resource*while_gru_cell_readvariableop_4_resource_0"R
&while_gru_cell_readvariableop_resource(while_gru_cell_readvariableop_resource_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????5: : :::2>
while/gru_cell/ReadVariableOpwhile/gru_cell/ReadVariableOp2B
while/gru_cell/ReadVariableOp_1while/gru_cell/ReadVariableOp_12B
while/gru_cell/ReadVariableOp_2while/gru_cell/ReadVariableOp_22B
while/gru_cell/ReadVariableOp_3while/gru_cell/ReadVariableOp_32B
while/gru_cell/ReadVariableOp_4while/gru_cell/ReadVariableOp_42B
while/gru_cell/ReadVariableOp_5while/gru_cell/ReadVariableOp_52B
while/gru_cell/ReadVariableOp_6while/gru_cell/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
: 
??
?
sequential_gru_while_body_19008:
6sequential_gru_while_sequential_gru_while_loop_counter@
<sequential_gru_while_sequential_gru_while_maximum_iterations$
 sequential_gru_while_placeholder&
"sequential_gru_while_placeholder_1&
"sequential_gru_while_placeholder_29
5sequential_gru_while_sequential_gru_strided_slice_1_0u
qsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0;
7sequential_gru_while_gru_cell_readvariableop_resource_0=
9sequential_gru_while_gru_cell_readvariableop_1_resource_0=
9sequential_gru_while_gru_cell_readvariableop_4_resource_0!
sequential_gru_while_identity#
sequential_gru_while_identity_1#
sequential_gru_while_identity_2#
sequential_gru_while_identity_3#
sequential_gru_while_identity_47
3sequential_gru_while_sequential_gru_strided_slice_1s
osequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor9
5sequential_gru_while_gru_cell_readvariableop_resource;
7sequential_gru_while_gru_cell_readvariableop_1_resource;
7sequential_gru_while_gru_cell_readvariableop_4_resource??,sequential/gru/while/gru_cell/ReadVariableOp?.sequential/gru/while/gru_cell/ReadVariableOp_1?.sequential/gru/while/gru_cell/ReadVariableOp_2?.sequential/gru/while/gru_cell/ReadVariableOp_3?.sequential/gru/while/gru_cell/ReadVariableOp_4?.sequential/gru/while/gru_cell/ReadVariableOp_5?.sequential/gru/while/gru_cell/ReadVariableOp_6?
Fsequential/gru/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   2H
Fsequential/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape?
8sequential/gru/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemqsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0 sequential_gru_while_placeholderOsequential/gru/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02:
8sequential/gru/while/TensorArrayV2Read/TensorListGetItem?
-sequential/gru/while/gru_cell/ones_like/ShapeShape?sequential/gru/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:2/
-sequential/gru/while/gru_cell/ones_like/Shape?
-sequential/gru/while/gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2/
-sequential/gru/while/gru_cell/ones_like/Const?
'sequential/gru/while/gru_cell/ones_likeFill6sequential/gru/while/gru_cell/ones_like/Shape:output:06sequential/gru/while/gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2)
'sequential/gru/while/gru_cell/ones_like?
/sequential/gru/while/gru_cell/ones_like_1/ShapeShape"sequential_gru_while_placeholder_2*
T0*
_output_shapes
:21
/sequential/gru/while/gru_cell/ones_like_1/Shape?
/sequential/gru/while/gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??21
/sequential/gru/while/gru_cell/ones_like_1/Const?
)sequential/gru/while/gru_cell/ones_like_1Fill8sequential/gru/while/gru_cell/ones_like_1/Shape:output:08sequential/gru/while/gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52+
)sequential/gru/while/gru_cell/ones_like_1?
,sequential/gru/while/gru_cell/ReadVariableOpReadVariableOp7sequential_gru_while_gru_cell_readvariableop_resource_0*
_output_shapes
:	?*
dtype02.
,sequential/gru/while/gru_cell/ReadVariableOp?
%sequential/gru/while/gru_cell/unstackUnpack4sequential/gru/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2'
%sequential/gru/while/gru_cell/unstack?
!sequential/gru/while/gru_cell/mulMul?sequential/gru/while/TensorArrayV2Read/TensorListGetItem:item:00sequential/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2#
!sequential/gru/while/gru_cell/mul?
#sequential/gru/while/gru_cell/mul_1Mul?sequential/gru/while/TensorArrayV2Read/TensorListGetItem:item:00sequential/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2%
#sequential/gru/while/gru_cell/mul_1?
#sequential/gru/while/gru_cell/mul_2Mul?sequential/gru/while/TensorArrayV2Read/TensorListGetItem:item:00sequential/gru/while/gru_cell/ones_like:output:0*
T0*(
_output_shapes
:??????????2%
#sequential/gru/while/gru_cell/mul_2?
.sequential/gru/while/gru_cell/ReadVariableOp_1ReadVariableOp9sequential_gru_while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype020
.sequential/gru/while/gru_cell/ReadVariableOp_1?
1sequential/gru/while/gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1sequential/gru/while/gru_cell/strided_slice/stack?
3sequential/gru/while/gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   25
3sequential/gru/while/gru_cell/strided_slice/stack_1?
3sequential/gru/while/gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3sequential/gru/while/gru_cell/strided_slice/stack_2?
+sequential/gru/while/gru_cell/strided_sliceStridedSlice6sequential/gru/while/gru_cell/ReadVariableOp_1:value:0:sequential/gru/while/gru_cell/strided_slice/stack:output:0<sequential/gru/while/gru_cell/strided_slice/stack_1:output:0<sequential/gru/while/gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2-
+sequential/gru/while/gru_cell/strided_slice?
$sequential/gru/while/gru_cell/MatMulMatMul%sequential/gru/while/gru_cell/mul:z:04sequential/gru/while/gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52&
$sequential/gru/while/gru_cell/MatMul?
.sequential/gru/while/gru_cell/ReadVariableOp_2ReadVariableOp9sequential_gru_while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype020
.sequential/gru/while/gru_cell/ReadVariableOp_2?
3sequential/gru/while/gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   25
3sequential/gru/while/gru_cell/strided_slice_1/stack?
5sequential/gru/while/gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   27
5sequential/gru/while/gru_cell/strided_slice_1/stack_1?
5sequential/gru/while/gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential/gru/while/gru_cell/strided_slice_1/stack_2?
-sequential/gru/while/gru_cell/strided_slice_1StridedSlice6sequential/gru/while/gru_cell/ReadVariableOp_2:value:0<sequential/gru/while/gru_cell/strided_slice_1/stack:output:0>sequential/gru/while/gru_cell/strided_slice_1/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2/
-sequential/gru/while/gru_cell/strided_slice_1?
&sequential/gru/while/gru_cell/MatMul_1MatMul'sequential/gru/while/gru_cell/mul_1:z:06sequential/gru/while/gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52(
&sequential/gru/while/gru_cell/MatMul_1?
.sequential/gru/while/gru_cell/ReadVariableOp_3ReadVariableOp9sequential_gru_while_gru_cell_readvariableop_1_resource_0* 
_output_shapes
:
??*
dtype020
.sequential/gru/while/gru_cell/ReadVariableOp_3?
3sequential/gru/while/gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   25
3sequential/gru/while/gru_cell/strided_slice_2/stack?
5sequential/gru/while/gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        27
5sequential/gru/while/gru_cell/strided_slice_2/stack_1?
5sequential/gru/while/gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential/gru/while/gru_cell/strided_slice_2/stack_2?
-sequential/gru/while/gru_cell/strided_slice_2StridedSlice6sequential/gru/while/gru_cell/ReadVariableOp_3:value:0<sequential/gru/while/gru_cell/strided_slice_2/stack:output:0>sequential/gru/while/gru_cell/strided_slice_2/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2/
-sequential/gru/while/gru_cell/strided_slice_2?
&sequential/gru/while/gru_cell/MatMul_2MatMul'sequential/gru/while/gru_cell/mul_2:z:06sequential/gru/while/gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52(
&sequential/gru/while/gru_cell/MatMul_2?
3sequential/gru/while/gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential/gru/while/gru_cell/strided_slice_3/stack?
5sequential/gru/while/gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:527
5sequential/gru/while/gru_cell/strided_slice_3/stack_1?
5sequential/gru/while/gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/gru/while/gru_cell/strided_slice_3/stack_2?
-sequential/gru/while/gru_cell/strided_slice_3StridedSlice.sequential/gru/while/gru_cell/unstack:output:0<sequential/gru/while/gru_cell/strided_slice_3/stack:output:0>sequential/gru/while/gru_cell/strided_slice_3/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2/
-sequential/gru/while/gru_cell/strided_slice_3?
%sequential/gru/while/gru_cell/BiasAddBiasAdd.sequential/gru/while/gru_cell/MatMul:product:06sequential/gru/while/gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52'
%sequential/gru/while/gru_cell/BiasAdd?
3sequential/gru/while/gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:525
3sequential/gru/while/gru_cell/strided_slice_4/stack?
5sequential/gru/while/gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j27
5sequential/gru/while/gru_cell/strided_slice_4/stack_1?
5sequential/gru/while/gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/gru/while/gru_cell/strided_slice_4/stack_2?
-sequential/gru/while/gru_cell/strided_slice_4StridedSlice.sequential/gru/while/gru_cell/unstack:output:0<sequential/gru/while/gru_cell/strided_slice_4/stack:output:0>sequential/gru/while/gru_cell/strided_slice_4/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52/
-sequential/gru/while/gru_cell/strided_slice_4?
'sequential/gru/while/gru_cell/BiasAdd_1BiasAdd0sequential/gru/while/gru_cell/MatMul_1:product:06sequential/gru/while/gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52)
'sequential/gru/while/gru_cell/BiasAdd_1?
3sequential/gru/while/gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j25
3sequential/gru/while/gru_cell/strided_slice_5/stack?
5sequential/gru/while/gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 27
5sequential/gru/while/gru_cell/strided_slice_5/stack_1?
5sequential/gru/while/gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/gru/while/gru_cell/strided_slice_5/stack_2?
-sequential/gru/while/gru_cell/strided_slice_5StridedSlice.sequential/gru/while/gru_cell/unstack:output:0<sequential/gru/while/gru_cell/strided_slice_5/stack:output:0>sequential/gru/while/gru_cell/strided_slice_5/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2/
-sequential/gru/while/gru_cell/strided_slice_5?
'sequential/gru/while/gru_cell/BiasAdd_2BiasAdd0sequential/gru/while/gru_cell/MatMul_2:product:06sequential/gru/while/gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52)
'sequential/gru/while/gru_cell/BiasAdd_2?
#sequential/gru/while/gru_cell/mul_3Mul"sequential_gru_while_placeholder_22sequential/gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52%
#sequential/gru/while/gru_cell/mul_3?
#sequential/gru/while/gru_cell/mul_4Mul"sequential_gru_while_placeholder_22sequential/gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52%
#sequential/gru/while/gru_cell/mul_4?
#sequential/gru/while/gru_cell/mul_5Mul"sequential_gru_while_placeholder_22sequential/gru/while/gru_cell/ones_like_1:output:0*
T0*'
_output_shapes
:?????????52%
#sequential/gru/while/gru_cell/mul_5?
.sequential/gru/while/gru_cell/ReadVariableOp_4ReadVariableOp9sequential_gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype020
.sequential/gru/while/gru_cell/ReadVariableOp_4?
3sequential/gru/while/gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3sequential/gru/while/gru_cell/strided_slice_6/stack?
5sequential/gru/while/gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   27
5sequential/gru/while/gru_cell/strided_slice_6/stack_1?
5sequential/gru/while/gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential/gru/while/gru_cell/strided_slice_6/stack_2?
-sequential/gru/while/gru_cell/strided_slice_6StridedSlice6sequential/gru/while/gru_cell/ReadVariableOp_4:value:0<sequential/gru/while/gru_cell/strided_slice_6/stack:output:0>sequential/gru/while/gru_cell/strided_slice_6/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2/
-sequential/gru/while/gru_cell/strided_slice_6?
&sequential/gru/while/gru_cell/MatMul_3MatMul'sequential/gru/while/gru_cell/mul_3:z:06sequential/gru/while/gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52(
&sequential/gru/while/gru_cell/MatMul_3?
.sequential/gru/while/gru_cell/ReadVariableOp_5ReadVariableOp9sequential_gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype020
.sequential/gru/while/gru_cell/ReadVariableOp_5?
3sequential/gru/while/gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   25
3sequential/gru/while/gru_cell/strided_slice_7/stack?
5sequential/gru/while/gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   27
5sequential/gru/while/gru_cell/strided_slice_7/stack_1?
5sequential/gru/while/gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential/gru/while/gru_cell/strided_slice_7/stack_2?
-sequential/gru/while/gru_cell/strided_slice_7StridedSlice6sequential/gru/while/gru_cell/ReadVariableOp_5:value:0<sequential/gru/while/gru_cell/strided_slice_7/stack:output:0>sequential/gru/while/gru_cell/strided_slice_7/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2/
-sequential/gru/while/gru_cell/strided_slice_7?
&sequential/gru/while/gru_cell/MatMul_4MatMul'sequential/gru/while/gru_cell/mul_4:z:06sequential/gru/while/gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52(
&sequential/gru/while/gru_cell/MatMul_4?
3sequential/gru/while/gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 25
3sequential/gru/while/gru_cell/strided_slice_8/stack?
5sequential/gru/while/gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:527
5sequential/gru/while/gru_cell/strided_slice_8/stack_1?
5sequential/gru/while/gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/gru/while/gru_cell/strided_slice_8/stack_2?
-sequential/gru/while/gru_cell/strided_slice_8StridedSlice.sequential/gru/while/gru_cell/unstack:output:1<sequential/gru/while/gru_cell/strided_slice_8/stack:output:0>sequential/gru/while/gru_cell/strided_slice_8/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2/
-sequential/gru/while/gru_cell/strided_slice_8?
'sequential/gru/while/gru_cell/BiasAdd_3BiasAdd0sequential/gru/while/gru_cell/MatMul_3:product:06sequential/gru/while/gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52)
'sequential/gru/while/gru_cell/BiasAdd_3?
3sequential/gru/while/gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:525
3sequential/gru/while/gru_cell/strided_slice_9/stack?
5sequential/gru/while/gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j27
5sequential/gru/while/gru_cell/strided_slice_9/stack_1?
5sequential/gru/while/gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:27
5sequential/gru/while/gru_cell/strided_slice_9/stack_2?
-sequential/gru/while/gru_cell/strided_slice_9StridedSlice.sequential/gru/while/gru_cell/unstack:output:1<sequential/gru/while/gru_cell/strided_slice_9/stack:output:0>sequential/gru/while/gru_cell/strided_slice_9/stack_1:output:0>sequential/gru/while/gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52/
-sequential/gru/while/gru_cell/strided_slice_9?
'sequential/gru/while/gru_cell/BiasAdd_4BiasAdd0sequential/gru/while/gru_cell/MatMul_4:product:06sequential/gru/while/gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52)
'sequential/gru/while/gru_cell/BiasAdd_4?
!sequential/gru/while/gru_cell/addAddV2.sequential/gru/while/gru_cell/BiasAdd:output:00sequential/gru/while/gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52#
!sequential/gru/while/gru_cell/add?
%sequential/gru/while/gru_cell/SigmoidSigmoid%sequential/gru/while/gru_cell/add:z:0*
T0*'
_output_shapes
:?????????52'
%sequential/gru/while/gru_cell/Sigmoid?
#sequential/gru/while/gru_cell/add_1AddV20sequential/gru/while/gru_cell/BiasAdd_1:output:00sequential/gru/while/gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52%
#sequential/gru/while/gru_cell/add_1?
'sequential/gru/while/gru_cell/Sigmoid_1Sigmoid'sequential/gru/while/gru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52)
'sequential/gru/while/gru_cell/Sigmoid_1?
.sequential/gru/while/gru_cell/ReadVariableOp_6ReadVariableOp9sequential_gru_while_gru_cell_readvariableop_4_resource_0*
_output_shapes
:	5?*
dtype020
.sequential/gru/while/gru_cell/ReadVariableOp_6?
4sequential/gru/while/gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   26
4sequential/gru/while/gru_cell/strided_slice_10/stack?
6sequential/gru/while/gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        28
6sequential/gru/while/gru_cell/strided_slice_10/stack_1?
6sequential/gru/while/gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      28
6sequential/gru/while/gru_cell/strided_slice_10/stack_2?
.sequential/gru/while/gru_cell/strided_slice_10StridedSlice6sequential/gru/while/gru_cell/ReadVariableOp_6:value:0=sequential/gru/while/gru_cell/strided_slice_10/stack:output:0?sequential/gru/while/gru_cell/strided_slice_10/stack_1:output:0?sequential/gru/while/gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask20
.sequential/gru/while/gru_cell/strided_slice_10?
&sequential/gru/while/gru_cell/MatMul_5MatMul'sequential/gru/while/gru_cell/mul_5:z:07sequential/gru/while/gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52(
&sequential/gru/while/gru_cell/MatMul_5?
4sequential/gru/while/gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j26
4sequential/gru/while/gru_cell/strided_slice_11/stack?
6sequential/gru/while/gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 28
6sequential/gru/while/gru_cell/strided_slice_11/stack_1?
6sequential/gru/while/gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential/gru/while/gru_cell/strided_slice_11/stack_2?
.sequential/gru/while/gru_cell/strided_slice_11StridedSlice.sequential/gru/while/gru_cell/unstack:output:1=sequential/gru/while/gru_cell/strided_slice_11/stack:output:0?sequential/gru/while/gru_cell/strided_slice_11/stack_1:output:0?sequential/gru/while/gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask20
.sequential/gru/while/gru_cell/strided_slice_11?
'sequential/gru/while/gru_cell/BiasAdd_5BiasAdd0sequential/gru/while/gru_cell/MatMul_5:product:07sequential/gru/while/gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52)
'sequential/gru/while/gru_cell/BiasAdd_5?
#sequential/gru/while/gru_cell/mul_6Mul+sequential/gru/while/gru_cell/Sigmoid_1:y:00sequential/gru/while/gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52%
#sequential/gru/while/gru_cell/mul_6?
#sequential/gru/while/gru_cell/add_2AddV20sequential/gru/while/gru_cell/BiasAdd_2:output:0'sequential/gru/while/gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52%
#sequential/gru/while/gru_cell/add_2?
"sequential/gru/while/gru_cell/TanhTanh'sequential/gru/while/gru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52$
"sequential/gru/while/gru_cell/Tanh?
#sequential/gru/while/gru_cell/mul_7Mul)sequential/gru/while/gru_cell/Sigmoid:y:0"sequential_gru_while_placeholder_2*
T0*'
_output_shapes
:?????????52%
#sequential/gru/while/gru_cell/mul_7?
#sequential/gru/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2%
#sequential/gru/while/gru_cell/sub/x?
!sequential/gru/while/gru_cell/subSub,sequential/gru/while/gru_cell/sub/x:output:0)sequential/gru/while/gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52#
!sequential/gru/while/gru_cell/sub?
#sequential/gru/while/gru_cell/mul_8Mul%sequential/gru/while/gru_cell/sub:z:0&sequential/gru/while/gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52%
#sequential/gru/while/gru_cell/mul_8?
#sequential/gru/while/gru_cell/add_3AddV2'sequential/gru/while/gru_cell/mul_7:z:0'sequential/gru/while/gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52%
#sequential/gru/while/gru_cell/add_3?
9sequential/gru/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem"sequential_gru_while_placeholder_1 sequential_gru_while_placeholder'sequential/gru/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02;
9sequential/gru/while/TensorArrayV2Write/TensorListSetItemz
sequential/gru/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/gru/while/add/y?
sequential/gru/while/addAddV2 sequential_gru_while_placeholder#sequential/gru/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/while/add~
sequential/gru/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/gru/while/add_1/y?
sequential/gru/while/add_1AddV26sequential_gru_while_sequential_gru_while_loop_counter%sequential/gru/while/add_1/y:output:0*
T0*
_output_shapes
: 2
sequential/gru/while/add_1?
sequential/gru/while/IdentityIdentitysequential/gru/while/add_1:z:0-^sequential/gru/while/gru_cell/ReadVariableOp/^sequential/gru/while/gru_cell/ReadVariableOp_1/^sequential/gru/while/gru_cell/ReadVariableOp_2/^sequential/gru/while/gru_cell/ReadVariableOp_3/^sequential/gru/while/gru_cell/ReadVariableOp_4/^sequential/gru/while/gru_cell/ReadVariableOp_5/^sequential/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2
sequential/gru/while/Identity?
sequential/gru/while/Identity_1Identity<sequential_gru_while_sequential_gru_while_maximum_iterations-^sequential/gru/while/gru_cell/ReadVariableOp/^sequential/gru/while/gru_cell/ReadVariableOp_1/^sequential/gru/while/gru_cell/ReadVariableOp_2/^sequential/gru/while/gru_cell/ReadVariableOp_3/^sequential/gru/while/gru_cell/ReadVariableOp_4/^sequential/gru/while/gru_cell/ReadVariableOp_5/^sequential/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2!
sequential/gru/while/Identity_1?
sequential/gru/while/Identity_2Identitysequential/gru/while/add:z:0-^sequential/gru/while/gru_cell/ReadVariableOp/^sequential/gru/while/gru_cell/ReadVariableOp_1/^sequential/gru/while/gru_cell/ReadVariableOp_2/^sequential/gru/while/gru_cell/ReadVariableOp_3/^sequential/gru/while/gru_cell/ReadVariableOp_4/^sequential/gru/while/gru_cell/ReadVariableOp_5/^sequential/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2!
sequential/gru/while/Identity_2?
sequential/gru/while/Identity_3IdentityIsequential/gru/while/TensorArrayV2Write/TensorListSetItem:output_handle:0-^sequential/gru/while/gru_cell/ReadVariableOp/^sequential/gru/while/gru_cell/ReadVariableOp_1/^sequential/gru/while/gru_cell/ReadVariableOp_2/^sequential/gru/while/gru_cell/ReadVariableOp_3/^sequential/gru/while/gru_cell/ReadVariableOp_4/^sequential/gru/while/gru_cell/ReadVariableOp_5/^sequential/gru/while/gru_cell/ReadVariableOp_6*
T0*
_output_shapes
: 2!
sequential/gru/while/Identity_3?
sequential/gru/while/Identity_4Identity'sequential/gru/while/gru_cell/add_3:z:0-^sequential/gru/while/gru_cell/ReadVariableOp/^sequential/gru/while/gru_cell/ReadVariableOp_1/^sequential/gru/while/gru_cell/ReadVariableOp_2/^sequential/gru/while/gru_cell/ReadVariableOp_3/^sequential/gru/while/gru_cell/ReadVariableOp_4/^sequential/gru/while/gru_cell/ReadVariableOp_5/^sequential/gru/while/gru_cell/ReadVariableOp_6*
T0*'
_output_shapes
:?????????52!
sequential/gru/while/Identity_4"t
7sequential_gru_while_gru_cell_readvariableop_1_resource9sequential_gru_while_gru_cell_readvariableop_1_resource_0"t
7sequential_gru_while_gru_cell_readvariableop_4_resource9sequential_gru_while_gru_cell_readvariableop_4_resource_0"p
5sequential_gru_while_gru_cell_readvariableop_resource7sequential_gru_while_gru_cell_readvariableop_resource_0"G
sequential_gru_while_identity&sequential/gru/while/Identity:output:0"K
sequential_gru_while_identity_1(sequential/gru/while/Identity_1:output:0"K
sequential_gru_while_identity_2(sequential/gru/while/Identity_2:output:0"K
sequential_gru_while_identity_3(sequential/gru/while/Identity_3:output:0"K
sequential_gru_while_identity_4(sequential/gru/while/Identity_4:output:0"l
3sequential_gru_while_sequential_gru_strided_slice_15sequential_gru_while_sequential_gru_strided_slice_1_0"?
osequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensorqsequential_gru_while_tensorarrayv2read_tensorlistgetitem_sequential_gru_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????5: : :::2\
,sequential/gru/while/gru_cell/ReadVariableOp,sequential/gru/while/gru_cell/ReadVariableOp2`
.sequential/gru/while/gru_cell/ReadVariableOp_1.sequential/gru/while/gru_cell/ReadVariableOp_12`
.sequential/gru/while/gru_cell/ReadVariableOp_2.sequential/gru/while/gru_cell/ReadVariableOp_22`
.sequential/gru/while/gru_cell/ReadVariableOp_3.sequential/gru/while/gru_cell/ReadVariableOp_32`
.sequential/gru/while/gru_cell/ReadVariableOp_4.sequential/gru/while/gru_cell/ReadVariableOp_42`
.sequential/gru/while/gru_cell/ReadVariableOp_5.sequential/gru/while/gru_cell/ReadVariableOp_52`
.sequential/gru/while/gru_cell/ReadVariableOp_6.sequential/gru/while/gru_cell/ReadVariableOp_6: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
: 
?
?
while_cond_23303
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_23303___redundant_placeholder03
/while_while_cond_23303___redundant_placeholder13
/while_while_cond_23303___redundant_placeholder23
/while_while_cond_23303___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????5: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
:
?
?
sequential_gru_while_cond_19007:
6sequential_gru_while_sequential_gru_while_loop_counter@
<sequential_gru_while_sequential_gru_while_maximum_iterations$
 sequential_gru_while_placeholder&
"sequential_gru_while_placeholder_1&
"sequential_gru_while_placeholder_2<
8sequential_gru_while_less_sequential_gru_strided_slice_1Q
Msequential_gru_while_sequential_gru_while_cond_19007___redundant_placeholder0Q
Msequential_gru_while_sequential_gru_while_cond_19007___redundant_placeholder1Q
Msequential_gru_while_sequential_gru_while_cond_19007___redundant_placeholder2Q
Msequential_gru_while_sequential_gru_while_cond_19007___redundant_placeholder3!
sequential_gru_while_identity
?
sequential/gru/while/LessLess sequential_gru_while_placeholder8sequential_gru_while_less_sequential_gru_strided_slice_1*
T0*
_output_shapes
: 2
sequential/gru/while/Less?
sequential/gru/while/IdentityIdentitysequential/gru/while/Less:z:0*
T0
*
_output_shapes
: 2
sequential/gru/while/Identity"G
sequential_gru_while_identity&sequential/gru/while/Identity:output:0*@
_input_shapes/
-: : : : :?????????5: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
:
?
?
3__inference_batch_normalization_layer_call_fn_22028

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:???????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_193102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:???????????????????2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:???????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:???????????????????
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_21062

inputs
batch_normalization_21030
batch_normalization_21032
batch_normalization_21034
batch_normalization_21036
	gru_21039
	gru_21041
	gru_21043
dense_21046
dense_21048
dense_1_21051
dense_1_21053
dense_2_21056
dense_2_21058
identity??+batch_normalization/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?gru/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_21030batch_normalization_21032batch_normalization_21034batch_normalization_21036*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????5?*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_201142-
+batch_normalization/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0	gru_21039	gru_21041	gru_21043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_208112
gru/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_21046dense_21048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_208522
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_21051dense_1_21053*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_208792!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_21056dense_2_21058*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_209062!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????5?:::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
?
?
#__inference_gru_layer_call_fn_22783

inputs
unknown
	unknown_0
	unknown_1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_205282
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????52

Identity"
identityIdentity:output:0*7
_input_shapes&
$:?????????5?:::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
?m
?
C__inference_gru_cell_layer_call_and_return_conditional_losses_23790

inputs
states_0
readvariableop_resource
readvariableop_1_resource
readvariableop_4_resource
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
	ones_like^
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
ones_like_1y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack`
mulMulinputsones_like:output:0*
T0*(
_output_shapes
:??????????2
muld
mul_1Mulinputsones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_1d
mul_2Mulinputsones_like:output:0*
T0*(
_output_shapes
:??????????2
mul_2?
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????52
MatMul?
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52

MatMul_1?
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52

MatMul_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceunstack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52	
BiasAddx
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_1x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_2g
mul_3Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
mul_3g
mul_4Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
mul_4g
mul_5Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:?????????52
mul_5
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:	5?*
dtype02
ReadVariableOp_4
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_4:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
strided_slice_6u
MatMul_3MatMul	mul_3:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52

MatMul_3
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource*
_output_shapes
:	5?*
dtype02
ReadVariableOp_5
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_5:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
strided_slice_7u
MatMul_4MatMul	mul_4:z:0strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52

MatMul_4x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
strided_slice_8?
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_3x
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_9/stack|
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2?
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
strided_slice_9?
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????52	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????52
	Sigmoid_1
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes
:	5?*
dtype02
ReadVariableOp_6?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_6:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
strided_slice_10v
MatMul_5MatMul	mul_5:z:0strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52

MatMul_5z
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2?
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
strided_slice_11?
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:?????????52
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????52
Tanh^
mul_7MulSigmoid:y:0states_0*
T0*'
_output_shapes
:?????????52
mul_7S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????52
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:?????????52
add_3?
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*'
_output_shapes
:?????????52

Identity?

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*'
_output_shapes
:?????????52

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????5:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_6:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????5
"
_user_specified_name
states/0
??
?
>__inference_gru_layer_call_and_return_conditional_losses_23173
inputs_0$
 gru_cell_readvariableop_resource&
"gru_cell_readvariableop_1_resource&
"gru_cell_readvariableop_4_resource
identity??gru_cell/ReadVariableOp?gru_cell/ReadVariableOp_1?gru_cell/ReadVariableOp_2?gru_cell/ReadVariableOp_3?gru_cell/ReadVariableOp_4?gru_cell/ReadVariableOp_5?gru_cell/ReadVariableOp_6?whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice\
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :52
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :?2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessb
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :52
zeros/packed/1?
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constu
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:?????????52
zerosu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm?
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:???????????????????2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
TensorArrayV2/element_shape?
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2?
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   27
5TensorArrayUnstack/TensorListFromTensor/element_shape?
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????*
shrink_axis_mask2
strided_slice_2|
gru_cell/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like/Shapey
gru_cell/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like/Const?
gru_cell/ones_likeFill!gru_cell/ones_like/Shape:output:0!gru_cell/ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/ones_likeu
gru_cell/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
gru_cell/dropout/Const?
gru_cell/dropout/MulMulgru_cell/ones_like:output:0gru_cell/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/Mul{
gru_cell/dropout/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout/Shape?
-gru_cell/dropout/random_uniform/RandomUniformRandomUniformgru_cell/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2/
-gru_cell/dropout/random_uniform/RandomUniform?
gru_cell/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2!
gru_cell/dropout/GreaterEqual/y?
gru_cell/dropout/GreaterEqualGreaterEqual6gru_cell/dropout/random_uniform/RandomUniform:output:0(gru_cell/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/GreaterEqual?
gru_cell/dropout/CastCast!gru_cell/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout/Cast?
gru_cell/dropout/Mul_1Mulgru_cell/dropout/Mul:z:0gru_cell/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout/Mul_1y
gru_cell/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
gru_cell/dropout_1/Const?
gru_cell/dropout_1/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_1/Mul
gru_cell/dropout_1/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_1/Shape?
/gru_cell/dropout_1/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???21
/gru_cell/dropout_1/random_uniform/RandomUniform?
!gru_cell/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!gru_cell/dropout_1/GreaterEqual/y?
gru_cell/dropout_1/GreaterEqualGreaterEqual8gru_cell/dropout_1/random_uniform/RandomUniform:output:0*gru_cell/dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell/dropout_1/GreaterEqual?
gru_cell/dropout_1/CastCast#gru_cell/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout_1/Cast?
gru_cell/dropout_1/Mul_1Mulgru_cell/dropout_1/Mul:z:0gru_cell/dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_1/Mul_1y
gru_cell/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
gru_cell/dropout_2/Const?
gru_cell/dropout_2/MulMulgru_cell/ones_like:output:0!gru_cell/dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_2/Mul
gru_cell/dropout_2/ShapeShapegru_cell/ones_like:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_2/Shape?
/gru_cell/dropout_2/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???21
/gru_cell/dropout_2/random_uniform/RandomUniform?
!gru_cell/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2#
!gru_cell/dropout_2/GreaterEqual/y?
gru_cell/dropout_2/GreaterEqualGreaterEqual8gru_cell/dropout_2/random_uniform/RandomUniform:output:0*gru_cell/dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
gru_cell/dropout_2/GreaterEqual?
gru_cell/dropout_2/CastCast#gru_cell/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
gru_cell/dropout_2/Cast?
gru_cell/dropout_2/Mul_1Mulgru_cell/dropout_2/Mul:z:0gru_cell/dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
gru_cell/dropout_2/Mul_1v
gru_cell/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:2
gru_cell/ones_like_1/Shape}
gru_cell/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/ones_like_1/Const?
gru_cell/ones_like_1Fill#gru_cell/ones_like_1/Shape:output:0#gru_cell/ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/ones_like_1y
gru_cell/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/dropout_3/Const?
gru_cell/dropout_3/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_3/Mul?
gru_cell/dropout_3/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_3/Shape?
/gru_cell/dropout_3/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2Ȱ?21
/gru_cell/dropout_3/random_uniform/RandomUniform?
!gru_cell/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!gru_cell/dropout_3/GreaterEqual/y?
gru_cell/dropout_3/GreaterEqualGreaterEqual8gru_cell/dropout_3/random_uniform/RandomUniform:output:0*gru_cell/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52!
gru_cell/dropout_3/GreaterEqual?
gru_cell/dropout_3/CastCast#gru_cell/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
gru_cell/dropout_3/Cast?
gru_cell/dropout_3/Mul_1Mulgru_cell/dropout_3/Mul:z:0gru_cell/dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_3/Mul_1y
gru_cell/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/dropout_4/Const?
gru_cell/dropout_4/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_4/Mul?
gru_cell/dropout_4/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_4/Shape?
/gru_cell/dropout_4/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???21
/gru_cell/dropout_4/random_uniform/RandomUniform?
!gru_cell/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!gru_cell/dropout_4/GreaterEqual/y?
gru_cell/dropout_4/GreaterEqualGreaterEqual8gru_cell/dropout_4/random_uniform/RandomUniform:output:0*gru_cell/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52!
gru_cell/dropout_4/GreaterEqual?
gru_cell/dropout_4/CastCast#gru_cell/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
gru_cell/dropout_4/Cast?
gru_cell/dropout_4/Mul_1Mulgru_cell/dropout_4/Mul:z:0gru_cell/dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_4/Mul_1y
gru_cell/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/dropout_5/Const?
gru_cell/dropout_5/MulMulgru_cell/ones_like_1:output:0!gru_cell/dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_5/Mul?
gru_cell/dropout_5/ShapeShapegru_cell/ones_like_1:output:0*
T0*
_output_shapes
:2
gru_cell/dropout_5/Shape?
/gru_cell/dropout_5/random_uniform/RandomUniformRandomUniform!gru_cell/dropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???21
/gru_cell/dropout_5/random_uniform/RandomUniform?
!gru_cell/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!gru_cell/dropout_5/GreaterEqual/y?
gru_cell/dropout_5/GreaterEqualGreaterEqual8gru_cell/dropout_5/random_uniform/RandomUniform:output:0*gru_cell/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52!
gru_cell/dropout_5/GreaterEqual?
gru_cell/dropout_5/CastCast#gru_cell/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
gru_cell/dropout_5/Cast?
gru_cell/dropout_5/Mul_1Mulgru_cell/dropout_5/Mul:z:0gru_cell/dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/dropout_5/Mul_1?
gru_cell/ReadVariableOpReadVariableOp gru_cell_readvariableop_resource*
_output_shapes
:	?*
dtype02
gru_cell/ReadVariableOp?
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2
gru_cell/unstack?
gru_cell/mulMulstrided_slice_2:output:0gru_cell/dropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul?
gru_cell/mul_1Mulstrided_slice_2:output:0gru_cell/dropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_1?
gru_cell/mul_2Mulstrided_slice_2:output:0gru_cell/dropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
gru_cell/mul_2?
gru_cell/ReadVariableOp_1ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_1?
gru_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
gru_cell/strided_slice/stack?
gru_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice/stack_1?
gru_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2 
gru_cell/strided_slice/stack_2?
gru_cell/strided_sliceStridedSlice!gru_cell/ReadVariableOp_1:value:0%gru_cell/strided_slice/stack:output:0'gru_cell/strided_slice/stack_1:output:0'gru_cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice?
gru_cell/MatMulMatMulgru_cell/mul:z:0gru_cell/strided_slice:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul?
gru_cell/ReadVariableOp_2ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_2?
gru_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice_1/stack?
 gru_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2"
 gru_cell/strided_slice_1/stack_1?
 gru_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_1/stack_2?
gru_cell/strided_slice_1StridedSlice!gru_cell/ReadVariableOp_2:value:0'gru_cell/strided_slice_1/stack:output:0)gru_cell/strided_slice_1/stack_1:output:0)gru_cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice_1?
gru_cell/MatMul_1MatMulgru_cell/mul_1:z:0!gru_cell/strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_1?
gru_cell/ReadVariableOp_3ReadVariableOp"gru_cell_readvariableop_1_resource* 
_output_shapes
:
??*
dtype02
gru_cell/ReadVariableOp_3?
gru_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2 
gru_cell/strided_slice_2/stack?
 gru_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2"
 gru_cell/strided_slice_2/stack_1?
 gru_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_2/stack_2?
gru_cell/strided_slice_2StridedSlice!gru_cell/ReadVariableOp_3:value:0'gru_cell/strided_slice_2/stack:output:0)gru_cell/strided_slice_2/stack_1:output:0)gru_cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
gru_cell/strided_slice_2?
gru_cell/MatMul_2MatMulgru_cell/mul_2:z:0!gru_cell/strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_2?
gru_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_3/stack?
 gru_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52"
 gru_cell/strided_slice_3/stack_1?
 gru_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_3/stack_2?
gru_cell/strided_slice_3StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_3/stack:output:0)gru_cell/strided_slice_3/stack_1:output:0)gru_cell/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru_cell/strided_slice_3?
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0!gru_cell/strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd?
gru_cell/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52 
gru_cell/strided_slice_4/stack?
 gru_cell/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2"
 gru_cell/strided_slice_4/stack_1?
 gru_cell/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_4/stack_2?
gru_cell/strided_slice_4StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_4/stack:output:0)gru_cell/strided_slice_4/stack_1:output:0)gru_cell/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru_cell/strided_slice_4?
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0!gru_cell/strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_1?
gru_cell/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2 
gru_cell/strided_slice_5/stack?
 gru_cell/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru_cell/strided_slice_5/stack_1?
 gru_cell/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_5/stack_2?
gru_cell/strided_slice_5StridedSlicegru_cell/unstack:output:0'gru_cell/strided_slice_5/stack:output:0)gru_cell/strided_slice_5/stack_1:output:0)gru_cell/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru_cell/strided_slice_5?
gru_cell/BiasAdd_2BiasAddgru_cell/MatMul_2:product:0!gru_cell/strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_2?
gru_cell/mul_3Mulzeros:output:0gru_cell/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_3?
gru_cell/mul_4Mulzeros:output:0gru_cell/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_4?
gru_cell/mul_5Mulzeros:output:0gru_cell/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_5?
gru_cell/ReadVariableOp_4ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_4?
gru_cell/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2 
gru_cell/strided_slice_6/stack?
 gru_cell/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2"
 gru_cell/strided_slice_6/stack_1?
 gru_cell/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_6/stack_2?
gru_cell/strided_slice_6StridedSlice!gru_cell/ReadVariableOp_4:value:0'gru_cell/strided_slice_6/stack:output:0)gru_cell/strided_slice_6/stack_1:output:0)gru_cell/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_6?
gru_cell/MatMul_3MatMulgru_cell/mul_3:z:0!gru_cell/strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_3?
gru_cell/ReadVariableOp_5ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_5?
gru_cell/strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2 
gru_cell/strided_slice_7/stack?
 gru_cell/strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2"
 gru_cell/strided_slice_7/stack_1?
 gru_cell/strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2"
 gru_cell/strided_slice_7/stack_2?
gru_cell/strided_slice_7StridedSlice!gru_cell/ReadVariableOp_5:value:0'gru_cell/strided_slice_7/stack:output:0)gru_cell/strided_slice_7/stack_1:output:0)gru_cell/strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_7?
gru_cell/MatMul_4MatMulgru_cell/mul_4:z:0!gru_cell/strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_4?
gru_cell/strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru_cell/strided_slice_8/stack?
 gru_cell/strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52"
 gru_cell/strided_slice_8/stack_1?
 gru_cell/strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_8/stack_2?
gru_cell/strided_slice_8StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_8/stack:output:0)gru_cell/strided_slice_8/stack_1:output:0)gru_cell/strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
gru_cell/strided_slice_8?
gru_cell/BiasAdd_3BiasAddgru_cell/MatMul_3:product:0!gru_cell/strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_3?
gru_cell/strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52 
gru_cell/strided_slice_9/stack?
 gru_cell/strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2"
 gru_cell/strided_slice_9/stack_1?
 gru_cell/strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru_cell/strided_slice_9/stack_2?
gru_cell/strided_slice_9StridedSlicegru_cell/unstack:output:1'gru_cell/strided_slice_9/stack:output:0)gru_cell/strided_slice_9/stack_1:output:0)gru_cell/strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
gru_cell/strided_slice_9?
gru_cell/BiasAdd_4BiasAddgru_cell/MatMul_4:product:0!gru_cell/strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_4?
gru_cell/addAddV2gru_cell/BiasAdd:output:0gru_cell/BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/adds
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Sigmoid?
gru_cell/add_1AddV2gru_cell/BiasAdd_1:output:0gru_cell/BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_1y
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Sigmoid_1?
gru_cell/ReadVariableOp_6ReadVariableOp"gru_cell_readvariableop_4_resource*
_output_shapes
:	5?*
dtype02
gru_cell/ReadVariableOp_6?
gru_cell/strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2!
gru_cell/strided_slice_10/stack?
!gru_cell/strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!gru_cell/strided_slice_10/stack_1?
!gru_cell/strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!gru_cell/strided_slice_10/stack_2?
gru_cell/strided_slice_10StridedSlice!gru_cell/ReadVariableOp_6:value:0(gru_cell/strided_slice_10/stack:output:0*gru_cell/strided_slice_10/stack_1:output:0*gru_cell/strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
gru_cell/strided_slice_10?
gru_cell/MatMul_5MatMulgru_cell/mul_5:z:0"gru_cell/strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/MatMul_5?
gru_cell/strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2!
gru_cell/strided_slice_11/stack?
!gru_cell/strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!gru_cell/strided_slice_11/stack_1?
!gru_cell/strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2#
!gru_cell/strided_slice_11/stack_2?
gru_cell/strided_slice_11StridedSlicegru_cell/unstack:output:1(gru_cell/strided_slice_11/stack:output:0*gru_cell/strided_slice_11/stack_1:output:0*gru_cell/strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
gru_cell/strided_slice_11?
gru_cell/BiasAdd_5BiasAddgru_cell/MatMul_5:product:0"gru_cell/strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/BiasAdd_5?
gru_cell/mul_6Mulgru_cell/Sigmoid_1:y:0gru_cell/BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_6?
gru_cell/add_2AddV2gru_cell/BiasAdd_2:output:0gru_cell/mul_6:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_2l
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/Tanh
gru_cell/mul_7Mulgru_cell/Sigmoid:y:0zeros:output:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_7e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
gru_cell/sub/x?
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/sub~
gru_cell/mul_8Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*'
_output_shapes
:?????????52
gru_cell/mul_8?
gru_cell/add_3AddV2gru_cell/mul_7:z:0gru_cell/mul_8:z:0*
T0*'
_output_shapes
:?????????52
gru_cell/add_3?
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   2
TensorArrayV2_1/element_shape?
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter?
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0 gru_cell_readvariableop_resource"gru_cell_readvariableop_1_resource"gru_cell_readvariableop_4_resource*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*9
_output_shapes'
%: : : : :?????????5: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_22973*
condR
while_cond_22972*8
output_shapes'
%: : : : :?????????5: : : : : *
parallel_iterations 2
while?
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????5   22
0TensorArrayV2Stack/TensorListStack/element_shape?
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :??????????????????5*
element_dtype02$
"TensorArrayV2Stack/TensorListStack?
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????5*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm?
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :??????????????????52
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime?
IdentityIdentitystrided_slice_3:output:0^gru_cell/ReadVariableOp^gru_cell/ReadVariableOp_1^gru_cell/ReadVariableOp_2^gru_cell/ReadVariableOp_3^gru_cell/ReadVariableOp_4^gru_cell/ReadVariableOp_5^gru_cell/ReadVariableOp_6^while*
T0*'
_output_shapes
:?????????52

Identity"
identityIdentity:output:0*@
_input_shapes/
-:???????????????????:::22
gru_cell/ReadVariableOpgru_cell/ReadVariableOp26
gru_cell/ReadVariableOp_1gru_cell/ReadVariableOp_126
gru_cell/ReadVariableOp_2gru_cell/ReadVariableOp_226
gru_cell/ReadVariableOp_3gru_cell/ReadVariableOp_326
gru_cell/ReadVariableOp_4gru_cell/ReadVariableOp_426
gru_cell/ReadVariableOp_5gru_cell/ReadVariableOp_526
gru_cell/ReadVariableOp_6gru_cell/ReadVariableOp_62
whilewhile:_ [
5
_output_shapes#
!:???????????????????
"
_user_specified_name
inputs/0
?	
?
(__inference_gru_cell_layer_call_fn_23818

inputs
states_0
unknown
	unknown_0
	unknown_1
identity

identity_1??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0unknown	unknown_0	unknown_1*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????5:?????????5*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_196052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????52

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????52

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????5:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:QM
'
_output_shapes
:?????????5
"
_user_specified_name
states/0
??
?
C__inference_gru_cell_layer_call_and_return_conditional_losses_19503

inputs

states
readvariableop_resource
readvariableop_1_resource
readvariableop_4_resource
identity

identity_1??ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?ReadVariableOp_3?ReadVariableOp_4?ReadVariableOp_5?ReadVariableOp_6X
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*(
_output_shapes
:??????????2
	ones_likec
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout/Const?
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/Mul`
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1g
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_1/Const?
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/Muld
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_1/Shape?
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_1/random_uniform/RandomUniformy
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout_1/GreaterEqual/y?
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_1/GreaterEqual?
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_1/Cast?
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_1/Mul_1g
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *?8??2
dropout_2/Const?
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/Muld
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:2
dropout_2/Shape?
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype0*
seed???)*
seed2???2(
&dropout_2/random_uniform/RandomUniformy
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *???=2
dropout_2/GreaterEqual/y?
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout_2/GreaterEqual?
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_2/Cast?
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_2/Mul_1\
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:2
ones_like_1/Shapek
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like_1/Const?
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:?????????52
ones_like_1g
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_3/Const?
dropout_3/MulMulones_like_1:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:?????????52
dropout_3/Mulf
dropout_3/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_3/Shape?
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???2(
&dropout_3/random_uniform/RandomUniformy
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_3/GreaterEqual/y?
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52
dropout_3/GreaterEqual?
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
dropout_3/Cast?
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:?????????52
dropout_3/Mul_1g
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_4/Const?
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:?????????52
dropout_4/Mulf
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_4/Shape?
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2??>2(
&dropout_4/random_uniform/RandomUniformy
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_4/GreaterEqual/y?
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52
dropout_4/GreaterEqual?
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
dropout_4/Cast?
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:?????????52
dropout_4/Mul_1g
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_5/Const?
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:?????????52
dropout_5/Mulf
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:2
dropout_5/Shape?
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:?????????5*
dtype0*
seed???)*
seed2???2(
&dropout_5/random_uniform/RandomUniformy
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout_5/GreaterEqual/y?
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:?????????52
dropout_5/GreaterEqual?
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:?????????52
dropout_5/Cast?
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:?????????52
dropout_5/Mul_1y
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:	?*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:?:?*	
num2	
unstack_
mulMulinputsdropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mule
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_1e
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*(
_output_shapes
:??????????2
mul_2?
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_1{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp_1:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
strided_slicem
MatMulMatMulmul:z:0strided_slice:output:0*
T0*'
_output_shapes
:?????????52
MatMul?
ReadVariableOp_2ReadVariableOpreadvariableop_1_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_2
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceReadVariableOp_2:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
strided_slice_1u
MatMul_1MatMul	mul_1:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:?????????52

MatMul_1?
ReadVariableOp_3ReadVariableOpreadvariableop_1_resource* 
_output_shapes
:
??*
dtype02
ReadVariableOp_3
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceReadVariableOp_3:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	?5*

begin_mask*
end_mask2
strided_slice_2u
MatMul_2MatMul	mul_2:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:?????????52

MatMul_2x
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceunstack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
strided_slice_3{
BiasAddBiasAddMatMul:product:0strided_slice_3:output:0*
T0*'
_output_shapes
:?????????52	
BiasAddx
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceunstack:output:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
strided_slice_4?
	BiasAdd_1BiasAddMatMul_1:product:0strided_slice_4:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_1x
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_5/stack|
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_5/stack_1|
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_5/stack_2?
strided_slice_5StridedSliceunstack:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
strided_slice_5?
	BiasAdd_2BiasAddMatMul_2:product:0strided_slice_5:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_2d
mul_3Mulstatesdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
mul_3d
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
mul_4d
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*'
_output_shapes
:?????????52
mul_5
ReadVariableOp_4ReadVariableOpreadvariableop_4_resource*
_output_shapes
:	5?*
dtype02
ReadVariableOp_4
strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_6/stack?
strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice_6/stack_1?
strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_6/stack_2?
strided_slice_6StridedSliceReadVariableOp_4:value:0strided_slice_6/stack:output:0 strided_slice_6/stack_1:output:0 strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
strided_slice_6u
MatMul_3MatMul	mul_3:z:0strided_slice_6:output:0*
T0*'
_output_shapes
:?????????52

MatMul_3
ReadVariableOp_5ReadVariableOpreadvariableop_4_resource*
_output_shapes
:	5?*
dtype02
ReadVariableOp_5
strided_slice_7/stackConst*
_output_shapes
:*
dtype0*
valueB"    5   2
strided_slice_7/stack?
strided_slice_7/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_7/stack_1?
strided_slice_7/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_7/stack_2?
strided_slice_7StridedSliceReadVariableOp_5:value:0strided_slice_7/stack:output:0 strided_slice_7/stack_1:output:0 strided_slice_7/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
strided_slice_7u
MatMul_4MatMul	mul_4:z:0strided_slice_7:output:0*
T0*'
_output_shapes
:?????????52

MatMul_4x
strided_slice_8/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_8/stack|
strided_slice_8/stack_1Const*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_8/stack_1|
strided_slice_8/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_8/stack_2?
strided_slice_8StridedSliceunstack:output:1strided_slice_8/stack:output:0 strided_slice_8/stack_1:output:0 strided_slice_8/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*

begin_mask2
strided_slice_8?
	BiasAdd_3BiasAddMatMul_3:product:0strided_slice_8:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_3x
strided_slice_9/stackConst*
_output_shapes
:*
dtype0*
valueB:52
strided_slice_9/stack|
strided_slice_9/stack_1Const*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_9/stack_1|
strided_slice_9/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_9/stack_2?
strided_slice_9StridedSliceunstack:output:1strided_slice_9/stack:output:0 strided_slice_9/stack_1:output:0 strided_slice_9/stack_2:output:0*
Index0*
T0*
_output_shapes
:52
strided_slice_9?
	BiasAdd_4BiasAddMatMul_4:product:0strided_slice_9:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_4k
addAddV2BiasAdd:output:0BiasAdd_3:output:0*
T0*'
_output_shapes
:?????????52
addX
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:?????????52	
Sigmoidq
add_1AddV2BiasAdd_1:output:0BiasAdd_4:output:0*
T0*'
_output_shapes
:?????????52
add_1^
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:?????????52
	Sigmoid_1
ReadVariableOp_6ReadVariableOpreadvariableop_4_resource*
_output_shapes
:	5?*
dtype02
ReadVariableOp_6?
strided_slice_10/stackConst*
_output_shapes
:*
dtype0*
valueB"    j   2
strided_slice_10/stack?
strided_slice_10/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_10/stack_1?
strided_slice_10/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_10/stack_2?
strided_slice_10StridedSliceReadVariableOp_6:value:0strided_slice_10/stack:output:0!strided_slice_10/stack_1:output:0!strided_slice_10/stack_2:output:0*
Index0*
T0*
_output_shapes

:55*

begin_mask*
end_mask2
strided_slice_10v
MatMul_5MatMul	mul_5:z:0strided_slice_10:output:0*
T0*'
_output_shapes
:?????????52

MatMul_5z
strided_slice_11/stackConst*
_output_shapes
:*
dtype0*
valueB:j2
strided_slice_11/stack~
strided_slice_11/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_11/stack_1~
strided_slice_11/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_11/stack_2?
strided_slice_11StridedSliceunstack:output:1strided_slice_11/stack:output:0!strided_slice_11/stack_1:output:0!strided_slice_11/stack_2:output:0*
Index0*
T0*
_output_shapes
:5*
end_mask2
strided_slice_11?
	BiasAdd_5BiasAddMatMul_5:product:0strided_slice_11:output:0*
T0*'
_output_shapes
:?????????52
	BiasAdd_5j
mul_6MulSigmoid_1:y:0BiasAdd_5:output:0*
T0*'
_output_shapes
:?????????52
mul_6h
add_2AddV2BiasAdd_2:output:0	mul_6:z:0*
T0*'
_output_shapes
:?????????52
add_2Q
TanhTanh	add_2:z:0*
T0*'
_output_shapes
:?????????52
Tanh\
mul_7MulSigmoid:y:0states*
T0*'
_output_shapes
:?????????52
mul_7S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/x`
subSubsub/x:output:0Sigmoid:y:0*
T0*'
_output_shapes
:?????????52
subZ
mul_8Mulsub:z:0Tanh:y:0*
T0*'
_output_shapes
:?????????52
mul_8_
add_3AddV2	mul_7:z:0	mul_8:z:0*
T0*'
_output_shapes
:?????????52
add_3?
IdentityIdentity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*'
_output_shapes
:?????????52

Identity?

Identity_1Identity	add_3:z:0^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^ReadVariableOp_4^ReadVariableOp_5^ReadVariableOp_6*
T0*'
_output_shapes
:?????????52

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*F
_input_shapes5
3:??????????:?????????5:::2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32$
ReadVariableOp_4ReadVariableOp_42$
ReadVariableOp_5ReadVariableOp_52$
ReadVariableOp_6ReadVariableOp_6:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????5
 
_user_specified_namestates
?!
?
while_body_19864
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_gru_cell_19886_0
while_gru_cell_19888_0
while_gru_cell_19890_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_gru_cell_19886
while_gru_cell_19888
while_gru_cell_19890??&while/gru_cell/StatefulPartitionedCall?
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"????   29
7while/TensorArrayV2Read/TensorListGetItem/element_shape?
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:??????????*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem?
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_gru_cell_19886_0while_gru_cell_19888_0while_gru_cell_19890_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:?????????5:?????????5*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_gru_cell_layer_call_and_return_conditional_losses_195032(
&while/gru_cell/StatefulPartitionedCall?
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1?
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity?
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1?
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2?
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3?
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????52
while/Identity_4".
while_gru_cell_19886while_gru_cell_19886_0".
while_gru_cell_19888while_gru_cell_19888_0".
while_gru_cell_19890while_gru_cell_19890_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"0
while_strided_slice_1while_strided_slice_1_0"?
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*>
_input_shapes-
+: : : : :?????????5: : :::2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
: 
??
?
!__inference__traced_restore_24049
file_prefix.
*assignvariableop_batch_normalization_gamma/
+assignvariableop_1_batch_normalization_beta6
2assignvariableop_2_batch_normalization_moving_mean:
6assignvariableop_3_batch_normalization_moving_variance#
assignvariableop_4_dense_kernel!
assignvariableop_5_dense_bias%
!assignvariableop_6_dense_1_kernel#
assignvariableop_7_dense_1_bias%
!assignvariableop_8_dense_2_kernel#
assignvariableop_9_dense_2_bias$
 assignvariableop_10_rmsprop_iter%
!assignvariableop_11_rmsprop_decay-
)assignvariableop_12_rmsprop_learning_rate(
$assignvariableop_13_rmsprop_momentum#
assignvariableop_14_rmsprop_rho+
'assignvariableop_15_gru_gru_cell_kernel5
1assignvariableop_16_gru_gru_cell_recurrent_kernel)
%assignvariableop_17_gru_gru_cell_bias
assignvariableop_18_total
assignvariableop_19_count
assignvariableop_20_total_1
assignvariableop_21_count_1=
9assignvariableop_22_rmsprop_batch_normalization_gamma_rms<
8assignvariableop_23_rmsprop_batch_normalization_beta_rms0
,assignvariableop_24_rmsprop_dense_kernel_rms.
*assignvariableop_25_rmsprop_dense_bias_rms2
.assignvariableop_26_rmsprop_dense_1_kernel_rms0
,assignvariableop_27_rmsprop_dense_1_bias_rms2
.assignvariableop_28_rmsprop_dense_2_kernel_rms0
,assignvariableop_29_rmsprop_dense_2_bias_rms7
3assignvariableop_30_rmsprop_gru_gru_cell_kernel_rmsA
=assignvariableop_31_rmsprop_gru_gru_cell_recurrent_kernel_rms5
1assignvariableop_32_rmsprop_gru_gru_cell_bias_rms
identity_34??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*?
value?B?"B5layer_with_weights-0/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-0/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-0/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-0/gamma/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/beta/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/6/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp*assignvariableop_batch_normalization_gammaIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp+assignvariableop_1_batch_normalization_betaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp2assignvariableop_2_batch_normalization_moving_meanIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp6assignvariableop_3_batch_normalization_moving_varianceIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp assignvariableop_10_rmsprop_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp!assignvariableop_11_rmsprop_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp)assignvariableop_12_rmsprop_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp$assignvariableop_13_rmsprop_momentumIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_rmsprop_rhoIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp'assignvariableop_15_gru_gru_cell_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp1assignvariableop_16_gru_gru_cell_recurrent_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp%assignvariableop_17_gru_gru_cell_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_total_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp9assignvariableop_22_rmsprop_batch_normalization_gamma_rmsIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp8assignvariableop_23_rmsprop_batch_normalization_beta_rmsIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp,assignvariableop_24_rmsprop_dense_kernel_rmsIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp*assignvariableop_25_rmsprop_dense_bias_rmsIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp.assignvariableop_26_rmsprop_dense_1_kernel_rmsIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp,assignvariableop_27_rmsprop_dense_1_bias_rmsIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp.assignvariableop_28_rmsprop_dense_2_kernel_rmsIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp,assignvariableop_29_rmsprop_dense_2_bias_rmsIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp3assignvariableop_30_rmsprop_gru_gru_cell_kernel_rmsIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp=assignvariableop_31_rmsprop_gru_gru_cell_recurrent_kernel_rmsIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp1assignvariableop_32_rmsprop_gru_gru_cell_bias_rmsIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33?
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_20923
batch_normalization_input
batch_normalization_20141
batch_normalization_20143
batch_normalization_20145
batch_normalization_20147
	gru_20834
	gru_20836
	gru_20838
dense_20863
dense_20865
dense_1_20890
dense_1_20892
dense_2_20917
dense_2_20919
identity??+batch_normalization/StatefulPartitionedCall?dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?gru/StatefulPartitionedCall?
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCallbatch_normalization_inputbatch_normalization_20141batch_normalization_20143batch_normalization_20145batch_normalization_20147*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????5?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_200942-
+batch_normalization/StatefulPartitionedCall?
gru/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0	gru_20834	gru_20836	gru_20838*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????5*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *G
fBR@
>__inference_gru_layer_call_and_return_conditional_losses_205282
gru/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall$gru/StatefulPartitionedCall:output:0dense_20863dense_20865*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_208522
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_20890dense_1_20892*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_208792!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_20917dense_2_20919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_209062!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*_
_input_shapesN
L:?????????5?:::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:g c
,
_output_shapes
:?????????5?
3
_user_specified_namebatch_normalization_input
?
?
while_cond_20658
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice_13
/while_while_cond_20658___redundant_placeholder03
/while_while_cond_20658___redundant_placeholder13
/while_while_cond_20658___redundant_placeholder23
/while_while_cond_20658___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*@
_input_shapes/
-: : : : :?????????5: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:?????????5:

_output_shapes
: :

_output_shapes
:
?
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22084

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identity??batchnorm/ReadVariableOp?batchnorm/ReadVariableOp_1?batchnorm/ReadVariableOp_2?batchnorm/mul/ReadVariableOp?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????5?2
batchnorm/mul_1?
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_1?
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp_2?
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????5?2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:?????????5?2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????5?::::24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs
?0
?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_20094

inputs
assignmovingavg_20069
assignmovingavg_1_20075)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identity??#AssignMovingAvg/AssignSubVariableOp?AssignMovingAvg/ReadVariableOp?%AssignMovingAvg_1/AssignSubVariableOp? AssignMovingAvg_1/ReadVariableOp?batchnorm/ReadVariableOp?batchnorm/mul/ReadVariableOp?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indices?
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*#
_output_shapes
:?2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:?????????5?2
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*#
_output_shapes
:?*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:?*
squeeze_dims
 2
moments/Squeeze_1?
AssignMovingAvg/decayConst",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/20069*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg/decay?
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_20069*
_output_shapes	
:?*
dtype02 
AssignMovingAvg/ReadVariableOp?
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/20069*
_output_shapes	
:?2
AssignMovingAvg/sub?
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*(
_class
loc:@AssignMovingAvg/20069*
_output_shapes	
:?2
AssignMovingAvg/mul?
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_20069AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*(
_class
loc:@AssignMovingAvg/20069*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp?
AssignMovingAvg_1/decayConst",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/20075*
_output_shapes
: *
dtype0*
valueB
 *
?#<2
AssignMovingAvg_1/decay?
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_20075*
_output_shapes	
:?*
dtype02"
 AssignMovingAvg_1/ReadVariableOp?
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/20075*
_output_shapes	
:?2
AssignMovingAvg_1/sub?
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0**
_class 
loc:@AssignMovingAvg_1/20075*
_output_shapes	
:?2
AssignMovingAvg_1/mul?
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_20075AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0**
_class 
loc:@AssignMovingAvg_1/20075*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2
batchnorm/add/y?
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:?2
batchnorm/addd
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:?2
batchnorm/Rsqrt?
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/mul/ReadVariableOp?
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:?2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:?????????5?2
batchnorm/mul_1|
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:?2
batchnorm/mul_2?
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:?*
dtype02
batchnorm/ReadVariableOp?
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:?2
batchnorm/sub?
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:?????????5?2
batchnorm/add_1?
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*
T0*,
_output_shapes
:?????????5?2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:?????????5?::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:T P
,
_output_shapes
:?????????5?
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
d
batch_normalization_inputG
+serving_default_batch_normalization_input:0?????????5?;
dense_20
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?:
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
	optimizer
	variables
trainable_variables
	regularization_losses

	keras_api

signatures
s__call__
t_default_save_signature
*u&call_and_return_all_conditional_losses"?7
_tf_keras_sequential?6{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 53, 512]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "batch_normalization_input"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 53, 512]}, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 53, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.1, "recurrent_dropout": 0.2, "implementation": 1, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 26, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 13, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 53, 512]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 53, 512]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "batch_normalization_input"}}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 53, 512]}, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 53, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.1, "recurrent_dropout": 0.2, "implementation": 1, "reset_after": true}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 26, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 13, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "binary_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
?

axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
regularization_losses
	keras_api
v__call__
*w&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 53, 512]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 53, 512]}, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 53, 512]}}
?
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
x__call__
*y&call_and_return_all_conditional_losses"?

_tf_keras_rnn_layer?	{"class_name": "GRU", "name": "gru", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 53, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.1, "recurrent_dropout": 0.2, "implementation": 1, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 512]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 53, 512]}}
?

kernel
bias
	variables
trainable_variables
regularization_losses
 	keras_api
z__call__
*{&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 26, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 53}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 53]}}
?

!kernel
"bias
#	variables
$trainable_variables
%regularization_losses
&	keras_api
|__call__
*}&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 13, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 26}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 26]}}
?

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
~__call__
*&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 13}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13]}}
?
-iter
	.decay
/learning_rate
0momentum
1rho	rmsh	rmsi	rmsj	rmsk	!rmsl	"rmsm	'rmsn	(rmso	2rmsp	3rmsq	4rmsr"
	optimizer
~
0
1
2
3
24
35
46
7
8
!9
"10
'11
(12"
trackable_list_wrapper
n
0
1
22
33
44
5
6
!7
"8
'9
(10"
trackable_list_wrapper
 "
trackable_list_wrapper
?
5non_trainable_variables
	variables
6metrics
7layer_metrics
8layer_regularization_losses
trainable_variables

9layers
	regularization_losses
s__call__
t_default_save_signature
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
(:&?2batch_normalization/gamma
':%?2batch_normalization/beta
0:.? (2batch_normalization/moving_mean
4:2? (2#batch_normalization/moving_variance
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
:non_trainable_variables
	variables
;metrics
<layer_metrics
=layer_regularization_losses
trainable_variables

>layers
regularization_losses
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
?

2kernel
3recurrent_kernel
4bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "GRUCell", "name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 53, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.1, "recurrent_dropout": 0.2, "implementation": 1, "reset_after": true}}
 "
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Cnon_trainable_variables
	variables
Dmetrics
Elayer_metrics
Flayer_regularization_losses
trainable_variables

Glayers
regularization_losses

Hstates
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
:52dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Inon_trainable_variables
	variables
Jmetrics
Klayer_metrics
Llayer_regularization_losses
trainable_variables

Mlayers
regularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nnon_trainable_variables
#	variables
Ometrics
Player_metrics
Qlayer_regularization_losses
$trainable_variables

Rlayers
%regularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 :2dense_2/kernel
:2dense_2/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Snon_trainable_variables
)	variables
Tmetrics
Ulayer_metrics
Vlayer_regularization_losses
*trainable_variables

Wlayers
+regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
':%
??2gru/gru_cell/kernel
0:.	5?2gru/gru_cell/recurrent_kernel
$:"	?2gru/gru_cell/bias
.
0
1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
5
20
31
42"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Znon_trainable_variables
?	variables
[metrics
\layer_metrics
]layer_regularization_losses
@trainable_variables

^layers
Aregularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	_total
	`count
a	variables
b	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
?
	ctotal
	dcount
e
_fn_kwargs
f	variables
g	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
.
_0
`1"
trackable_list_wrapper
-
a	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
c0
d1"
trackable_list_wrapper
-
f	variables"
_generic_user_object
2:0?2%RMSprop/batch_normalization/gamma/rms
1:/?2$RMSprop/batch_normalization/beta/rms
(:&52RMSprop/dense/kernel/rms
": 2RMSprop/dense/bias/rms
*:(2RMSprop/dense_1/kernel/rms
$:"2RMSprop/dense_1/bias/rms
*:(2RMSprop/dense_2/kernel/rms
$:"2RMSprop/dense_2/bias/rms
1:/
??2RMSprop/gru/gru_cell/kernel/rms
::8	5?2)RMSprop/gru/gru_cell/recurrent_kernel/rms
.:,	?2RMSprop/gru/gru_cell/bias/rms
?2?
*__inference_sequential_layer_call_fn_21025
*__inference_sequential_layer_call_fn_21946
*__inference_sequential_layer_call_fn_21091
*__inference_sequential_layer_call_fn_21915?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_19181?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *=?:
8?5
batch_normalization_input?????????5?
?2?
E__inference_sequential_layer_call_and_return_conditional_losses_21564
E__inference_sequential_layer_call_and_return_conditional_losses_21884
E__inference_sequential_layer_call_and_return_conditional_losses_20923
E__inference_sequential_layer_call_and_return_conditional_losses_20958?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
3__inference_batch_normalization_layer_call_fn_22097
3__inference_batch_normalization_layer_call_fn_22015
3__inference_batch_normalization_layer_call_fn_22110
3__inference_batch_normalization_layer_call_fn_22028?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22084
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22064
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22002
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21982?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference_gru_layer_call_fn_23467
#__inference_gru_layer_call_fn_22794
#__inference_gru_layer_call_fn_22783
#__inference_gru_layer_call_fn_23478?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
>__inference_gru_layer_call_and_return_conditional_losses_23173
>__inference_gru_layer_call_and_return_conditional_losses_22489
>__inference_gru_layer_call_and_return_conditional_losses_23456
>__inference_gru_layer_call_and_return_conditional_losses_22772?
???
FullArgSpecB
args:?7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults?

 
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_23498?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_23489?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_23518?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_23509?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_23538?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_23529?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_21132batch_normalization_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_gru_cell_layer_call_fn_23818
(__inference_gru_cell_layer_call_fn_23804?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_gru_cell_layer_call_and_return_conditional_losses_23688
C__inference_gru_cell_layer_call_and_return_conditional_losses_23790?
???
FullArgSpec3
args+?(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 ?
 __inference__wrapped_model_19181?423!"'(G?D
=?:
8?5
batch_normalization_input?????????5?
? "1?.
,
dense_2!?
dense_2??????????
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21982~A?>
7?4
.?+
inputs???????????????????
p
? "3?0
)?&
0???????????????????
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22002~A?>
7?4
.?+
inputs???????????????????
p 
? "3?0
)?&
0???????????????????
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22064l8?5
.?+
%?"
inputs?????????5?
p
? "*?'
 ?
0?????????5?
? ?
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22084l8?5
.?+
%?"
inputs?????????5?
p 
? "*?'
 ?
0?????????5?
? ?
3__inference_batch_normalization_layer_call_fn_22015qA?>
7?4
.?+
inputs???????????????????
p
? "&?#????????????????????
3__inference_batch_normalization_layer_call_fn_22028qA?>
7?4
.?+
inputs???????????????????
p 
? "&?#????????????????????
3__inference_batch_normalization_layer_call_fn_22097_8?5
.?+
%?"
inputs?????????5?
p
? "??????????5??
3__inference_batch_normalization_layer_call_fn_22110_8?5
.?+
%?"
inputs?????????5?
p 
? "??????????5??
B__inference_dense_1_layer_call_and_return_conditional_losses_23509\!"/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_1_layer_call_fn_23518O!"/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dense_2_layer_call_and_return_conditional_losses_23529\'(/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_2_layer_call_fn_23538O'(/?,
%?"
 ?
inputs?????????
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_23489\/?,
%?"
 ?
inputs?????????5
? "%?"
?
0?????????
? x
%__inference_dense_layer_call_fn_23498O/?,
%?"
 ?
inputs?????????5
? "???????????
C__inference_gru_cell_layer_call_and_return_conditional_losses_23688?423]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????5
p
? "R?O
H?E
?
0/0?????????5
$?!
?
0/1/0?????????5
? ?
C__inference_gru_cell_layer_call_and_return_conditional_losses_23790?423]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????5
p 
? "R?O
H?E
?
0/0?????????5
$?!
?
0/1/0?????????5
? ?
(__inference_gru_cell_layer_call_fn_23804?423]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????5
p
? "D?A
?
0?????????5
"?
?
1/0?????????5?
(__inference_gru_cell_layer_call_fn_23818?423]?Z
S?P
!?
inputs??????????
'?$
"?
states/0?????????5
p 
? "D?A
?
0?????????5
"?
?
1/0?????????5?
>__inference_gru_layer_call_and_return_conditional_losses_22489n423@?=
6?3
%?"
inputs?????????5?

 
p

 
? "%?"
?
0?????????5
? ?
>__inference_gru_layer_call_and_return_conditional_losses_22772n423@?=
6?3
%?"
inputs?????????5?

 
p 

 
? "%?"
?
0?????????5
? ?
>__inference_gru_layer_call_and_return_conditional_losses_23173~423P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "%?"
?
0?????????5
? ?
>__inference_gru_layer_call_and_return_conditional_losses_23456~423P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "%?"
?
0?????????5
? ?
#__inference_gru_layer_call_fn_22783a423@?=
6?3
%?"
inputs?????????5?

 
p

 
? "??????????5?
#__inference_gru_layer_call_fn_22794a423@?=
6?3
%?"
inputs?????????5?

 
p 

 
? "??????????5?
#__inference_gru_layer_call_fn_23467q423P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p

 
? "??????????5?
#__inference_gru_layer_call_fn_23478q423P?M
F?C
5?2
0?-
inputs/0???????????????????

 
p 

 
? "??????????5?
E__inference_sequential_layer_call_and_return_conditional_losses_20923?423!"'(O?L
E?B
8?5
batch_normalization_input?????????5?
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_20958?423!"'(O?L
E?B
8?5
batch_normalization_input?????????5?
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_21564t423!"'(<?9
2?/
%?"
inputs?????????5?
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_21884t423!"'(<?9
2?/
%?"
inputs?????????5?
p 

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_21025z423!"'(O?L
E?B
8?5
batch_normalization_input?????????5?
p

 
? "???????????
*__inference_sequential_layer_call_fn_21091z423!"'(O?L
E?B
8?5
batch_normalization_input?????????5?
p 

 
? "???????????
*__inference_sequential_layer_call_fn_21915g423!"'(<?9
2?/
%?"
inputs?????????5?
p

 
? "???????????
*__inference_sequential_layer_call_fn_21946g423!"'(<?9
2?/
%?"
inputs?????????5?
p 

 
? "???????????
#__inference_signature_wrapper_21132?423!"'(d?a
? 
Z?W
U
batch_normalization_input8?5
batch_normalization_input?????????5?"1?.
,
dense_2!?
dense_2?????????