       �K"	  ��`:�Abrain.Event:2h��A     j�)�	ў�`:�A"��
�
ConstConst*�
value�B~ Bx/Users/arthur/Documents/natural_lanugage_understanding/nlu_project2/data/processed/train_stories_skip_thoughts.tfrecords*
dtype0*
_output_shapes
: 
g
flat_filenames/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
i
flat_filenamesReshapeConstflat_filenames/shape*
T0*
Tshape0*
_output_shapes
:
v
PlaceholderPlaceholder*
dtype0*+
_output_shapes
:���������* 
shape:���������
h
Placeholder_1Placeholder*
shape:���������*
dtype0*#
_output_shapes
:���������
N
Placeholder_2Placeholder*
dtype0*
_output_shapes
: *
shape: 
�
Const_1Const*�
value�B~ Bx/Users/arthur/Documents/natural_lanugage_understanding/nlu_project2/data/processed/train_stories_skip_thoughts.tfrecords*
dtype0*
_output_shapes
: 
i
flat_filenames_1/shapeConst*
dtype0*
_output_shapes
:*
valueB:
���������
o
flat_filenames_1ReshapeConst_1flat_filenames_1/shape*
T0*
Tshape0*
_output_shapes
:
T
num_parallel_callsConst*
value	B :*
dtype0*
_output_shapes
: 
V
num_parallel_calls_1Const*
value	B :*
dtype0*
_output_shapes
: 
H
countConst*
value
B	 R�'*
dtype0	*
_output_shapes
: 
N
buffer_sizeConst*
value
B	 R�'*
dtype0	*
_output_shapes
: 
F
seedConst*
dtype0	*
_output_shapes
: *
value	B	 R 
G
seed2Const*
dtype0	*
_output_shapes
: *
value	B	 R 
M

batch_sizeConst*
value
B	 R�*
dtype0	*
_output_shapes
: 
P
drop_remainderConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
Const_2Const*�
valueB} Bw/Users/arthur/Documents/natural_lanugage_understanding/nlu_project2/data/processed/eval_stories_skip_thoughts.tfrecords*
dtype0*
_output_shapes
: 
i
flat_filenames_2/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
o
flat_filenames_2ReshapeConst_2flat_filenames_2/shape*
T0*
Tshape0*
_output_shapes
:
V
num_parallel_calls_2Const*
value	B :*
dtype0*
_output_shapes
: 
P
buffer_size_1Const*
value
B	 R�'*
dtype0	*
_output_shapes
: 
H
seed_1Const*
value	B	 R *
dtype0	*
_output_shapes
: 
I
seed2_1Const*
dtype0	*
_output_shapes
: *
value	B	 R 
J
count_1Const*
value
B	 R�*
dtype0	*
_output_shapes
: 
O
batch_size_1Const*
value
B	 R�*
dtype0	*
_output_shapes
: 
R
drop_remainder_1Const*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
optimizationsConst*
dtype0*
_output_shapes
:*V
valueMBKBmap_and_batch_fusionBnoop_eliminationBshuffle_and_repeat_fusion
�

IteratorV2
IteratorV2*
shared_name **
output_shapes
:��%:�*
_output_shapes
: *
	container *
output_types
2
�
TensorSliceDatasetTensorSliceDatasetflat_filenames_1*
output_shapes
: *
Toutput_types
2*
_class
loc:@IteratorV2*
_output_shapes
: 
�
FlatMapDatasetFlatMapDatasetTensorSliceDataset**
f%R#
!Dataset_flat_map_read_one_file_29*
output_types
2*

Targuments
 *
_output_shapes
: *
output_shapes
: *
_class
loc:@IteratorV2
�

MapDataset
MapDatasetFlatMapDataset*
_output_shapes
: *
preserve_cardinality( *6
output_shapes%
#:�%:�%:�%:�%:�%*
_class
loc:@IteratorV2*"
fR
Dataset_map_extract_fn_35*
use_inter_op_parallelism(*
output_types	
2*

Targuments
 
�
ParallelMapDatasetParallelMapDataset
MapDatasetnum_parallel_calls*
_output_shapes
: *
preserve_cardinality( *)
output_shapes
:	�%:	�%*
_class
loc:@IteratorV2*5
f0R.
,Dataset_map_split_skip_thoughts_sentences_48*
sloppy( *
use_inter_op_parallelism(*
output_types
2*

Targuments
 
�
ParallelMapDataset_1ParallelMapDatasetParallelMapDatasetnum_parallel_calls_1* 
output_shapes
:	�%: *
_class
loc:@IteratorV2*3
f.R,
*Dataset_map_<class 'functools.partial'>_64*
sloppy( *
output_types
2*
use_inter_op_parallelism(*

Targuments
 *
_output_shapes
: *
preserve_cardinality( 
�
RepeatDatasetRepeatDatasetParallelMapDataset_1count* 
output_shapes
:	�%: *
_class
loc:@IteratorV2*
_output_shapes
: *
output_types
2
�
ShuffleDatasetShuffleDatasetRepeatDatasetbuffer_sizeseedseed2* 
output_shapes
:	�%: *
_class
loc:@IteratorV2*
reshuffle_each_iteration(*
_output_shapes
: *
output_types
2
�
BatchDatasetV2BatchDatasetV2ShuffleDataset
batch_sizedrop_remainder*
_output_shapes
: *
output_types
2**
output_shapes
:��%:�*
_class
loc:@IteratorV2
�
OptimizeDatasetOptimizeDatasetBatchDatasetV2optimizations**
output_shapes
:��%:�*
_class
loc:@IteratorV2*
_output_shapes
: *
output_types
2
�
ModelDatasetModelDatasetOptimizeDataset*
_output_shapes
: *
output_types
2**
output_shapes
:��%:�*
_class
loc:@IteratorV2
U
MakeIteratorMakeIteratorModelDataset
IteratorV2*
_class
loc:@IteratorV2
T
IteratorToStringHandleIteratorToStringHandle
IteratorV2*
_output_shapes
: 
�
optimizations_1Const*
dtype0*
_output_shapes
:*V
valueMBKBmap_and_batch_fusionBnoop_eliminationBshuffle_and_repeat_fusion
�
IteratorV2_1
IteratorV2**
output_shapes
:��%:�*
_output_shapes
: *
	container *
output_types
2*
shared_name 
�
TensorSliceDataset_1TensorSliceDatasetflat_filenames_2*
output_shapes
: *
Toutput_types
2*
_class
loc:@IteratorV2_1*
_output_shapes
: 
�
FlatMapDataset_1FlatMapDatasetTensorSliceDataset_1*
output_types
2*

Targuments
 *
_output_shapes
: *
output_shapes
: *
_class
loc:@IteratorV2_1*+
f&R$
"Dataset_flat_map_read_one_file_151
�
MapDataset_1
MapDatasetFlatMapDataset_1*

Targuments
 *
_output_shapes
: *
preserve_cardinality( *=
output_shapes,
*:�%:�%:�%:�%:�%:�%*
_class
loc:@IteratorV2_1*#
fR
Dataset_map_extract_fn_157*
use_inter_op_parallelism(*
output_types

2
�
ParallelMapDataset_2ParallelMapDatasetMapDataset_1num_parallel_calls_2*
output_shapes
:	�%*
_class
loc:@IteratorV2_1*'
f"R 
Dataset_map_tensorize_dict_172*
sloppy( *
output_types
2*
use_inter_op_parallelism(*

Targuments
 *
_output_shapes
: *
preserve_cardinality( 
�
TensorSliceDataset_2TensorSliceDatasetPlaceholder_1*
output_shapes
: *
Toutput_types
2*
_class
loc:@IteratorV2_1*
_output_shapes
: 
�

ZipDataset
ZipDatasetParallelMapDataset_2TensorSliceDataset_2* 
output_shapes
:	�%: *
_class
loc:@IteratorV2_1*
N*
_output_shapes
: *
output_types
2
�
ShuffleDataset_1ShuffleDataset
ZipDatasetbuffer_size_1seed_1seed2_1* 
output_shapes
:	�%: *
_class
loc:@IteratorV2_1*
reshuffle_each_iteration(*
_output_shapes
: *
output_types
2
�
RepeatDataset_1RepeatDatasetShuffleDataset_1count_1*
output_types
2* 
output_shapes
:	�%: *
_class
loc:@IteratorV2_1*
_output_shapes
: 
�
BatchDatasetV2_1BatchDatasetV2RepeatDataset_1batch_size_1drop_remainder_1*
_output_shapes
: *
output_types
2**
output_shapes
:��%:�*
_class
loc:@IteratorV2_1
�
OptimizeDataset_1OptimizeDatasetBatchDatasetV2_1optimizations_1**
output_shapes
:��%:�*
_class
loc:@IteratorV2_1*
_output_shapes
: *
output_types
2
�
ModelDataset_1ModelDatasetOptimizeDataset_1*
output_types
2**
output_shapes
:��%:�*
_class
loc:@IteratorV2_1*
_output_shapes
: 
]
MakeIterator_1MakeIteratorModelDataset_1IteratorV2_1*
_class
loc:@IteratorV2_1
X
IteratorToStringHandle_1IteratorToStringHandleIteratorV2_1*
_output_shapes
: 
�
IteratorFromStringHandleV2IteratorFromStringHandleV2Placeholder_2"/device:CPU:0**
output_shapes
:��%:�*
_output_shapes
: *
output_types
2
f
IteratorToStringHandle_2IteratorToStringHandleIteratorFromStringHandleV2*
_output_shapes
: 
�
IteratorGetNextIteratorGetNextIteratorFromStringHandleV2**
output_shapes
:��%:�*+
_output_shapes
:��%:�*
output_types
2
]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
strided_sliceStridedSliceIteratorGetNextstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
end_mask *
_output_shapes
:	�%*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask 
_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_slice_1StridedSliceIteratorGetNext:1strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0
�
TensorSliceDataset_3TensorSliceDatasetflat_filenames_1"/device:CPU:0*
output_shapes
: *
Toutput_types
2*-
_class#
!loc:@IteratorFromStringHandleV2*
_output_shapes
: 
�
FlatMapDataset_2FlatMapDatasetTensorSliceDataset_3"/device:CPU:0*

Targuments
 *
_output_shapes
: *
output_shapes
: *-
_class#
!loc:@IteratorFromStringHandleV2**
f%R#
!Dataset_flat_map_read_one_file_29*
output_types
2
�
MapDataset_2
MapDatasetFlatMapDataset_2"/device:CPU:0*
_output_shapes
: *
preserve_cardinality( *6
output_shapes%
#:�%:�%:�%:�%:�%*-
_class#
!loc:@IteratorFromStringHandleV2*"
fR
Dataset_map_extract_fn_35*
use_inter_op_parallelism(*
output_types	
2*

Targuments
 
�
ParallelMapDataset_3ParallelMapDatasetMapDataset_2num_parallel_calls"/device:CPU:0*
_output_shapes
: *
preserve_cardinality( *)
output_shapes
:	�%:	�%*-
_class#
!loc:@IteratorFromStringHandleV2*5
f0R.
,Dataset_map_split_skip_thoughts_sentences_48*
sloppy( *
use_inter_op_parallelism(*
output_types
2*

Targuments
 
�
ParallelMapDataset_4ParallelMapDatasetParallelMapDataset_3num_parallel_calls_1"/device:CPU:0*
output_types
2*
use_inter_op_parallelism(*

Targuments
 *
_output_shapes
: *
preserve_cardinality( * 
output_shapes
:	�%: *-
_class#
!loc:@IteratorFromStringHandleV2*3
f.R,
*Dataset_map_<class 'functools.partial'>_64*
sloppy( 
�
RepeatDataset_2RepeatDatasetParallelMapDataset_4count"/device:CPU:0*
_output_shapes
: *
output_types
2* 
output_shapes
:	�%: *-
_class#
!loc:@IteratorFromStringHandleV2
�
ShuffleDataset_2ShuffleDatasetRepeatDataset_2buffer_sizeseedseed2"/device:CPU:0* 
output_shapes
:	�%: *-
_class#
!loc:@IteratorFromStringHandleV2*
reshuffle_each_iteration(*
_output_shapes
: *
output_types
2
�
BatchDatasetV2_2BatchDatasetV2ShuffleDataset_2
batch_sizedrop_remainder"/device:CPU:0**
output_shapes
:��%:�*-
_class#
!loc:@IteratorFromStringHandleV2*
_output_shapes
: *
output_types
2
�
train_datasetMakeIteratorBatchDatasetV2_2IteratorFromStringHandleV2"/device:CPU:0*-
_class#
!loc:@IteratorFromStringHandleV2
�
TensorSliceDataset_4TensorSliceDatasetflat_filenames_2"/device:CPU:0*
output_shapes
: *
Toutput_types
2*-
_class#
!loc:@IteratorFromStringHandleV2*
_output_shapes
: 
�
FlatMapDataset_3FlatMapDatasetTensorSliceDataset_4"/device:CPU:0*
output_types
2*

Targuments
 *
_output_shapes
: *
output_shapes
: *-
_class#
!loc:@IteratorFromStringHandleV2*+
f&R$
"Dataset_flat_map_read_one_file_151
�
MapDataset_3
MapDatasetFlatMapDataset_3"/device:CPU:0*=
output_shapes,
*:�%:�%:�%:�%:�%:�%*-
_class#
!loc:@IteratorFromStringHandleV2*#
fR
Dataset_map_extract_fn_157*
output_types

2*
use_inter_op_parallelism(*

Targuments
 *
_output_shapes
: *
preserve_cardinality( 
�
ParallelMapDataset_5ParallelMapDatasetMapDataset_3num_parallel_calls_2"/device:CPU:0*
_output_shapes
: *
preserve_cardinality( *
output_shapes
:	�%*-
_class#
!loc:@IteratorFromStringHandleV2*'
f"R 
Dataset_map_tensorize_dict_172*
sloppy( *
use_inter_op_parallelism(*
output_types
2*

Targuments
 
�
TensorSliceDataset_5TensorSliceDatasetPlaceholder_1"/device:CPU:0*
output_shapes
: *
Toutput_types
2*-
_class#
!loc:@IteratorFromStringHandleV2*
_output_shapes
: 
�
ZipDataset_1
ZipDatasetParallelMapDataset_5TensorSliceDataset_5"/device:CPU:0* 
output_shapes
:	�%: *-
_class#
!loc:@IteratorFromStringHandleV2*
N*
_output_shapes
: *
output_types
2
�
ShuffleDataset_3ShuffleDatasetZipDataset_1buffer_size_1seed_1seed2_1"/device:CPU:0*
reshuffle_each_iteration(*
_output_shapes
: *
output_types
2* 
output_shapes
:	�%: *-
_class#
!loc:@IteratorFromStringHandleV2
�
RepeatDataset_3RepeatDatasetShuffleDataset_3count_1"/device:CPU:0*
_output_shapes
: *
output_types
2* 
output_shapes
:	�%: *-
_class#
!loc:@IteratorFromStringHandleV2
�
BatchDatasetV2_3BatchDatasetV2RepeatDataset_3batch_size_1drop_remainder_1"/device:CPU:0*
output_types
2**
output_shapes
:��%:�*-
_class#
!loc:@IteratorFromStringHandleV2*
_output_shapes
: 
�
test_datasetMakeIteratorBatchDatasetV2_3IteratorFromStringHandleV2"/device:CPU:0*-
_class#
!loc:@IteratorFromStringHandleV2
v
!split_endings/strided_slice/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
x
#split_endings/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*!
valueB"           
x
#split_endings/strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
�
split_endings/strided_sliceStridedSliceIteratorGetNext!split_endings/strided_slice/stack#split_endings/strided_slice/stack_1#split_endings/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*$
_output_shapes
:��%
x
#split_endings/strided_slice_1/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
z
%split_endings/strided_slice_1/stack_1Const*!
valueB"            *
dtype0*
_output_shapes
:
z
%split_endings/strided_slice_1/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
�
split_endings/strided_slice_1StridedSliceIteratorGetNext#split_endings/strided_slice_1/stack%split_endings/strided_slice_1/stack_1%split_endings/strided_slice_1/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*$
_output_shapes
:��%
|
'ending/sentence_rnn/strided_slice/stackConst*
dtype0*
_output_shapes
:*!
valueB"            
~
)ending/sentence_rnn/strided_slice/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
~
)ending/sentence_rnn/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*!
valueB"         
�
!ending/sentence_rnn/strided_sliceStridedSlicesplit_endings/strided_slice_1'ending/sentence_rnn/strided_slice/stack)ending/sentence_rnn/strided_slice/stack_1)ending/sentence_rnn/strided_slice/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*$
_output_shapes
:��%*
T0*
Index0
a
ending/sentence_rnn/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
ending/sentence_rnn/concatConcatV2split_endings/strided_slice!ending/sentence_rnn/strided_sliceending/sentence_rnn/concat/axis*
T0*
N*$
_output_shapes
:��%*

Tidx0
�
ending/sentence_rnn/unstackUnpackending/sentence_rnn/concat*	
num*
T0*

axis*P
_output_shapes>
<:
��%:
��%:
��%:
��%:
��%
z
/ending/sentence_rnn/rnn/LSTMCellZeroState/ConstConst*
valueB:�*
dtype0*
_output_shapes
:
|
1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_1Const*
valueB:�*
dtype0*
_output_shapes
:
w
5ending/sentence_rnn/rnn/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
0ending/sentence_rnn/rnn/LSTMCellZeroState/concatConcatV2/ending/sentence_rnn/rnn/LSTMCellZeroState/Const1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_15ending/sentence_rnn/rnn/LSTMCellZeroState/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
z
5ending/sentence_rnn/rnn/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/ending/sentence_rnn/rnn/LSTMCellZeroState/zerosFill0ending/sentence_rnn/rnn/LSTMCellZeroState/concat5ending/sentence_rnn/rnn/LSTMCellZeroState/zeros/Const*
T0*

index_type0* 
_output_shapes
:
��
|
1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:
|
1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_3Const*
dtype0*
_output_shapes
:*
valueB:�
|
1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_4Const*
valueB:�*
dtype0*
_output_shapes
:
|
1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_5Const*
valueB:�*
dtype0*
_output_shapes
:
y
7ending/sentence_rnn/rnn/LSTMCellZeroState/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
2ending/sentence_rnn/rnn/LSTMCellZeroState/concat_1ConcatV21ending/sentence_rnn/rnn/LSTMCellZeroState/Const_41ending/sentence_rnn/rnn/LSTMCellZeroState/Const_57ending/sentence_rnn/rnn/LSTMCellZeroState/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
|
7ending/sentence_rnn/rnn/LSTMCellZeroState/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
1ending/sentence_rnn/rnn/LSTMCellZeroState/zeros_1Fill2ending/sentence_rnn/rnn/LSTMCellZeroState/concat_17ending/sentence_rnn/rnn/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0* 
_output_shapes
:
��
|
1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_6Const*
dtype0*
_output_shapes
:*
valueB:�
|
1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_7Const*
dtype0*
_output_shapes
:*
valueB:�
�
@ending/rnn/sentence_cell/kernel/Initializer/random_uniform/shapeConst*
valueB"�  �  *2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
dtype0*
_output_shapes
:
�
>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *�ʼ*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel
�
>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *��<*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel
�
Hending/rnn/sentence_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform@ending/rnn/sentence_cell/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
�-�*

seed**
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
seed2x
�
>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/subSub>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/max>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
_output_shapes
: 
�
>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/mulMulHending/rnn/sentence_cell/kernel/Initializer/random_uniform/RandomUniform>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel* 
_output_shapes
:
�-�
�
:ending/rnn/sentence_cell/kernel/Initializer/random_uniformAdd>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/mul>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel* 
_output_shapes
:
�-�
�
ending/rnn/sentence_cell/kernel
VariableV2*
dtype0* 
_output_shapes
:
�-�*
shared_name *2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
	container *
shape:
�-�
�
&ending/rnn/sentence_cell/kernel/AssignAssignending/rnn/sentence_cell/kernel:ending/rnn/sentence_cell/kernel/Initializer/random_uniform*
validate_shape(* 
_output_shapes
:
�-�*
use_locking(*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel
|
$ending/rnn/sentence_cell/kernel/readIdentityending/rnn/sentence_cell/kernel*
T0* 
_output_shapes
:
�-�
�
?ending/rnn/sentence_cell/bias/Initializer/zeros/shape_as_tensorConst*
valueB:�*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
dtype0*
_output_shapes
:
�
5ending/rnn/sentence_cell/bias/Initializer/zeros/ConstConst*
valueB
 *    *0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
dtype0*
_output_shapes
: 
�
/ending/rnn/sentence_cell/bias/Initializer/zerosFill?ending/rnn/sentence_cell/bias/Initializer/zeros/shape_as_tensor5ending/rnn/sentence_cell/bias/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
_output_shapes	
:�
�
ending/rnn/sentence_cell/bias
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
	container 
�
$ending/rnn/sentence_cell/bias/AssignAssignending/rnn/sentence_cell/bias/ending/rnn/sentence_cell/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias
s
"ending/rnn/sentence_cell/bias/readIdentityending/rnn/sentence_cell/bias*
T0*
_output_shapes	
:�
s
1ending/sentence_rnn/rnn/sentence_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
,ending/sentence_rnn/rnn/sentence_cell/concatConcatV2ending/sentence_rnn/unstack1ending/sentence_rnn/rnn/LSTMCellZeroState/zeros_11ending/sentence_rnn/rnn/sentence_cell/concat/axis*
T0*
N* 
_output_shapes
:
��-*

Tidx0
�
,ending/sentence_rnn/rnn/sentence_cell/MatMulMatMul,ending/sentence_rnn/rnn/sentence_cell/concat$ending/rnn/sentence_cell/kernel/read* 
_output_shapes
:
��*
transpose_a( *
transpose_b( *
T0
�
-ending/sentence_rnn/rnn/sentence_cell/BiasAddBiasAdd,ending/sentence_rnn/rnn/sentence_cell/MatMul"ending/rnn/sentence_cell/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
��
m
+ending/sentence_rnn/rnn/sentence_cell/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
w
5ending/sentence_rnn/rnn/sentence_cell/split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
�
+ending/sentence_rnn/rnn/sentence_cell/splitSplit5ending/sentence_rnn/rnn/sentence_cell/split/split_dim-ending/sentence_rnn/rnn/sentence_cell/BiasAdd*
T0*D
_output_shapes2
0:
��:
��:
��:
��*
	num_split
p
+ending/sentence_rnn/rnn/sentence_cell/add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
)ending/sentence_rnn/rnn/sentence_cell/addAdd-ending/sentence_rnn/rnn/sentence_cell/split:2+ending/sentence_rnn/rnn/sentence_cell/add/y*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn/sentence_cell/SigmoidSigmoid)ending/sentence_rnn/rnn/sentence_cell/add*
T0* 
_output_shapes
:
��
�
)ending/sentence_rnn/rnn/sentence_cell/mulMul-ending/sentence_rnn/rnn/sentence_cell/Sigmoid/ending/sentence_rnn/rnn/LSTMCellZeroState/zeros*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_1Sigmoid+ending/sentence_rnn/rnn/sentence_cell/split*
T0* 
_output_shapes
:
��
�
*ending/sentence_rnn/rnn/sentence_cell/TanhTanh-ending/sentence_rnn/rnn/sentence_cell/split:1*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_1Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_1*ending/sentence_rnn/rnn/sentence_cell/Tanh*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/add_1Add)ending/sentence_rnn/rnn/sentence_cell/mul+ending/sentence_rnn/rnn/sentence_cell/mul_1*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_2Sigmoid-ending/sentence_rnn/rnn/sentence_cell/split:3*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_1Tanh+ending/sentence_rnn/rnn/sentence_cell/add_1*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_2Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_2,ending/sentence_rnn/rnn/sentence_cell/Tanh_1*
T0* 
_output_shapes
:
��
u
3ending/sentence_rnn/rnn/sentence_cell/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
.ending/sentence_rnn/rnn/sentence_cell/concat_1ConcatV2ending/sentence_rnn/unstack:1+ending/sentence_rnn/rnn/sentence_cell/mul_23ending/sentence_rnn/rnn/sentence_cell/concat_1/axis*
T0*
N* 
_output_shapes
:
��-*

Tidx0
�
.ending/sentence_rnn/rnn/sentence_cell/MatMul_1MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_1$ending/rnn/sentence_cell/kernel/read*
T0* 
_output_shapes
:
��*
transpose_a( *
transpose_b( 
�
/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1BiasAdd.ending/sentence_rnn/rnn/sentence_cell/MatMul_1"ending/rnn/sentence_cell/bias/read*
data_formatNHWC* 
_output_shapes
:
��*
T0
o
-ending/sentence_rnn/rnn/sentence_cell/Const_1Const*
dtype0*
_output_shapes
: *
value	B :
y
7ending/sentence_rnn/rnn/sentence_cell/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn/sentence_cell/split_1Split7ending/sentence_rnn/rnn/sentence_cell/split_1/split_dim/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1*
T0*D
_output_shapes2
0:
��:
��:
��:
��*
	num_split
r
-ending/sentence_rnn/rnn/sentence_cell/add_2/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+ending/sentence_rnn/rnn/sentence_cell/add_2Add/ending/sentence_rnn/rnn/sentence_cell/split_1:2-ending/sentence_rnn/rnn/sentence_cell/add_2/y*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_3Sigmoid+ending/sentence_rnn/rnn/sentence_cell/add_2*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_3Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_3+ending/sentence_rnn/rnn/sentence_cell/add_1* 
_output_shapes
:
��*
T0
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_4Sigmoid-ending/sentence_rnn/rnn/sentence_cell/split_1*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_2Tanh/ending/sentence_rnn/rnn/sentence_cell/split_1:1* 
_output_shapes
:
��*
T0
�
+ending/sentence_rnn/rnn/sentence_cell/mul_4Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_4,ending/sentence_rnn/rnn/sentence_cell/Tanh_2*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/add_3Add+ending/sentence_rnn/rnn/sentence_cell/mul_3+ending/sentence_rnn/rnn/sentence_cell/mul_4*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_5Sigmoid/ending/sentence_rnn/rnn/sentence_cell/split_1:3*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_3Tanh+ending/sentence_rnn/rnn/sentence_cell/add_3*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_5Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_5,ending/sentence_rnn/rnn/sentence_cell/Tanh_3*
T0* 
_output_shapes
:
��
u
3ending/sentence_rnn/rnn/sentence_cell/concat_2/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
.ending/sentence_rnn/rnn/sentence_cell/concat_2ConcatV2ending/sentence_rnn/unstack:2+ending/sentence_rnn/rnn/sentence_cell/mul_53ending/sentence_rnn/rnn/sentence_cell/concat_2/axis*
T0*
N* 
_output_shapes
:
��-*

Tidx0
�
.ending/sentence_rnn/rnn/sentence_cell/MatMul_2MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_2$ending/rnn/sentence_cell/kernel/read* 
_output_shapes
:
��*
transpose_a( *
transpose_b( *
T0
�
/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2BiasAdd.ending/sentence_rnn/rnn/sentence_cell/MatMul_2"ending/rnn/sentence_cell/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
��
o
-ending/sentence_rnn/rnn/sentence_cell/Const_2Const*
value	B :*
dtype0*
_output_shapes
: 
y
7ending/sentence_rnn/rnn/sentence_cell/split_2/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn/sentence_cell/split_2Split7ending/sentence_rnn/rnn/sentence_cell/split_2/split_dim/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2*
T0*D
_output_shapes2
0:
��:
��:
��:
��*
	num_split
r
-ending/sentence_rnn/rnn/sentence_cell/add_4/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+ending/sentence_rnn/rnn/sentence_cell/add_4Add/ending/sentence_rnn/rnn/sentence_cell/split_2:2-ending/sentence_rnn/rnn/sentence_cell/add_4/y*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_6Sigmoid+ending/sentence_rnn/rnn/sentence_cell/add_4*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_6Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_6+ending/sentence_rnn/rnn/sentence_cell/add_3* 
_output_shapes
:
��*
T0
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_7Sigmoid-ending/sentence_rnn/rnn/sentence_cell/split_2* 
_output_shapes
:
��*
T0
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_4Tanh/ending/sentence_rnn/rnn/sentence_cell/split_2:1*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_7Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_7,ending/sentence_rnn/rnn/sentence_cell/Tanh_4*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/add_5Add+ending/sentence_rnn/rnn/sentence_cell/mul_6+ending/sentence_rnn/rnn/sentence_cell/mul_7*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_8Sigmoid/ending/sentence_rnn/rnn/sentence_cell/split_2:3*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_5Tanh+ending/sentence_rnn/rnn/sentence_cell/add_5* 
_output_shapes
:
��*
T0
�
+ending/sentence_rnn/rnn/sentence_cell/mul_8Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_8,ending/sentence_rnn/rnn/sentence_cell/Tanh_5*
T0* 
_output_shapes
:
��
u
3ending/sentence_rnn/rnn/sentence_cell/concat_3/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
.ending/sentence_rnn/rnn/sentence_cell/concat_3ConcatV2ending/sentence_rnn/unstack:3+ending/sentence_rnn/rnn/sentence_cell/mul_83ending/sentence_rnn/rnn/sentence_cell/concat_3/axis*
N* 
_output_shapes
:
��-*

Tidx0*
T0
�
.ending/sentence_rnn/rnn/sentence_cell/MatMul_3MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_3$ending/rnn/sentence_cell/kernel/read*
T0* 
_output_shapes
:
��*
transpose_a( *
transpose_b( 
�
/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3BiasAdd.ending/sentence_rnn/rnn/sentence_cell/MatMul_3"ending/rnn/sentence_cell/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
��
o
-ending/sentence_rnn/rnn/sentence_cell/Const_3Const*
value	B :*
dtype0*
_output_shapes
: 
y
7ending/sentence_rnn/rnn/sentence_cell/split_3/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn/sentence_cell/split_3Split7ending/sentence_rnn/rnn/sentence_cell/split_3/split_dim/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3*
T0*D
_output_shapes2
0:
��:
��:
��:
��*
	num_split
r
-ending/sentence_rnn/rnn/sentence_cell/add_6/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+ending/sentence_rnn/rnn/sentence_cell/add_6Add/ending/sentence_rnn/rnn/sentence_cell/split_3:2-ending/sentence_rnn/rnn/sentence_cell/add_6/y*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_9Sigmoid+ending/sentence_rnn/rnn/sentence_cell/add_6*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_9Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_9+ending/sentence_rnn/rnn/sentence_cell/add_5*
T0* 
_output_shapes
:
��
�
0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_10Sigmoid-ending/sentence_rnn/rnn/sentence_cell/split_3* 
_output_shapes
:
��*
T0
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_6Tanh/ending/sentence_rnn/rnn/sentence_cell/split_3:1*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/mul_10Mul0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_10,ending/sentence_rnn/rnn/sentence_cell/Tanh_6*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/add_7Add+ending/sentence_rnn/rnn/sentence_cell/mul_9,ending/sentence_rnn/rnn/sentence_cell/mul_10*
T0* 
_output_shapes
:
��
�
0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_11Sigmoid/ending/sentence_rnn/rnn/sentence_cell/split_3:3*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_7Tanh+ending/sentence_rnn/rnn/sentence_cell/add_7*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/mul_11Mul0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_11,ending/sentence_rnn/rnn/sentence_cell/Tanh_7*
T0* 
_output_shapes
:
��
u
3ending/sentence_rnn/rnn/sentence_cell/concat_4/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
.ending/sentence_rnn/rnn/sentence_cell/concat_4ConcatV2ending/sentence_rnn/unstack:4,ending/sentence_rnn/rnn/sentence_cell/mul_113ending/sentence_rnn/rnn/sentence_cell/concat_4/axis*

Tidx0*
T0*
N* 
_output_shapes
:
��-
�
.ending/sentence_rnn/rnn/sentence_cell/MatMul_4MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_4$ending/rnn/sentence_cell/kernel/read* 
_output_shapes
:
��*
transpose_a( *
transpose_b( *
T0
�
/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4BiasAdd.ending/sentence_rnn/rnn/sentence_cell/MatMul_4"ending/rnn/sentence_cell/bias/read*
data_formatNHWC* 
_output_shapes
:
��*
T0
o
-ending/sentence_rnn/rnn/sentence_cell/Const_4Const*
value	B :*
dtype0*
_output_shapes
: 
y
7ending/sentence_rnn/rnn/sentence_cell/split_4/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn/sentence_cell/split_4Split7ending/sentence_rnn/rnn/sentence_cell/split_4/split_dim/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4*
T0*D
_output_shapes2
0:
��:
��:
��:
��*
	num_split
r
-ending/sentence_rnn/rnn/sentence_cell/add_8/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+ending/sentence_rnn/rnn/sentence_cell/add_8Add/ending/sentence_rnn/rnn/sentence_cell/split_4:2-ending/sentence_rnn/rnn/sentence_cell/add_8/y*
T0* 
_output_shapes
:
��
�
0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_12Sigmoid+ending/sentence_rnn/rnn/sentence_cell/add_8*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/mul_12Mul0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_12+ending/sentence_rnn/rnn/sentence_cell/add_7* 
_output_shapes
:
��*
T0
�
0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_13Sigmoid-ending/sentence_rnn/rnn/sentence_cell/split_4*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_8Tanh/ending/sentence_rnn/rnn/sentence_cell/split_4:1*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/mul_13Mul0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_13,ending/sentence_rnn/rnn/sentence_cell/Tanh_8*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/add_9Add,ending/sentence_rnn/rnn/sentence_cell/mul_12,ending/sentence_rnn/rnn/sentence_cell/mul_13* 
_output_shapes
:
��*
T0
�
0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_14Sigmoid/ending/sentence_rnn/rnn/sentence_cell/split_4:3* 
_output_shapes
:
��*
T0
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_9Tanh+ending/sentence_rnn/rnn/sentence_cell/add_9* 
_output_shapes
:
��*
T0
�
,ending/sentence_rnn/rnn/sentence_cell/mul_14Mul0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_14,ending/sentence_rnn/rnn/sentence_cell/Tanh_9*
T0* 
_output_shapes
:
��
d
"ending/sentence_rnn/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
ending/sentence_rnn/ExpandDims
ExpandDims+ending/sentence_rnn/rnn/sentence_cell/mul_2"ending/sentence_rnn/ExpandDims/dim*

Tdim0*
T0*$
_output_shapes
:��
f
$ending/sentence_rnn/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
 ending/sentence_rnn/ExpandDims_1
ExpandDims+ending/sentence_rnn/rnn/sentence_cell/mul_5$ending/sentence_rnn/ExpandDims_1/dim*

Tdim0*
T0*$
_output_shapes
:��
f
$ending/sentence_rnn/ExpandDims_2/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
 ending/sentence_rnn/ExpandDims_2
ExpandDims+ending/sentence_rnn/rnn/sentence_cell/mul_8$ending/sentence_rnn/ExpandDims_2/dim*
T0*$
_output_shapes
:��*

Tdim0
f
$ending/sentence_rnn/ExpandDims_3/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
 ending/sentence_rnn/ExpandDims_3
ExpandDims,ending/sentence_rnn/rnn/sentence_cell/mul_11$ending/sentence_rnn/ExpandDims_3/dim*

Tdim0*
T0*$
_output_shapes
:��
f
$ending/sentence_rnn/ExpandDims_4/dimConst*
dtype0*
_output_shapes
: *
value	B :
�
 ending/sentence_rnn/ExpandDims_4
ExpandDims,ending/sentence_rnn/rnn/sentence_cell/mul_14$ending/sentence_rnn/ExpandDims_4/dim*$
_output_shapes
:��*

Tdim0*
T0
c
!ending/sentence_rnn/concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
ending/sentence_rnn/concat_1ConcatV2ending/sentence_rnn/ExpandDims ending/sentence_rnn/ExpandDims_1 ending/sentence_rnn/ExpandDims_2 ending/sentence_rnn/ExpandDims_3 ending/sentence_rnn/ExpandDims_4!ending/sentence_rnn/concat_1/axis*

Tidx0*
T0*
N*$
_output_shapes
:��
i
'ending/sentence_rnn/concat_2/concat_dimConst*
dtype0*
_output_shapes
: *
value	B :
�
ending/sentence_rnn/concat_2Identity+ending/sentence_rnn/rnn/sentence_cell/add_9*
T0* 
_output_shapes
:
��
e
 ending/sentence_rnn/dropout/rateConst*
dtype0*
_output_shapes
: *
valueB
 *   ?
r
!ending/sentence_rnn/dropout/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
f
!ending/sentence_rnn/dropout/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
ending/sentence_rnn/dropout/subSub!ending/sentence_rnn/dropout/sub/x ending/sentence_rnn/dropout/rate*
T0*
_output_shapes
: 
s
.ending/sentence_rnn/dropout/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
s
.ending/sentence_rnn/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
8ending/sentence_rnn/dropout/random_uniform/RandomUniformRandomUniform!ending/sentence_rnn/dropout/Shape*
T0*
dtype0* 
_output_shapes
:
��*
seed2�*

seed*
�
.ending/sentence_rnn/dropout/random_uniform/subSub.ending/sentence_rnn/dropout/random_uniform/max.ending/sentence_rnn/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
.ending/sentence_rnn/dropout/random_uniform/mulMul8ending/sentence_rnn/dropout/random_uniform/RandomUniform.ending/sentence_rnn/dropout/random_uniform/sub*
T0* 
_output_shapes
:
��
�
*ending/sentence_rnn/dropout/random_uniformAdd.ending/sentence_rnn/dropout/random_uniform/mul.ending/sentence_rnn/dropout/random_uniform/min* 
_output_shapes
:
��*
T0
�
ending/sentence_rnn/dropout/addAddending/sentence_rnn/dropout/sub*ending/sentence_rnn/dropout/random_uniform*
T0* 
_output_shapes
:
��
v
!ending/sentence_rnn/dropout/FloorFloorending/sentence_rnn/dropout/add*
T0* 
_output_shapes
:
��
�
#ending/sentence_rnn/dropout/truedivRealDivending/sentence_rnn/concat_2ending/sentence_rnn/dropout/sub*
T0* 
_output_shapes
:
��
�
ending/sentence_rnn/dropout/mulMul#ending/sentence_rnn/dropout/truediv!ending/sentence_rnn/dropout/Floor* 
_output_shapes
:
��*
T0
�
5ending/output/kernel/Initializer/random_uniform/shapeConst*
valueB"�     *'
_class
loc:@ending/output/kernel*
dtype0*
_output_shapes
:
�
3ending/output/kernel/Initializer/random_uniform/minConst*
valueB
 *⎞�*'
_class
loc:@ending/output/kernel*
dtype0*
_output_shapes
: 
�
3ending/output/kernel/Initializer/random_uniform/maxConst*
valueB
 *⎞=*'
_class
loc:@ending/output/kernel*
dtype0*
_output_shapes
: 
�
=ending/output/kernel/Initializer/random_uniform/RandomUniformRandomUniform5ending/output/kernel/Initializer/random_uniform/shape*
T0*'
_class
loc:@ending/output/kernel*
seed2�*
dtype0*
_output_shapes
:	�*

seed*
�
3ending/output/kernel/Initializer/random_uniform/subSub3ending/output/kernel/Initializer/random_uniform/max3ending/output/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*'
_class
loc:@ending/output/kernel
�
3ending/output/kernel/Initializer/random_uniform/mulMul=ending/output/kernel/Initializer/random_uniform/RandomUniform3ending/output/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@ending/output/kernel*
_output_shapes
:	�
�
/ending/output/kernel/Initializer/random_uniformAdd3ending/output/kernel/Initializer/random_uniform/mul3ending/output/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@ending/output/kernel*
_output_shapes
:	�
�
ending/output/kernel
VariableV2*'
_class
loc:@ending/output/kernel*
	container *
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name 
�
ending/output/kernel/AssignAssignending/output/kernel/ending/output/kernel/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@ending/output/kernel*
validate_shape(*
_output_shapes
:	�
�
ending/output/kernel/readIdentityending/output/kernel*
T0*'
_class
loc:@ending/output/kernel*
_output_shapes
:	�
�
$ending/output/bias/Initializer/zerosConst*
valueB*    *%
_class
loc:@ending/output/bias*
dtype0*
_output_shapes
:
�
ending/output/bias
VariableV2*
shared_name *%
_class
loc:@ending/output/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
ending/output/bias/AssignAssignending/output/bias$ending/output/bias/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@ending/output/bias*
validate_shape(*
_output_shapes
:
�
ending/output/bias/readIdentityending/output/bias*
T0*%
_class
loc:@ending/output/bias*
_output_shapes
:
�
$ending/sentence_rnn/fc/output/MatMulMatMulending/sentence_rnn/dropout/mulending/output/kernel/read*
transpose_b( *
T0*
_output_shapes
:	�*
transpose_a( 
�
%ending/sentence_rnn/fc/output/BiasAddBiasAdd$ending/sentence_rnn/fc/output/MatMulending/output/bias/read*
T0*
data_formatNHWC*
_output_shapes
:	�
~
)ending/sentence_rnn/strided_slice_1/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
�
+ending/sentence_rnn/strided_slice_1/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
�
+ending/sentence_rnn/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*!
valueB"         
�
#ending/sentence_rnn/strided_slice_1StridedSlicesplit_endings/strided_slice_1)ending/sentence_rnn/strided_slice_1/stack+ending/sentence_rnn/strided_slice_1/stack_1+ending/sentence_rnn/strided_slice_1/stack_2*
T0*
Index0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*$
_output_shapes
:��%
c
!ending/sentence_rnn/concat_3/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
ending/sentence_rnn/concat_3ConcatV2split_endings/strided_slice#ending/sentence_rnn/strided_slice_1!ending/sentence_rnn/concat_3/axis*
T0*
N*$
_output_shapes
:��%*

Tidx0
�
ending/sentence_rnn/unstack_1Unpackending/sentence_rnn/concat_3*	
num*
T0*

axis*P
_output_shapes>
<:
��%:
��%:
��%:
��%:
��%
|
1ending/sentence_rnn/rnn_1/LSTMCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:�
~
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_1Const*
valueB:�*
dtype0*
_output_shapes
:
y
7ending/sentence_rnn/rnn_1/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
2ending/sentence_rnn/rnn_1/LSTMCellZeroState/concatConcatV21ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_17ending/sentence_rnn/rnn_1/LSTMCellZeroState/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
|
7ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1ending/sentence_rnn/rnn_1/LSTMCellZeroState/zerosFill2ending/sentence_rnn/rnn_1/LSTMCellZeroState/concat7ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros/Const*
T0*

index_type0* 
_output_shapes
:
��
~
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_2Const*
dtype0*
_output_shapes
:*
valueB:�
~
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_3Const*
valueB:�*
dtype0*
_output_shapes
:
~
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_4Const*
valueB:�*
dtype0*
_output_shapes
:
~
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_5Const*
valueB:�*
dtype0*
_output_shapes
:
{
9ending/sentence_rnn/rnn_1/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
4ending/sentence_rnn/rnn_1/LSTMCellZeroState/concat_1ConcatV23ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_43ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_59ending/sentence_rnn/rnn_1/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
~
9ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros_1Fill4ending/sentence_rnn/rnn_1/LSTMCellZeroState/concat_19ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0* 
_output_shapes
:
��
~
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_6Const*
dtype0*
_output_shapes
:*
valueB:�
~
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_7Const*
dtype0*
_output_shapes
:*
valueB:�
u
3ending/sentence_rnn/rnn_1/sentence_cell/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
.ending/sentence_rnn/rnn_1/sentence_cell/concatConcatV2ending/sentence_rnn/unstack_13ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros_13ending/sentence_rnn/rnn_1/sentence_cell/concat/axis*
N* 
_output_shapes
:
��-*

Tidx0*
T0
�
.ending/sentence_rnn/rnn_1/sentence_cell/MatMulMatMul.ending/sentence_rnn/rnn_1/sentence_cell/concat$ending/rnn/sentence_cell/kernel/read*
T0* 
_output_shapes
:
��*
transpose_a( *
transpose_b( 
�
/ending/sentence_rnn/rnn_1/sentence_cell/BiasAddBiasAdd.ending/sentence_rnn/rnn_1/sentence_cell/MatMul"ending/rnn/sentence_cell/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
��
o
-ending/sentence_rnn/rnn_1/sentence_cell/ConstConst*
dtype0*
_output_shapes
: *
value	B :
y
7ending/sentence_rnn/rnn_1/sentence_cell/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn_1/sentence_cell/splitSplit7ending/sentence_rnn/rnn_1/sentence_cell/split/split_dim/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd*
T0*D
_output_shapes2
0:
��:
��:
��:
��*
	num_split
r
-ending/sentence_rnn/rnn_1/sentence_cell/add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+ending/sentence_rnn/rnn_1/sentence_cell/addAdd/ending/sentence_rnn/rnn_1/sentence_cell/split:2-ending/sentence_rnn/rnn_1/sentence_cell/add/y*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn_1/sentence_cell/SigmoidSigmoid+ending/sentence_rnn/rnn_1/sentence_cell/add*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn_1/sentence_cell/mulMul/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid1ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros* 
_output_shapes
:
��*
T0
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_1Sigmoid-ending/sentence_rnn/rnn_1/sentence_cell/split*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn_1/sentence_cell/TanhTanh/ending/sentence_rnn/rnn_1/sentence_cell/split:1*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_1Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_1,ending/sentence_rnn/rnn_1/sentence_cell/Tanh*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_1Add+ending/sentence_rnn/rnn_1/sentence_cell/mul-ending/sentence_rnn/rnn_1/sentence_cell/mul_1*
T0* 
_output_shapes
:
��
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_2Sigmoid/ending/sentence_rnn/rnn_1/sentence_cell/split:3*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_1Tanh-ending/sentence_rnn/rnn_1/sentence_cell/add_1*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_2Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_2.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_1*
T0* 
_output_shapes
:
��
w
5ending/sentence_rnn/rnn_1/sentence_cell/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
0ending/sentence_rnn/rnn_1/sentence_cell/concat_1ConcatV2ending/sentence_rnn/unstack_1:1-ending/sentence_rnn/rnn_1/sentence_cell/mul_25ending/sentence_rnn/rnn_1/sentence_cell/concat_1/axis*
N* 
_output_shapes
:
��-*

Tidx0*
T0
�
0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_1$ending/rnn/sentence_cell/kernel/read* 
_output_shapes
:
��*
transpose_a( *
transpose_b( *
T0
�
1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1BiasAdd0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1"ending/rnn/sentence_cell/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
��
q
/ending/sentence_rnn/rnn_1/sentence_cell/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
{
9ending/sentence_rnn/rnn_1/sentence_cell/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
/ending/sentence_rnn/rnn_1/sentence_cell/split_1Split9ending/sentence_rnn/rnn_1/sentence_cell/split_1/split_dim1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1*
T0*D
_output_shapes2
0:
��:
��:
��:
��*
	num_split
t
/ending/sentence_rnn/rnn_1/sentence_cell/add_2/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_2Add1ending/sentence_rnn/rnn_1/sentence_cell/split_1:2/ending/sentence_rnn/rnn_1/sentence_cell/add_2/y*
T0* 
_output_shapes
:
��
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_3Sigmoid-ending/sentence_rnn/rnn_1/sentence_cell/add_2*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_3Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_3-ending/sentence_rnn/rnn_1/sentence_cell/add_1*
T0* 
_output_shapes
:
��
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_4Sigmoid/ending/sentence_rnn/rnn_1/sentence_cell/split_1*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_2Tanh1ending/sentence_rnn/rnn_1/sentence_cell/split_1:1* 
_output_shapes
:
��*
T0
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_4Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_4.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_2*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_3Add-ending/sentence_rnn/rnn_1/sentence_cell/mul_3-ending/sentence_rnn/rnn_1/sentence_cell/mul_4* 
_output_shapes
:
��*
T0
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_5Sigmoid1ending/sentence_rnn/rnn_1/sentence_cell/split_1:3*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_3Tanh-ending/sentence_rnn/rnn_1/sentence_cell/add_3* 
_output_shapes
:
��*
T0
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_5Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_5.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_3*
T0* 
_output_shapes
:
��
w
5ending/sentence_rnn/rnn_1/sentence_cell/concat_2/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
0ending/sentence_rnn/rnn_1/sentence_cell/concat_2ConcatV2ending/sentence_rnn/unstack_1:2-ending/sentence_rnn/rnn_1/sentence_cell/mul_55ending/sentence_rnn/rnn_1/sentence_cell/concat_2/axis*
T0*
N* 
_output_shapes
:
��-*

Tidx0
�
0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_2$ending/rnn/sentence_cell/kernel/read* 
_output_shapes
:
��*
transpose_a( *
transpose_b( *
T0
�
1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2BiasAdd0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2"ending/rnn/sentence_cell/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
��
q
/ending/sentence_rnn/rnn_1/sentence_cell/Const_2Const*
value	B :*
dtype0*
_output_shapes
: 
{
9ending/sentence_rnn/rnn_1/sentence_cell/split_2/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
/ending/sentence_rnn/rnn_1/sentence_cell/split_2Split9ending/sentence_rnn/rnn_1/sentence_cell/split_2/split_dim1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2*
T0*D
_output_shapes2
0:
��:
��:
��:
��*
	num_split
t
/ending/sentence_rnn/rnn_1/sentence_cell/add_4/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_4Add1ending/sentence_rnn/rnn_1/sentence_cell/split_2:2/ending/sentence_rnn/rnn_1/sentence_cell/add_4/y*
T0* 
_output_shapes
:
��
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_6Sigmoid-ending/sentence_rnn/rnn_1/sentence_cell/add_4*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_6Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_6-ending/sentence_rnn/rnn_1/sentence_cell/add_3*
T0* 
_output_shapes
:
��
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_7Sigmoid/ending/sentence_rnn/rnn_1/sentence_cell/split_2* 
_output_shapes
:
��*
T0
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_4Tanh1ending/sentence_rnn/rnn_1/sentence_cell/split_2:1*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_7Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_7.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_4*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_5Add-ending/sentence_rnn/rnn_1/sentence_cell/mul_6-ending/sentence_rnn/rnn_1/sentence_cell/mul_7*
T0* 
_output_shapes
:
��
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_8Sigmoid1ending/sentence_rnn/rnn_1/sentence_cell/split_2:3* 
_output_shapes
:
��*
T0
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_5Tanh-ending/sentence_rnn/rnn_1/sentence_cell/add_5* 
_output_shapes
:
��*
T0
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_8Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_8.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_5*
T0* 
_output_shapes
:
��
w
5ending/sentence_rnn/rnn_1/sentence_cell/concat_3/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
0ending/sentence_rnn/rnn_1/sentence_cell/concat_3ConcatV2ending/sentence_rnn/unstack_1:3-ending/sentence_rnn/rnn_1/sentence_cell/mul_85ending/sentence_rnn/rnn_1/sentence_cell/concat_3/axis*
T0*
N* 
_output_shapes
:
��-*

Tidx0
�
0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_3$ending/rnn/sentence_cell/kernel/read* 
_output_shapes
:
��*
transpose_a( *
transpose_b( *
T0
�
1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3BiasAdd0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3"ending/rnn/sentence_cell/bias/read*
data_formatNHWC* 
_output_shapes
:
��*
T0
q
/ending/sentence_rnn/rnn_1/sentence_cell/Const_3Const*
value	B :*
dtype0*
_output_shapes
: 
{
9ending/sentence_rnn/rnn_1/sentence_cell/split_3/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
/ending/sentence_rnn/rnn_1/sentence_cell/split_3Split9ending/sentence_rnn/rnn_1/sentence_cell/split_3/split_dim1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3*D
_output_shapes2
0:
��:
��:
��:
��*
	num_split*
T0
t
/ending/sentence_rnn/rnn_1/sentence_cell/add_6/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_6Add1ending/sentence_rnn/rnn_1/sentence_cell/split_3:2/ending/sentence_rnn/rnn_1/sentence_cell/add_6/y* 
_output_shapes
:
��*
T0
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_9Sigmoid-ending/sentence_rnn/rnn_1/sentence_cell/add_6*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_9Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_9-ending/sentence_rnn/rnn_1/sentence_cell/add_5*
T0* 
_output_shapes
:
��
�
2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_10Sigmoid/ending/sentence_rnn/rnn_1/sentence_cell/split_3* 
_output_shapes
:
��*
T0
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_6Tanh1ending/sentence_rnn/rnn_1/sentence_cell/split_3:1* 
_output_shapes
:
��*
T0
�
.ending/sentence_rnn/rnn_1/sentence_cell/mul_10Mul2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_10.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_6*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_7Add-ending/sentence_rnn/rnn_1/sentence_cell/mul_9.ending/sentence_rnn/rnn_1/sentence_cell/mul_10* 
_output_shapes
:
��*
T0
�
2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_11Sigmoid1ending/sentence_rnn/rnn_1/sentence_cell/split_3:3*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_7Tanh-ending/sentence_rnn/rnn_1/sentence_cell/add_7*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/mul_11Mul2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_11.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_7* 
_output_shapes
:
��*
T0
w
5ending/sentence_rnn/rnn_1/sentence_cell/concat_4/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
0ending/sentence_rnn/rnn_1/sentence_cell/concat_4ConcatV2ending/sentence_rnn/unstack_1:4.ending/sentence_rnn/rnn_1/sentence_cell/mul_115ending/sentence_rnn/rnn_1/sentence_cell/concat_4/axis*
T0*
N* 
_output_shapes
:
��-*

Tidx0
�
0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_4$ending/rnn/sentence_cell/kernel/read*
T0* 
_output_shapes
:
��*
transpose_a( *
transpose_b( 
�
1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4BiasAdd0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4"ending/rnn/sentence_cell/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
��
q
/ending/sentence_rnn/rnn_1/sentence_cell/Const_4Const*
value	B :*
dtype0*
_output_shapes
: 
{
9ending/sentence_rnn/rnn_1/sentence_cell/split_4/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
/ending/sentence_rnn/rnn_1/sentence_cell/split_4Split9ending/sentence_rnn/rnn_1/sentence_cell/split_4/split_dim1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4*D
_output_shapes2
0:
��:
��:
��:
��*
	num_split*
T0
t
/ending/sentence_rnn/rnn_1/sentence_cell/add_8/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_8Add1ending/sentence_rnn/rnn_1/sentence_cell/split_4:2/ending/sentence_rnn/rnn_1/sentence_cell/add_8/y*
T0* 
_output_shapes
:
��
�
2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_12Sigmoid-ending/sentence_rnn/rnn_1/sentence_cell/add_8*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/mul_12Mul2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_12-ending/sentence_rnn/rnn_1/sentence_cell/add_7*
T0* 
_output_shapes
:
��
�
2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_13Sigmoid/ending/sentence_rnn/rnn_1/sentence_cell/split_4* 
_output_shapes
:
��*
T0
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_8Tanh1ending/sentence_rnn/rnn_1/sentence_cell/split_4:1*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/mul_13Mul2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_13.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_8*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_9Add.ending/sentence_rnn/rnn_1/sentence_cell/mul_12.ending/sentence_rnn/rnn_1/sentence_cell/mul_13*
T0* 
_output_shapes
:
��
�
2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_14Sigmoid1ending/sentence_rnn/rnn_1/sentence_cell/split_4:3*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_9Tanh-ending/sentence_rnn/rnn_1/sentence_cell/add_9*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/mul_14Mul2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_14.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_9*
T0* 
_output_shapes
:
��
f
$ending/sentence_rnn/ExpandDims_5/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
 ending/sentence_rnn/ExpandDims_5
ExpandDims-ending/sentence_rnn/rnn_1/sentence_cell/mul_2$ending/sentence_rnn/ExpandDims_5/dim*

Tdim0*
T0*$
_output_shapes
:��
f
$ending/sentence_rnn/ExpandDims_6/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
 ending/sentence_rnn/ExpandDims_6
ExpandDims-ending/sentence_rnn/rnn_1/sentence_cell/mul_5$ending/sentence_rnn/ExpandDims_6/dim*

Tdim0*
T0*$
_output_shapes
:��
f
$ending/sentence_rnn/ExpandDims_7/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
 ending/sentence_rnn/ExpandDims_7
ExpandDims-ending/sentence_rnn/rnn_1/sentence_cell/mul_8$ending/sentence_rnn/ExpandDims_7/dim*

Tdim0*
T0*$
_output_shapes
:��
f
$ending/sentence_rnn/ExpandDims_8/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
 ending/sentence_rnn/ExpandDims_8
ExpandDims.ending/sentence_rnn/rnn_1/sentence_cell/mul_11$ending/sentence_rnn/ExpandDims_8/dim*$
_output_shapes
:��*

Tdim0*
T0
f
$ending/sentence_rnn/ExpandDims_9/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
 ending/sentence_rnn/ExpandDims_9
ExpandDims.ending/sentence_rnn/rnn_1/sentence_cell/mul_14$ending/sentence_rnn/ExpandDims_9/dim*$
_output_shapes
:��*

Tdim0*
T0
c
!ending/sentence_rnn/concat_4/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
ending/sentence_rnn/concat_4ConcatV2 ending/sentence_rnn/ExpandDims_5 ending/sentence_rnn/ExpandDims_6 ending/sentence_rnn/ExpandDims_7 ending/sentence_rnn/ExpandDims_8 ending/sentence_rnn/ExpandDims_9!ending/sentence_rnn/concat_4/axis*

Tidx0*
T0*
N*$
_output_shapes
:��
i
'ending/sentence_rnn/concat_5/concat_dimConst*
dtype0*
_output_shapes
: *
value	B :
�
ending/sentence_rnn/concat_5Identity-ending/sentence_rnn/rnn_1/sentence_cell/add_9*
T0* 
_output_shapes
:
��
g
"ending/sentence_rnn/dropout_1/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
t
#ending/sentence_rnn/dropout_1/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
h
#ending/sentence_rnn/dropout_1/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
!ending/sentence_rnn/dropout_1/subSub#ending/sentence_rnn/dropout_1/sub/x"ending/sentence_rnn/dropout_1/rate*
T0*
_output_shapes
: 
u
0ending/sentence_rnn/dropout_1/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *    
u
0ending/sentence_rnn/dropout_1/random_uniform/maxConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
:ending/sentence_rnn/dropout_1/random_uniform/RandomUniformRandomUniform#ending/sentence_rnn/dropout_1/Shape*
T0*
dtype0* 
_output_shapes
:
��*
seed2�*

seed*
�
0ending/sentence_rnn/dropout_1/random_uniform/subSub0ending/sentence_rnn/dropout_1/random_uniform/max0ending/sentence_rnn/dropout_1/random_uniform/min*
_output_shapes
: *
T0
�
0ending/sentence_rnn/dropout_1/random_uniform/mulMul:ending/sentence_rnn/dropout_1/random_uniform/RandomUniform0ending/sentence_rnn/dropout_1/random_uniform/sub*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/dropout_1/random_uniformAdd0ending/sentence_rnn/dropout_1/random_uniform/mul0ending/sentence_rnn/dropout_1/random_uniform/min* 
_output_shapes
:
��*
T0
�
!ending/sentence_rnn/dropout_1/addAdd!ending/sentence_rnn/dropout_1/sub,ending/sentence_rnn/dropout_1/random_uniform*
T0* 
_output_shapes
:
��
z
#ending/sentence_rnn/dropout_1/FloorFloor!ending/sentence_rnn/dropout_1/add*
T0* 
_output_shapes
:
��
�
%ending/sentence_rnn/dropout_1/truedivRealDivending/sentence_rnn/concat_5!ending/sentence_rnn/dropout_1/sub*
T0* 
_output_shapes
:
��
�
!ending/sentence_rnn/dropout_1/mulMul%ending/sentence_rnn/dropout_1/truediv#ending/sentence_rnn/dropout_1/Floor*
T0* 
_output_shapes
:
��
�
&ending/sentence_rnn/fc_1/output/MatMulMatMul!ending/sentence_rnn/dropout_1/mulending/output/kernel/read*
_output_shapes
:	�*
transpose_a( *
transpose_b( *
T0
�
'ending/sentence_rnn/fc_1/output/BiasAddBiasAdd&ending/sentence_rnn/fc_1/output/MatMulending/output/bias/read*
T0*
data_formatNHWC*
_output_shapes
:	�
�
ending/sentence_rnn/stackPack%ending/sentence_rnn/fc/output/BiasAdd'ending/sentence_rnn/fc_1/output/BiasAdd*
T0*

axis*
N*#
_output_shapes
:�

eval_predictions/SqueezeSqueezeending/sentence_rnn/stack*
_output_shapes
:	�*
squeeze_dims
*
T0
c
!eval_predictions/ArgMax/dimensionConst*
value	B :*
dtype0*
_output_shapes
: 
�
eval_predictions/ArgMaxArgMaxeval_predictions/Squeeze!eval_predictions/ArgMax/dimension*
output_type0	*
_output_shapes	
:�*

Tidx0*
T0
~
eval_predictions/ToInt32Casteval_predictions/ArgMax*
Truncate( *
_output_shapes	
:�*

DstT0*

SrcT0	
v
%train_predictions/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
x
'train_predictions/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
x
'train_predictions/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
�
train_predictions/strided_sliceStridedSliceending/sentence_rnn/stack%train_predictions/strided_slice/stack'train_predictions/strided_slice/stack_1'train_predictions/strided_slice/stack_2*
shrink_axis_mask*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes
:	�*
Index0*
T0
�
train_predictions/SqueezeSqueezetrain_predictions/strided_slice*
_output_shapes	
:�*
squeeze_dims
*
T0
e
train_predictions/SigmoidSigmoidtrain_predictions/Squeeze*
T0*
_output_shapes	
:�
a
train_predictions/RoundRoundtrain_predictions/Sigmoid*
T0*
_output_shapes	
:�

train_predictions/ToInt32Casttrain_predictions/Round*
Truncate( *
_output_shapes	
:�*

DstT0*

SrcT0
t
)SparseSoftmaxCrossEntropyWithLogits/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
GSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitseval_predictions/SqueezeIteratorGetNext:1*
T0*&
_output_shapes
:�:	�*
Tlabels0
Q
Const_3Const*
valueB: *
dtype0*
_output_shapes
:
�
MeanMeanGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsConst_3*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
a
EqualEqualeval_predictions/ToInt32IteratorGetNext:1*
T0*
_output_shapes	
:�
X
CastCastEqual*

SrcT0
*
Truncate( *
_output_shapes	
:�*

DstT0
Q
Const_4Const*
dtype0*
_output_shapes
:*
valueB: 
[
Mean_1MeanCastConst_4*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
[
global_step/initial_valueConst*
dtype0*
_output_shapes
: *
value	B : 
o
global_step
VariableV2*
dtype0*
_output_shapes
: *
	container *
shape: *
shared_name 
�
global_step/AssignAssignglobal_stepglobal_step/initial_value*
use_locking(*
T0*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0*
_class
loc:@global_step*
_output_shapes
: 
R
gradients/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
X
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
d
gradients/Mean_grad/ConstConst*
valueB:�*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Const*
T0*
_output_shapes	
:�*

Tmultiples0
`
gradients/Mean_grad/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *   C
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Const_1*
T0*
_output_shapes	
:�
�
gradients/zeros_like	ZerosLikeISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes
:	�
�
fgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes
:	�*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()
�
egradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
agradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsgradients/Mean_grad/truedivegradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*

Tdim0*
T0*
_output_shapes
:	�
�
Zgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulagradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsfgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*
_output_shapes
:	�
�
-gradients/eval_predictions/Squeeze_grad/ShapeConst*!
valueB"�         *
dtype0*
_output_shapes
:
�
/gradients/eval_predictions/Squeeze_grad/ReshapeReshapeZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul-gradients/eval_predictions/Squeeze_grad/Shape*
T0*
Tshape0*#
_output_shapes
:�
�
0gradients/ending/sentence_rnn/stack_grad/unstackUnpack/gradients/eval_predictions/Squeeze_grad/Reshape*	
num*
T0*

axis**
_output_shapes
:	�:	�
t
9gradients/ending/sentence_rnn/stack_grad/tuple/group_depsNoOp1^gradients/ending/sentence_rnn/stack_grad/unstack
�
Agradients/ending/sentence_rnn/stack_grad/tuple/control_dependencyIdentity0gradients/ending/sentence_rnn/stack_grad/unstack:^gradients/ending/sentence_rnn/stack_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/ending/sentence_rnn/stack_grad/unstack*
_output_shapes
:	�
�
Cgradients/ending/sentence_rnn/stack_grad/tuple/control_dependency_1Identity2gradients/ending/sentence_rnn/stack_grad/unstack:1:^gradients/ending/sentence_rnn/stack_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/ending/sentence_rnn/stack_grad/unstack*
_output_shapes
:	�
�
@gradients/ending/sentence_rnn/fc/output/BiasAdd_grad/BiasAddGradBiasAddGradAgradients/ending/sentence_rnn/stack_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes
:*
T0
�
Egradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/fc/output/BiasAdd_grad/BiasAddGradB^gradients/ending/sentence_rnn/stack_grad/tuple/control_dependency
�
Mgradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/control_dependencyIdentityAgradients/ending/sentence_rnn/stack_grad/tuple/control_dependencyF^gradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/ending/sentence_rnn/stack_grad/unstack*
_output_shapes
:	�
�
Ogradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/fc/output/BiasAdd_grad/BiasAddGradF^gradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/fc/output/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
Bgradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/BiasAddGradBiasAddGradCgradients/ending/sentence_rnn/stack_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0
�
Ggradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/group_depsNoOpC^gradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/BiasAddGradD^gradients/ending/sentence_rnn/stack_grad/tuple/control_dependency_1
�
Ogradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/stack_grad/tuple/control_dependency_1H^gradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/ending/sentence_rnn/stack_grad/unstack*
_output_shapes
:	�
�
Qgradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/BiasAddGradH^gradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
:gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMulMatMulMgradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/control_dependencyending/output/kernel/read* 
_output_shapes
:
��*
transpose_a( *
transpose_b(*
T0
�
<gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMul_1MatMulending/sentence_rnn/dropout/mulMgradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes
:	�*
transpose_a(
�
Dgradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/group_depsNoOp;^gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMul=^gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMul_1
�
Lgradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/control_dependencyIdentity:gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMulE^gradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMul* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/control_dependency_1Identity<gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMul_1E^gradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
<gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMulMatMulOgradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/control_dependencyending/output/kernel/read*
T0* 
_output_shapes
:
��*
transpose_a( *
transpose_b(
�
>gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMul_1MatMul!ending/sentence_rnn/dropout_1/mulOgradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes
:	�*
transpose_a(*
transpose_b( 
�
Fgradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/group_depsNoOp=^gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMul?^gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMul_1
�
Ngradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/control_dependencyIdentity<gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMulG^gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMul* 
_output_shapes
:
��
�
Pgradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/control_dependency_1Identity>gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMul_1G^gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
gradients/AddNAddNOgradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/control_dependency_1Qgradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/fc/output/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
�
2gradients/ending/sentence_rnn/dropout/mul_grad/MulMulLgradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/control_dependency!ending/sentence_rnn/dropout/Floor*
T0* 
_output_shapes
:
��
�
4gradients/ending/sentence_rnn/dropout/mul_grad/Mul_1MulLgradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/control_dependency#ending/sentence_rnn/dropout/truediv* 
_output_shapes
:
��*
T0
�
?gradients/ending/sentence_rnn/dropout/mul_grad/tuple/group_depsNoOp3^gradients/ending/sentence_rnn/dropout/mul_grad/Mul5^gradients/ending/sentence_rnn/dropout/mul_grad/Mul_1
�
Ggradients/ending/sentence_rnn/dropout/mul_grad/tuple/control_dependencyIdentity2gradients/ending/sentence_rnn/dropout/mul_grad/Mul@^gradients/ending/sentence_rnn/dropout/mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/ending/sentence_rnn/dropout/mul_grad/Mul* 
_output_shapes
:
��
�
Igradients/ending/sentence_rnn/dropout/mul_grad/tuple/control_dependency_1Identity4gradients/ending/sentence_rnn/dropout/mul_grad/Mul_1@^gradients/ending/sentence_rnn/dropout/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/ending/sentence_rnn/dropout/mul_grad/Mul_1* 
_output_shapes
:
��
�
4gradients/ending/sentence_rnn/dropout_1/mul_grad/MulMulNgradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/control_dependency#ending/sentence_rnn/dropout_1/Floor*
T0* 
_output_shapes
:
��
�
6gradients/ending/sentence_rnn/dropout_1/mul_grad/Mul_1MulNgradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/control_dependency%ending/sentence_rnn/dropout_1/truediv*
T0* 
_output_shapes
:
��
�
Agradients/ending/sentence_rnn/dropout_1/mul_grad/tuple/group_depsNoOp5^gradients/ending/sentence_rnn/dropout_1/mul_grad/Mul7^gradients/ending/sentence_rnn/dropout_1/mul_grad/Mul_1
�
Igradients/ending/sentence_rnn/dropout_1/mul_grad/tuple/control_dependencyIdentity4gradients/ending/sentence_rnn/dropout_1/mul_grad/MulB^gradients/ending/sentence_rnn/dropout_1/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/ending/sentence_rnn/dropout_1/mul_grad/Mul* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/dropout_1/mul_grad/tuple/control_dependency_1Identity6gradients/ending/sentence_rnn/dropout_1/mul_grad/Mul_1B^gradients/ending/sentence_rnn/dropout_1/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/ending/sentence_rnn/dropout_1/mul_grad/Mul_1* 
_output_shapes
:
��
�
gradients/AddN_1AddNNgradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/control_dependency_1Pgradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/control_dependency_1*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMul_1*
N*
_output_shapes
:	�
�
8gradients/ending/sentence_rnn/dropout/truediv_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
}
:gradients/ending/sentence_rnn/dropout/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Hgradients/ending/sentence_rnn/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/ending/sentence_rnn/dropout/truediv_grad/Shape:gradients/ending/sentence_rnn/dropout/truediv_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
:gradients/ending/sentence_rnn/dropout/truediv_grad/RealDivRealDivGgradients/ending/sentence_rnn/dropout/mul_grad/tuple/control_dependencyending/sentence_rnn/dropout/sub*
T0* 
_output_shapes
:
��
�
6gradients/ending/sentence_rnn/dropout/truediv_grad/SumSum:gradients/ending/sentence_rnn/dropout/truediv_grad/RealDivHgradients/ending/sentence_rnn/dropout/truediv_grad/BroadcastGradientArgs* 
_output_shapes
:
��*
	keep_dims( *

Tidx0*
T0
�
:gradients/ending/sentence_rnn/dropout/truediv_grad/ReshapeReshape6gradients/ending/sentence_rnn/dropout/truediv_grad/Sum8gradients/ending/sentence_rnn/dropout/truediv_grad/Shape* 
_output_shapes
:
��*
T0*
Tshape0
�
6gradients/ending/sentence_rnn/dropout/truediv_grad/NegNegending/sentence_rnn/concat_2*
T0* 
_output_shapes
:
��
�
<gradients/ending/sentence_rnn/dropout/truediv_grad/RealDiv_1RealDiv6gradients/ending/sentence_rnn/dropout/truediv_grad/Negending/sentence_rnn/dropout/sub*
T0* 
_output_shapes
:
��
�
<gradients/ending/sentence_rnn/dropout/truediv_grad/RealDiv_2RealDiv<gradients/ending/sentence_rnn/dropout/truediv_grad/RealDiv_1ending/sentence_rnn/dropout/sub*
T0* 
_output_shapes
:
��
�
6gradients/ending/sentence_rnn/dropout/truediv_grad/mulMulGgradients/ending/sentence_rnn/dropout/mul_grad/tuple/control_dependency<gradients/ending/sentence_rnn/dropout/truediv_grad/RealDiv_2*
T0* 
_output_shapes
:
��
�
8gradients/ending/sentence_rnn/dropout/truediv_grad/Sum_1Sum6gradients/ending/sentence_rnn/dropout/truediv_grad/mulJgradients/ending/sentence_rnn/dropout/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
<gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape_1Reshape8gradients/ending/sentence_rnn/dropout/truediv_grad/Sum_1:gradients/ending/sentence_rnn/dropout/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/dropout/truediv_grad/tuple/group_depsNoOp;^gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape=^gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape_1
�
Kgradients/ending/sentence_rnn/dropout/truediv_grad/tuple/control_dependencyIdentity:gradients/ending/sentence_rnn/dropout/truediv_grad/ReshapeD^gradients/ending/sentence_rnn/dropout/truediv_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*M
_classC
A?loc:@gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape
�
Mgradients/ending/sentence_rnn/dropout/truediv_grad/tuple/control_dependency_1Identity<gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape_1D^gradients/ending/sentence_rnn/dropout/truediv_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape_1*
_output_shapes
: 
�
:gradients/ending/sentence_rnn/dropout_1/truediv_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:

<gradients/ending/sentence_rnn/dropout_1/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Jgradients/ending/sentence_rnn/dropout_1/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/ending/sentence_rnn/dropout_1/truediv_grad/Shape<gradients/ending/sentence_rnn/dropout_1/truediv_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<gradients/ending/sentence_rnn/dropout_1/truediv_grad/RealDivRealDivIgradients/ending/sentence_rnn/dropout_1/mul_grad/tuple/control_dependency!ending/sentence_rnn/dropout_1/sub*
T0* 
_output_shapes
:
��
�
8gradients/ending/sentence_rnn/dropout_1/truediv_grad/SumSum<gradients/ending/sentence_rnn/dropout_1/truediv_grad/RealDivJgradients/ending/sentence_rnn/dropout_1/truediv_grad/BroadcastGradientArgs*
T0* 
_output_shapes
:
��*
	keep_dims( *

Tidx0
�
<gradients/ending/sentence_rnn/dropout_1/truediv_grad/ReshapeReshape8gradients/ending/sentence_rnn/dropout_1/truediv_grad/Sum:gradients/ending/sentence_rnn/dropout_1/truediv_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
8gradients/ending/sentence_rnn/dropout_1/truediv_grad/NegNegending/sentence_rnn/concat_5*
T0* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/dropout_1/truediv_grad/RealDiv_1RealDiv8gradients/ending/sentence_rnn/dropout_1/truediv_grad/Neg!ending/sentence_rnn/dropout_1/sub* 
_output_shapes
:
��*
T0
�
>gradients/ending/sentence_rnn/dropout_1/truediv_grad/RealDiv_2RealDiv>gradients/ending/sentence_rnn/dropout_1/truediv_grad/RealDiv_1!ending/sentence_rnn/dropout_1/sub* 
_output_shapes
:
��*
T0
�
8gradients/ending/sentence_rnn/dropout_1/truediv_grad/mulMulIgradients/ending/sentence_rnn/dropout_1/mul_grad/tuple/control_dependency>gradients/ending/sentence_rnn/dropout_1/truediv_grad/RealDiv_2*
T0* 
_output_shapes
:
��
�
:gradients/ending/sentence_rnn/dropout_1/truediv_grad/Sum_1Sum8gradients/ending/sentence_rnn/dropout_1/truediv_grad/mulLgradients/ending/sentence_rnn/dropout_1/truediv_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
>gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape_1Reshape:gradients/ending/sentence_rnn/dropout_1/truediv_grad/Sum_1<gradients/ending/sentence_rnn/dropout_1/truediv_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
Egradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/group_depsNoOp=^gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape?^gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape_1
�
Mgradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/control_dependencyIdentity<gradients/ending/sentence_rnn/dropout_1/truediv_grad/ReshapeF^gradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape* 
_output_shapes
:
��
�
Ogradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/control_dependency_1Identity>gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape_1F^gradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape_1*
_output_shapes
: 
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/group_depsNoOpL^gradients/ending/sentence_rnn/dropout/truediv_grad/tuple/control_dependency
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/control_dependencyIdentityKgradients/ending/sentence_rnn/dropout/truediv_grad/tuple/control_dependencyL^gradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/control_dependency_1IdentityKgradients/ending/sentence_rnn/dropout/truediv_grad/tuple/control_dependencyL^gradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/group_depsNoOpN^gradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/control_dependency
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/control_dependencyIdentityMgradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/control_dependencyN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/control_dependency_1IdentityMgradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/control_dependencyN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape* 
_output_shapes
:
��
�
?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/MulMulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/control_dependency+ending/sentence_rnn/rnn/sentence_cell/add_7*
T0* 
_output_shapes
:
��
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul_1MulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/control_dependency0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_12* 
_output_shapes
:
��*
T0
�
Lgradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/tuple/group_depsNoOp@^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/MulB^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul_1
�
Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/tuple/control_dependencyIdentity?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/MulM^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul* 
_output_shapes
:
��
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/tuple/control_dependency_1IdentityAgradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul_1M^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul_1* 
_output_shapes
:
��
�
?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/MulMulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_8*
T0* 
_output_shapes
:
��
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/control_dependency_10ending/sentence_rnn/rnn/sentence_cell/Sigmoid_13* 
_output_shapes
:
��*
T0
�
Lgradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/tuple/group_depsNoOp@^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/MulB^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/Mul_1
�
Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/tuple/control_dependencyIdentity?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/MulM^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/Mul* 
_output_shapes
:
��
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/tuple/control_dependency_1IdentityAgradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/Mul_1M^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/Mul_1* 
_output_shapes
:
��
�
Agradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/MulMulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/control_dependency-ending/sentence_rnn/rnn_1/sentence_cell/add_7*
T0* 
_output_shapes
:
��
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/control_dependency2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_12*
T0* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/tuple/group_depsNoOpB^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/MulD^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul_1
�
Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/tuple/control_dependencyIdentityAgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/MulO^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul* 
_output_shapes
:
��
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/tuple/control_dependency_1IdentityCgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul_1O^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul_1* 
_output_shapes
:
��
�
Agradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/MulMulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_8*
T0* 
_output_shapes
:
��
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/Mul_1MulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/control_dependency_12ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_13*
T0* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/tuple/group_depsNoOpB^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/MulD^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/Mul_1
�
Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/tuple/control_dependencyIdentityAgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/MulO^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/Mul* 
_output_shapes
:
��
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/tuple/control_dependency_1IdentityCgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/Mul_1O^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/Mul_1* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_12_grad/SigmoidGradSigmoidGrad0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_12Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_13_grad/SigmoidGradSigmoidGrad0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_13Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_8_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_8Vgradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_12_grad/SigmoidGradSigmoidGrad2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_12Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_13_grad/SigmoidGradSigmoidGrad2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_13Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_8_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_8Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Pgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/ShapeBgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/SumSumKgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_12_grad/SigmoidGradPgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/ReshapeReshape>gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Sum@gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Sum_1SumKgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_12_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Reshape_1Reshape@gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Sum_1Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/tuple/group_depsNoOpC^gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/ReshapeE^gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Reshape_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/tuple/control_dependencyIdentityBgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/ReshapeL^gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Reshape
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/tuple/control_dependency_1IdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Reshape_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/tuple/group_deps*
_output_shapes
: *
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Reshape_1
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Rgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/ShapeDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/SumSumMgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_12_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/BroadcastGradientArgs*
T0* 
_output_shapes
:
��*
	keep_dims( *

Tidx0
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/ReshapeReshape@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/SumBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Sum_1SumMgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_12_grad/SigmoidGradTgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/BroadcastGradientArgs:1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Reshape_1ReshapeBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Sum_1Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/ReshapeG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Reshape_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/ReshapeN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Reshape
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Reshape_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Reshape_1*
_output_shapes
: 

gradients/zeros_like_1	ZerosLike/ending/sentence_rnn/rnn/sentence_cell/split_4:3*
T0* 
_output_shapes
:
��
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/split_4_grad/concatConcatV2Kgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_13_grad/SigmoidGradDgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_8_grad/TanhGradSgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/tuple/control_dependencygradients/zeros_like_17ending/sentence_rnn/rnn/sentence_cell/split_4/split_dim*
T0*
N* 
_output_shapes
:
��*

Tidx0
�
gradients/zeros_like_2	ZerosLike1ending/sentence_rnn/rnn_1/sentence_cell/split_4:3*
T0* 
_output_shapes
:
��
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/split_4_grad/concatConcatV2Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_13_grad/SigmoidGradFgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_8_grad/TanhGradUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/tuple/control_dependencygradients/zeros_like_29ending/sentence_rnn/rnn_1/sentence_cell/split_4/split_dim*
T0*
N* 
_output_shapes
:
��*

Tidx0
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/BiasAddGradBiasAddGradCgradients/ending/sentence_rnn/rnn/sentence_cell/split_4_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ogradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/group_depsNoOpK^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/BiasAddGradD^gradients/ending/sentence_rnn/rnn/sentence_cell/split_4_grad/concat
�
Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/split_4_grad/concatP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/split_4_grad/concat* 
_output_shapes
:
��
�
Ygradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/control_dependency_1IdentityJgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/BiasAddGradP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/BiasAddGrad*
_output_shapes	
:�
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/BiasAddGradBiasAddGradEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_4_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Qgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/group_depsNoOpM^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/BiasAddGradF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_4_grad/concat
�
Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_4_grad/concatR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_4_grad/concat* 
_output_shapes
:
��
�
[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/control_dependency_1IdentityLgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/BiasAddGradR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/BiasAddGrad*
_output_shapes	
:�
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMulMatMulWgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
transpose_b(*
T0* 
_output_shapes
:
��-*
transpose_a( 
�
Fgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMul_1MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_4Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
�-�*
transpose_a(
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMulG^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMul_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMulO^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/group_deps* 
_output_shapes
:
��-*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMul
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMul_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMul_1* 
_output_shapes
:
�-�
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMulMatMulYgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
T0* 
_output_shapes
:
��-*
transpose_a( *
transpose_b(
�
Hgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMul_1MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_4Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
�-�*
transpose_a(
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/group_depsNoOpG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMulI^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMul_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/control_dependencyIdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMulQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMul* 
_output_shapes
:
��-
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/control_dependency_1IdentityHgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMul_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMul_1* 
_output_shapes
:
�-�
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/modFloorMod3ending/sentence_rnn/rnn/sentence_cell/concat_4/axisBgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Rank*
T0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/ConcatOffsetConcatOffsetAgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/modCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/ShapeEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Shape_1*
N* 
_output_shapes
::
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/SliceSliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/control_dependencyJgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/ConcatOffsetCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Shape*
Index0*
T0* 
_output_shapes
:
��%
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Slice_1SliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/ConcatOffset:1Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Shape_1*
Index0*
T0* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/tuple/group_depsNoOpD^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/SliceF^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Slice_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/SliceO^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Slice* 
_output_shapes
:
��%
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/tuple/control_dependency_1IdentityEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Slice_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Slice_1* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/modFloorMod5ending/sentence_rnn/rnn_1/sentence_cell/concat_4/axisDgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Rank*
T0*
_output_shapes
: 
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/ConcatOffsetConcatOffsetCgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/modEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/ShapeGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Shape_1*
N* 
_output_shapes
::
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/SliceSliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/ConcatOffsetEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Shape*
Index0*
T0* 
_output_shapes
:
��%
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Slice_1SliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/control_dependencyNgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/ConcatOffset:1Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Shape_1*
Index0*
T0* 
_output_shapes
:
��
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/tuple/group_depsNoOpF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/SliceH^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Slice_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/SliceQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Slice* 
_output_shapes
:
��%
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/tuple/control_dependency_1IdentityGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Slice_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Slice_1* 
_output_shapes
:
��
�
?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/MulMulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_7*
T0* 
_output_shapes
:
��
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/Mul_1MulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/tuple/control_dependency_10ending/sentence_rnn/rnn/sentence_cell/Sigmoid_11*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/tuple/group_depsNoOp@^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/MulB^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/Mul_1
�
Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/tuple/control_dependencyIdentity?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/MulM^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/Mul* 
_output_shapes
:
��
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/tuple/control_dependency_1IdentityAgradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/Mul_1M^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/Mul_1
�
Agradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/MulMulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_7*
T0* 
_output_shapes
:
��
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/Mul_1MulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/tuple/control_dependency_12ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_11*
T0* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/tuple/group_depsNoOpB^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/MulD^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/Mul_1
�
Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/tuple/control_dependencyIdentityAgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/MulO^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/Mul* 
_output_shapes
:
��
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/tuple/control_dependency_1IdentityCgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/Mul_1O^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/Mul_1* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_11_grad/SigmoidGradSigmoidGrad0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_11Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_7_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_7Vgradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_11_grad/SigmoidGradSigmoidGrad2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_11Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_7_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_7Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
gradients/AddN_2AddNVgradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/tuple/control_dependency_1Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_7_grad/TanhGrad*
N* 
_output_shapes
:
��*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul_1
f
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/group_depsNoOp^gradients/AddN_2
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/control_dependencyIdentitygradients/AddN_2L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul_1* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/control_dependency_1Identitygradients/AddN_2L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul_1* 
_output_shapes
:
��
�
gradients/AddN_3AddNXgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/tuple/control_dependency_1Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_7_grad/TanhGrad*
N* 
_output_shapes
:
��*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul_1
h
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/group_depsNoOp^gradients/AddN_3
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/control_dependencyIdentitygradients/AddN_3N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul_1* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/control_dependency_1Identitygradients/AddN_3N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/MulMulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/control_dependency+ending/sentence_rnn/rnn/sentence_cell/add_5*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul_1MulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/control_dependency/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_9*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul_1
�
?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/MulMulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_6* 
_output_shapes
:
��*
T0
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/control_dependency_10ending/sentence_rnn/rnn/sentence_cell/Sigmoid_10*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/tuple/group_depsNoOp@^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/MulB^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/Mul_1
�
Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/tuple/control_dependencyIdentity?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/MulM^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/Mul* 
_output_shapes
:
��
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/tuple/control_dependency_1IdentityAgradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/Mul_1M^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/Mul_1* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/MulMulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/control_dependency-ending/sentence_rnn/rnn_1/sentence_cell/add_5*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/control_dependency1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_9*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul_1* 
_output_shapes
:
��
�
Agradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/MulMulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_6*
T0* 
_output_shapes
:
��
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/Mul_1MulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/control_dependency_12ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_10*
T0* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/tuple/group_depsNoOpB^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/MulD^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/Mul_1
�
Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/tuple/control_dependencyIdentityAgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/MulO^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/Mul* 
_output_shapes
:
��
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/tuple/control_dependency_1IdentityCgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/Mul_1O^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/Mul_1
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_9_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_9Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_10_grad/SigmoidGradSigmoidGrad0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_10Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_6_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_6Vgradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
T0
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_9_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_9Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_10_grad/SigmoidGradSigmoidGrad2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_10Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_6_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_6Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Pgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/ShapeBgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/SumSumJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_9_grad/SigmoidGradPgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/BroadcastGradientArgs*
T0* 
_output_shapes
:
��*
	keep_dims( *

Tidx0
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/ReshapeReshape>gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Sum@gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Sum_1SumJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_9_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Reshape_1Reshape@gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Sum_1Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/tuple/group_depsNoOpC^gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/ReshapeE^gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Reshape_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/tuple/control_dependencyIdentityBgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/ReshapeL^gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Reshape* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/tuple/control_dependency_1IdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Reshape_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Reshape_1*
_output_shapes
: 
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Rgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/ShapeDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/SumSumLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_9_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/BroadcastGradientArgs* 
_output_shapes
:
��*
	keep_dims( *

Tidx0*
T0
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/ReshapeReshape@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/SumBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Sum_1SumLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_9_grad/SigmoidGradTgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Reshape_1ReshapeBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Sum_1Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/ReshapeG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Reshape_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/ReshapeN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Reshape* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Reshape_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Reshape_1*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/split_3_grad/concatConcatV2Kgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_10_grad/SigmoidGradDgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_6_grad/TanhGradSgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/tuple/control_dependencyKgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_11_grad/SigmoidGrad7ending/sentence_rnn/rnn/sentence_cell/split_3/split_dim*
T0*
N* 
_output_shapes
:
��*

Tidx0
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/split_3_grad/concatConcatV2Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_10_grad/SigmoidGradFgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_6_grad/TanhGradUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/tuple/control_dependencyMgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_11_grad/SigmoidGrad9ending/sentence_rnn/rnn_1/sentence_cell/split_3/split_dim*
N* 
_output_shapes
:
��*

Tidx0*
T0
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/BiasAddGradBiasAddGradCgradients/ending/sentence_rnn/rnn/sentence_cell/split_3_grad/concat*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Ogradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/group_depsNoOpK^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/BiasAddGradD^gradients/ending/sentence_rnn/rnn/sentence_cell/split_3_grad/concat
�
Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/split_3_grad/concatP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/split_3_grad/concat* 
_output_shapes
:
��
�
Ygradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/control_dependency_1IdentityJgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/BiasAddGradP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/BiasAddGrad*
_output_shapes	
:�
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/BiasAddGradBiasAddGradEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_3_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Qgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/group_depsNoOpM^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/BiasAddGradF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_3_grad/concat
�
Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_3_grad/concatR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_3_grad/concat* 
_output_shapes
:
��
�
[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/control_dependency_1IdentityLgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/BiasAddGradR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/BiasAddGrad*
_output_shapes	
:�
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMulMatMulWgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read* 
_output_shapes
:
��-*
transpose_a( *
transpose_b(*
T0
�
Fgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMul_1MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_3Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/control_dependency*
T0* 
_output_shapes
:
�-�*
transpose_a(*
transpose_b( 
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMulG^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMul_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMulO^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMul* 
_output_shapes
:
��-
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMul_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/group_deps* 
_output_shapes
:
�-�*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMul_1
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMulMatMulYgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read* 
_output_shapes
:
��-*
transpose_a( *
transpose_b(*
T0
�
Hgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMul_1MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_3Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
�-�*
transpose_a(
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/group_depsNoOpG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMulI^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMul_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/control_dependencyIdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMulQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMul* 
_output_shapes
:
��-
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/control_dependency_1IdentityHgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMul_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMul_1* 
_output_shapes
:
�-�
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/modFloorMod3ending/sentence_rnn/rnn/sentence_cell/concat_3/axisBgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Rank*
T0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Shape_1Const*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/ConcatOffsetConcatOffsetAgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/modCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/ShapeEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Shape_1*
N* 
_output_shapes
::
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/SliceSliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/control_dependencyJgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/ConcatOffsetCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Shape*
Index0*
T0* 
_output_shapes
:
��%
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Slice_1SliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/ConcatOffset:1Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Shape_1*
Index0*
T0* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/tuple/group_depsNoOpD^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/SliceF^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Slice_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/SliceO^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/tuple/group_deps* 
_output_shapes
:
��%*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Slice
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/tuple/control_dependency_1IdentityEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Slice_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Slice_1
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/modFloorMod5ending/sentence_rnn/rnn_1/sentence_cell/concat_3/axisDgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Rank*
T0*
_output_shapes
: 
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Shape_1Const*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/ConcatOffsetConcatOffsetCgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/modEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/ShapeGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Shape_1*
N* 
_output_shapes
::
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/SliceSliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/ConcatOffsetEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Shape*
Index0*
T0* 
_output_shapes
:
��%
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Slice_1SliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/control_dependencyNgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/ConcatOffset:1Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Shape_1*
Index0*
T0* 
_output_shapes
:
��
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/tuple/group_depsNoOpF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/SliceH^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Slice_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/SliceQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Slice* 
_output_shapes
:
��%
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/tuple/control_dependency_1IdentityGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Slice_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*Z
_classP
NLloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Slice_1
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/MulMulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_5* 
_output_shapes
:
��*
T0
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/Mul_1MulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/tuple/control_dependency_1/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_8*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/Mul_1* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/MulMulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_5*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/Mul_1MulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/tuple/control_dependency_11ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_8*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/Mul* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/Mul_1* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_8_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_8Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_5_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_5Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_8_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_8Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_5_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_5Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
T0
�
gradients/AddN_4AddNUgradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/tuple/control_dependency_1Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_5_grad/TanhGrad*
N* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul_1
f
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/group_depsNoOp^gradients/AddN_4
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/control_dependencyIdentitygradients/AddN_4L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul_1* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/control_dependency_1Identitygradients/AddN_4L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul_1
�
gradients/AddN_5AddNWgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/tuple/control_dependency_1Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_5_grad/TanhGrad*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul_1*
N* 
_output_shapes
:
��
h
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/group_depsNoOp^gradients/AddN_5
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/control_dependencyIdentitygradients/AddN_5N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul_1* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/control_dependency_1Identitygradients/AddN_5N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/MulMulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/control_dependency+ending/sentence_rnn/rnn/sentence_cell/add_3*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul_1MulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/control_dependency/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_6*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/MulMulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_4* 
_output_shapes
:
��*
T0
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/control_dependency_1/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_7*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/Mul_1* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/MulMulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/control_dependency-ending/sentence_rnn/rnn_1/sentence_cell/add_3* 
_output_shapes
:
��*
T0
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/control_dependency1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_6*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul_1* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/MulMulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_4*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/Mul_1MulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/control_dependency_11ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_7*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/Mul* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/Mul_1* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_6_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_6Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_7_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_7Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_4_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_4Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_6_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_6Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_7_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_7Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_4_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_4Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Pgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/ShapeBgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/SumSumJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_6_grad/SigmoidGradPgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/ReshapeReshape>gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Sum@gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Shape* 
_output_shapes
:
��*
T0*
Tshape0
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Sum_1SumJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_6_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Reshape_1Reshape@gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Sum_1Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/tuple/group_depsNoOpC^gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/ReshapeE^gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Reshape_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/tuple/control_dependencyIdentityBgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/ReshapeL^gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Reshape* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/tuple/control_dependency_1IdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Reshape_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/tuple/group_deps*
_output_shapes
: *
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Reshape_1
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Rgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/ShapeDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/SumSumLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_6_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/BroadcastGradientArgs* 
_output_shapes
:
��*
	keep_dims( *

Tidx0*
T0
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/ReshapeReshape@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/SumBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Sum_1SumLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_6_grad/SigmoidGradTgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Reshape_1ReshapeBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Sum_1Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/ReshapeG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Reshape_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/ReshapeN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Reshape* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Reshape_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/tuple/group_deps*
_output_shapes
: *
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Reshape_1
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/split_2_grad/concatConcatV2Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_7_grad/SigmoidGradDgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_4_grad/TanhGradSgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/tuple/control_dependencyJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_8_grad/SigmoidGrad7ending/sentence_rnn/rnn/sentence_cell/split_2/split_dim*
T0*
N* 
_output_shapes
:
��*

Tidx0
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/split_2_grad/concatConcatV2Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_7_grad/SigmoidGradFgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_4_grad/TanhGradUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_8_grad/SigmoidGrad9ending/sentence_rnn/rnn_1/sentence_cell/split_2/split_dim*
T0*
N* 
_output_shapes
:
��*

Tidx0
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/BiasAddGradBiasAddGradCgradients/ending/sentence_rnn/rnn/sentence_cell/split_2_grad/concat*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Ogradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/group_depsNoOpK^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/BiasAddGradD^gradients/ending/sentence_rnn/rnn/sentence_cell/split_2_grad/concat
�
Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/split_2_grad/concatP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/split_2_grad/concat* 
_output_shapes
:
��
�
Ygradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/control_dependency_1IdentityJgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/BiasAddGradP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/BiasAddGrad*
_output_shapes	
:�
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/BiasAddGradBiasAddGradEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_2_grad/concat*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Qgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/group_depsNoOpM^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/BiasAddGradF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_2_grad/concat
�
Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_2_grad/concatR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_2_grad/concat* 
_output_shapes
:
��
�
[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/control_dependency_1IdentityLgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/BiasAddGradR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*_
_classU
SQloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/BiasAddGrad
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMulMatMulWgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
T0* 
_output_shapes
:
��-*
transpose_a( *
transpose_b(
�
Fgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMul_1MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_2Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/control_dependency*
T0* 
_output_shapes
:
�-�*
transpose_a(*
transpose_b( 
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMulG^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMul_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMulO^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMul* 
_output_shapes
:
��-
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMul_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMul_1* 
_output_shapes
:
�-�
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMulMatMulYgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
transpose_b(*
T0* 
_output_shapes
:
��-*
transpose_a( 
�
Hgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMul_1MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_2Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
�-�*
transpose_a(
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/group_depsNoOpG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMulI^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMul_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/control_dependencyIdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMulQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/group_deps* 
_output_shapes
:
��-*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMul
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/control_dependency_1IdentityHgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMul_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMul_1* 
_output_shapes
:
�-�
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/modFloorMod3ending/sentence_rnn/rnn/sentence_cell/concat_2/axisBgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Rank*
T0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Shape_1Const*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/ConcatOffsetConcatOffsetAgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/modCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/ShapeEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Shape_1*
N* 
_output_shapes
::
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/SliceSliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/control_dependencyJgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/ConcatOffsetCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Shape*
Index0*
T0* 
_output_shapes
:
��%
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Slice_1SliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/ConcatOffset:1Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Shape_1* 
_output_shapes
:
��*
Index0*
T0
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/tuple/group_depsNoOpD^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/SliceF^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Slice_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/SliceO^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Slice* 
_output_shapes
:
��%
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/tuple/control_dependency_1IdentityEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Slice_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Slice_1* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/modFloorMod5ending/sentence_rnn/rnn_1/sentence_cell/concat_2/axisDgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Rank*
T0*
_output_shapes
: 
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Shape_1Const*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/ConcatOffsetConcatOffsetCgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/modEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/ShapeGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Shape_1*
N* 
_output_shapes
::
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/SliceSliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/ConcatOffsetEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Shape*
Index0*
T0* 
_output_shapes
:
��%
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Slice_1SliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/control_dependencyNgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/ConcatOffset:1Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Shape_1*
Index0*
T0* 
_output_shapes
:
��
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/tuple/group_depsNoOpF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/SliceH^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Slice_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/SliceQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Slice* 
_output_shapes
:
��%
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/tuple/control_dependency_1IdentityGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Slice_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Slice_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/MulMulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_3*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/Mul_1MulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/tuple/control_dependency_1/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_5* 
_output_shapes
:
��*
T0
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/Mul_1* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/MulMulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_3*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/Mul_1MulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/tuple/control_dependency_11ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_5* 
_output_shapes
:
��*
T0
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/Mul* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/Mul_1* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_5_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_5Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_3_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_3Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_5_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_5Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_3_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_3Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
gradients/AddN_6AddNUgradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/tuple/control_dependency_1Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_3_grad/TanhGrad*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul_1*
N* 
_output_shapes
:
��
f
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/group_depsNoOp^gradients/AddN_6
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/control_dependencyIdentitygradients/AddN_6L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/control_dependency_1Identitygradients/AddN_6L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul_1* 
_output_shapes
:
��
�
gradients/AddN_7AddNWgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/tuple/control_dependency_1Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_3_grad/TanhGrad*
N* 
_output_shapes
:
��*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul_1
h
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/group_depsNoOp^gradients/AddN_7
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/control_dependencyIdentitygradients/AddN_7N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul_1* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/control_dependency_1Identitygradients/AddN_7N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul_1
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/MulMulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/control_dependency+ending/sentence_rnn/rnn/sentence_cell/add_1*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul_1MulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/control_dependency/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_3* 
_output_shapes
:
��*
T0
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul_1
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/MulMulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_2*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/control_dependency_1/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_4*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/Mul
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/Mul_1* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/MulMulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/control_dependency-ending/sentence_rnn/rnn_1/sentence_cell/add_1*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/control_dependency1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_3*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul_1* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/MulMulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_2*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/Mul_1MulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/control_dependency_11ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_4* 
_output_shapes
:
��*
T0
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/Mul
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/Mul_1* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_3_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_3Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_4_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_4Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_2_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_2Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_3_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_3Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_4_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_4Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_2_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_2Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Pgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/ShapeBgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/SumSumJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_3_grad/SigmoidGradPgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/BroadcastGradientArgs*
T0* 
_output_shapes
:
��*
	keep_dims( *

Tidx0
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/ReshapeReshape>gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Sum@gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Shape* 
_output_shapes
:
��*
T0*
Tshape0
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Sum_1SumJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_3_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Reshape_1Reshape@gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Sum_1Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/tuple/group_depsNoOpC^gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/ReshapeE^gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Reshape_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/tuple/control_dependencyIdentityBgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/ReshapeL^gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Reshape
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/tuple/control_dependency_1IdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Reshape_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Reshape_1*
_output_shapes
: 
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Rgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/ShapeDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/SumSumLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_3_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/BroadcastGradientArgs*
T0* 
_output_shapes
:
��*
	keep_dims( *

Tidx0
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/ReshapeReshape@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/SumBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Sum_1SumLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_3_grad/SigmoidGradTgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Reshape_1ReshapeBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Sum_1Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/ReshapeG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Reshape_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/ReshapeN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Reshape* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Reshape_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Reshape_1*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/split_1_grad/concatConcatV2Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_4_grad/SigmoidGradDgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_2_grad/TanhGradSgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/tuple/control_dependencyJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_5_grad/SigmoidGrad7ending/sentence_rnn/rnn/sentence_cell/split_1/split_dim*
T0*
N* 
_output_shapes
:
��*

Tidx0
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/split_1_grad/concatConcatV2Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_4_grad/SigmoidGradFgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_2_grad/TanhGradUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_5_grad/SigmoidGrad9ending/sentence_rnn/rnn_1/sentence_cell/split_1/split_dim*
T0*
N* 
_output_shapes
:
��*

Tidx0
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/BiasAddGradBiasAddGradCgradients/ending/sentence_rnn/rnn/sentence_cell/split_1_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ogradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/group_depsNoOpK^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/BiasAddGradD^gradients/ending/sentence_rnn/rnn/sentence_cell/split_1_grad/concat
�
Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/split_1_grad/concatP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/split_1_grad/concat* 
_output_shapes
:
��
�
Ygradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/control_dependency_1IdentityJgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/BiasAddGradP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/group_deps*
_output_shapes	
:�*
T0*]
_classS
QOloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/BiasAddGrad
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/BiasAddGradBiasAddGradEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_1_grad/concat*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Qgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/group_depsNoOpM^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/BiasAddGradF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_1_grad/concat
�
Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_1_grad/concatR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_1_grad/concat
�
[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/control_dependency_1IdentityLgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/BiasAddGradR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/BiasAddGrad*
_output_shapes	
:�
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMulMatMulWgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
T0* 
_output_shapes
:
��-*
transpose_a( *
transpose_b(
�
Fgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMul_1MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_1Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/control_dependency*
T0* 
_output_shapes
:
�-�*
transpose_a(*
transpose_b( 
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMulG^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMul_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMulO^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMul* 
_output_shapes
:
��-
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMul_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMul_1* 
_output_shapes
:
�-�
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMulMatMulYgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read* 
_output_shapes
:
��-*
transpose_a( *
transpose_b(*
T0
�
Hgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMul_1MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_1Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
�-�*
transpose_a(
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/group_depsNoOpG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMulI^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMul_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/control_dependencyIdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMulQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMul* 
_output_shapes
:
��-
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/control_dependency_1IdentityHgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMul_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMul_1* 
_output_shapes
:
�-�
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/modFloorMod3ending/sentence_rnn/rnn/sentence_cell/concat_1/axisBgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Rank*
T0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Shape_1Const*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/ConcatOffsetConcatOffsetAgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/modCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/ShapeEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Shape_1*
N* 
_output_shapes
::
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/SliceSliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/control_dependencyJgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/ConcatOffsetCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Shape*
Index0*
T0* 
_output_shapes
:
��%
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Slice_1SliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/ConcatOffset:1Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Shape_1*
Index0*
T0* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/tuple/group_depsNoOpD^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/SliceF^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Slice_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/SliceO^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Slice* 
_output_shapes
:
��%
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/tuple/control_dependency_1IdentityEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Slice_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Slice_1* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/modFloorMod5ending/sentence_rnn/rnn_1/sentence_cell/concat_1/axisDgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Rank*
T0*
_output_shapes
: 
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Shape_1Const*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/ConcatOffsetConcatOffsetCgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/modEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/ShapeGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Shape_1*
N* 
_output_shapes
::
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/SliceSliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/ConcatOffsetEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Shape* 
_output_shapes
:
��%*
Index0*
T0
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Slice_1SliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/control_dependencyNgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/ConcatOffset:1Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Shape_1*
Index0*
T0* 
_output_shapes
:
��
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/tuple/group_depsNoOpF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/SliceH^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Slice_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/SliceQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Slice* 
_output_shapes
:
��%
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/tuple/control_dependency_1IdentityGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Slice_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*Z
_classP
NLloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Slice_1
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/MulMulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_1*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/Mul_1MulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/tuple/control_dependency_1/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_2* 
_output_shapes
:
��*
T0
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/Mul_1* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/MulMulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_1*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/Mul_1MulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/tuple/control_dependency_11ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_2*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/Mul* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/Mul_1* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_2Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_1_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_1Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_2Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_1_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_1Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
T0
�
gradients/AddN_8AddNUgradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/tuple/control_dependency_1Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_1_grad/TanhGrad*
N* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul_1
f
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/group_depsNoOp^gradients/AddN_8
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/control_dependencyIdentitygradients/AddN_8L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul_1* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/control_dependency_1Identitygradients/AddN_8L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul_1
�
gradients/AddN_9AddNWgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/tuple/control_dependency_1Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_1_grad/TanhGrad*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul_1*
N* 
_output_shapes
:
��
h
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/group_depsNoOp^gradients/AddN_9
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/control_dependencyIdentitygradients/AddN_9N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul_1* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/control_dependency_1Identitygradients/AddN_9N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul_1* 
_output_shapes
:
��
�
<gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/MulMulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/control_dependency/ending/sentence_rnn/rnn/LSTMCellZeroState/zeros* 
_output_shapes
:
��*
T0
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/Mul_1MulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/control_dependency-ending/sentence_rnn/rnn/sentence_cell/Sigmoid*
T0* 
_output_shapes
:
��
�
Igradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/tuple/group_depsNoOp=^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/Mul?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/Mul_1
�
Qgradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/tuple/control_dependencyIdentity<gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/MulJ^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/Mul
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/tuple/control_dependency_1Identity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/Mul_1J^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/Mul_1
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/MulMulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/control_dependency_1*ending/sentence_rnn/rnn/sentence_cell/Tanh*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/control_dependency_1/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_1* 
_output_shapes
:
��*
T0
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/Mul_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/MulMulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/control_dependency1ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/control_dependency/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid* 
_output_shapes
:
��*
T0
�
Kgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/MulA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/MulL^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/Mul_1L^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/Mul_1
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/MulMulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn_1/sentence_cell/Tanh*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/Mul_1MulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/control_dependency_11ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_1*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/Mul* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/Mul_1
�
Hgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_grad/SigmoidGradSigmoidGrad-ending/sentence_rnn/rnn/sentence_cell/SigmoidQgradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_1Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_grad/TanhGradTanhGrad*ending/sentence_rnn/rnn/sentence_cell/TanhUgradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
T0
�
Jgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn_1/sentence_cell/SigmoidSgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_1Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn_1/sentence_cell/TanhWgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Shape@gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/SumSumHgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_grad/SigmoidGradNgradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/ReshapeReshape<gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Sum>gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Sum_1SumHgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_grad/SigmoidGradPgradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Reshape_1Reshape>gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Sum_1@gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Igradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/ReshapeC^gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Reshape_1
�
Qgradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/ReshapeJ^gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Reshape
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Reshape_1J^gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Reshape_1*
_output_shapes
: 
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/ShapeBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/SumSumJgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_grad/SigmoidGradPgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/BroadcastGradientArgs* 
_output_shapes
:
��*
	keep_dims( *

Tidx0*
T0
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/ReshapeReshape>gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Sum@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Sum_1SumJgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Reshape_1Reshape@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Sum_1Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Kgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/tuple/group_depsNoOpC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/ReshapeE^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Reshape_1
�
Sgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/tuple/control_dependencyIdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/ReshapeL^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Reshape* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/tuple/control_dependency_1IdentityDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Reshape_1L^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Reshape_1*
_output_shapes
: 
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/split_grad/concatConcatV2Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_1_grad/SigmoidGradBgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_grad/TanhGradQgradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/tuple/control_dependencyJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_2_grad/SigmoidGrad5ending/sentence_rnn/rnn/sentence_cell/split/split_dim*
T0*
N* 
_output_shapes
:
��*

Tidx0
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_grad/concatConcatV2Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_1_grad/SigmoidGradDgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_grad/TanhGradSgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_2_grad/SigmoidGrad7ending/sentence_rnn/rnn_1/sentence_cell/split/split_dim*
N* 
_output_shapes
:
��*

Tidx0*
T0
�
Hgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/BiasAddGradBiasAddGradAgradients/ending/sentence_rnn/rnn/sentence_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Mgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/group_depsNoOpI^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/BiasAddGradB^gradients/ending/sentence_rnn/rnn/sentence_cell/split_grad/concat
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/control_dependencyIdentityAgradients/ending/sentence_rnn/rnn/sentence_cell/split_grad/concatN^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/split_grad/concat* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/control_dependency_1IdentityHgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/BiasAddGradN^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*[
_classQ
OMloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/BiasAddGrad
�
Jgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/BiasAddGradBiasAddGradCgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ogradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/group_depsNoOpK^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/BiasAddGradD^gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_grad/concat
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_grad/concatP^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_grad/concat* 
_output_shapes
:
��
�
Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/BiasAddGradP^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*]
_classS
QOloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/BiasAddGrad
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMulMatMulUgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
transpose_b(*
T0* 
_output_shapes
:
��-*
transpose_a( 
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMul_1MatMul,ending/sentence_rnn/rnn/sentence_cell/concatUgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/control_dependency*
T0* 
_output_shapes
:
�-�*
transpose_a(*
transpose_b( 
�
Lgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/tuple/group_depsNoOpC^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMulE^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMul_1
�
Tgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/tuple/control_dependencyIdentityBgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMulM^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��-*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMul
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/tuple/control_dependency_1IdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMul_1M^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/tuple/group_deps* 
_output_shapes
:
�-�*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMul_1
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMulMatMulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
transpose_b(*
T0* 
_output_shapes
:
��-*
transpose_a( 
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMul_1MatMul.ending/sentence_rnn/rnn_1/sentence_cell/concatWgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/control_dependency* 
_output_shapes
:
�-�*
transpose_a(*
transpose_b( *
T0
�
Ngradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMulG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMul_1
�
Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMulO^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMul* 
_output_shapes
:
��-
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMul_1O^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
�-�
�
gradients/AddN_10AddNYgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/control_dependency_1[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/control_dependency_1Ygradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/control_dependency_1[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/control_dependency_1Ygradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/control_dependency_1[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/control_dependency_1Ygradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/control_dependency_1[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/control_dependency_1Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/control_dependency_1Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/control_dependency_1*
N
*
_output_shapes	
:�*
T0*]
_classS
QOloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/BiasAddGrad
�
gradients/AddN_11AddNXgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/control_dependency_1Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/control_dependency_1Xgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/control_dependency_1Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/control_dependency_1Xgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/control_dependency_1Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/control_dependency_1Xgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/control_dependency_1Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/control_dependency_1Vgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/tuple/control_dependency_1Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/tuple/control_dependency_1*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMul_1*
N
* 
_output_shapes
:
�-�
h
clip_by_norm/mulMulgradients/AddN_11gradients/AddN_11*
T0* 
_output_shapes
:
�-�
c
clip_by_norm/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
clip_by_norm/SumSumclip_by_norm/mulclip_by_norm/Const*
_output_shapes

:*
	keep_dims(*

Tidx0*
T0
[
clip_by_norm/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
r
clip_by_norm/GreaterGreaterclip_by_norm/Sumclip_by_norm/Greater/y*
T0*
_output_shapes

:
m
clip_by_norm/ones_like/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
a
clip_by_norm/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
clip_by_norm/ones_likeFillclip_by_norm/ones_like/Shapeclip_by_norm/ones_like/Const*
T0*

index_type0*
_output_shapes

:
�
clip_by_norm/SelectSelectclip_by_norm/Greaterclip_by_norm/Sumclip_by_norm/ones_like*
T0*
_output_shapes

:
W
clip_by_norm/SqrtSqrtclip_by_norm/Select*
T0*
_output_shapes

:
�
clip_by_norm/Select_1Selectclip_by_norm/Greaterclip_by_norm/Sqrtclip_by_norm/Sum*
T0*
_output_shapes

:
Y
clip_by_norm/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
m
clip_by_norm/mul_1Mulgradients/AddN_11clip_by_norm/mul_1/y*
T0* 
_output_shapes
:
�-�
[
clip_by_norm/Maximum/yConst*
dtype0*
_output_shapes
: *
valueB
 *   A
w
clip_by_norm/MaximumMaximumclip_by_norm/Select_1clip_by_norm/Maximum/y*
T0*
_output_shapes

:
t
clip_by_norm/truedivRealDivclip_by_norm/mul_1clip_by_norm/Maximum* 
_output_shapes
:
�-�*
T0
Y
clip_by_normIdentityclip_by_norm/truediv* 
_output_shapes
:
�-�*
T0
e
clip_by_norm_1/mulMulgradients/AddN_10gradients/AddN_10*
_output_shapes	
:�*
T0
^
clip_by_norm_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
clip_by_norm_1/SumSumclip_by_norm_1/mulclip_by_norm_1/Const*
T0*
_output_shapes
:*
	keep_dims(*

Tidx0
]
clip_by_norm_1/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
t
clip_by_norm_1/GreaterGreaterclip_by_norm_1/Sumclip_by_norm_1/Greater/y*
_output_shapes
:*
T0
h
clip_by_norm_1/ones_like/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
c
clip_by_norm_1/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
clip_by_norm_1/ones_likeFillclip_by_norm_1/ones_like/Shapeclip_by_norm_1/ones_like/Const*
T0*

index_type0*
_output_shapes
:
�
clip_by_norm_1/SelectSelectclip_by_norm_1/Greaterclip_by_norm_1/Sumclip_by_norm_1/ones_like*
T0*
_output_shapes
:
W
clip_by_norm_1/SqrtSqrtclip_by_norm_1/Select*
T0*
_output_shapes
:
�
clip_by_norm_1/Select_1Selectclip_by_norm_1/Greaterclip_by_norm_1/Sqrtclip_by_norm_1/Sum*
T0*
_output_shapes
:
[
clip_by_norm_1/mul_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   A
l
clip_by_norm_1/mul_1Mulgradients/AddN_10clip_by_norm_1/mul_1/y*
T0*
_output_shapes	
:�
]
clip_by_norm_1/Maximum/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
y
clip_by_norm_1/MaximumMaximumclip_by_norm_1/Select_1clip_by_norm_1/Maximum/y*
T0*
_output_shapes
:
u
clip_by_norm_1/truedivRealDivclip_by_norm_1/mul_1clip_by_norm_1/Maximum*
T0*
_output_shapes	
:�
X
clip_by_norm_1Identityclip_by_norm_1/truediv*
T0*
_output_shapes	
:�
g
clip_by_norm_2/mulMulgradients/AddN_1gradients/AddN_1*
T0*
_output_shapes
:	�
e
clip_by_norm_2/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
clip_by_norm_2/SumSumclip_by_norm_2/mulclip_by_norm_2/Const*
_output_shapes

:*
	keep_dims(*

Tidx0*
T0
]
clip_by_norm_2/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
x
clip_by_norm_2/GreaterGreaterclip_by_norm_2/Sumclip_by_norm_2/Greater/y*
T0*
_output_shapes

:
o
clip_by_norm_2/ones_like/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
c
clip_by_norm_2/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
clip_by_norm_2/ones_likeFillclip_by_norm_2/ones_like/Shapeclip_by_norm_2/ones_like/Const*
T0*

index_type0*
_output_shapes

:
�
clip_by_norm_2/SelectSelectclip_by_norm_2/Greaterclip_by_norm_2/Sumclip_by_norm_2/ones_like*
T0*
_output_shapes

:
[
clip_by_norm_2/SqrtSqrtclip_by_norm_2/Select*
T0*
_output_shapes

:
�
clip_by_norm_2/Select_1Selectclip_by_norm_2/Greaterclip_by_norm_2/Sqrtclip_by_norm_2/Sum*
T0*
_output_shapes

:
[
clip_by_norm_2/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
o
clip_by_norm_2/mul_1Mulgradients/AddN_1clip_by_norm_2/mul_1/y*
T0*
_output_shapes
:	�
]
clip_by_norm_2/Maximum/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
}
clip_by_norm_2/MaximumMaximumclip_by_norm_2/Select_1clip_by_norm_2/Maximum/y*
T0*
_output_shapes

:
y
clip_by_norm_2/truedivRealDivclip_by_norm_2/mul_1clip_by_norm_2/Maximum*
_output_shapes
:	�*
T0
\
clip_by_norm_2Identityclip_by_norm_2/truediv*
_output_shapes
:	�*
T0
^
clip_by_norm_3/mulMulgradients/AddNgradients/AddN*
T0*
_output_shapes
:
^
clip_by_norm_3/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
clip_by_norm_3/SumSumclip_by_norm_3/mulclip_by_norm_3/Const*
T0*
_output_shapes
:*
	keep_dims(*

Tidx0
]
clip_by_norm_3/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
t
clip_by_norm_3/GreaterGreaterclip_by_norm_3/Sumclip_by_norm_3/Greater/y*
T0*
_output_shapes
:
h
clip_by_norm_3/ones_like/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
c
clip_by_norm_3/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
clip_by_norm_3/ones_likeFillclip_by_norm_3/ones_like/Shapeclip_by_norm_3/ones_like/Const*
_output_shapes
:*
T0*

index_type0
�
clip_by_norm_3/SelectSelectclip_by_norm_3/Greaterclip_by_norm_3/Sumclip_by_norm_3/ones_like*
T0*
_output_shapes
:
W
clip_by_norm_3/SqrtSqrtclip_by_norm_3/Select*
_output_shapes
:*
T0
�
clip_by_norm_3/Select_1Selectclip_by_norm_3/Greaterclip_by_norm_3/Sqrtclip_by_norm_3/Sum*
T0*
_output_shapes
:
[
clip_by_norm_3/mul_1/yConst*
dtype0*
_output_shapes
: *
valueB
 *   A
h
clip_by_norm_3/mul_1Mulgradients/AddNclip_by_norm_3/mul_1/y*
T0*
_output_shapes
:
]
clip_by_norm_3/Maximum/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
y
clip_by_norm_3/MaximumMaximumclip_by_norm_3/Select_1clip_by_norm_3/Maximum/y*
T0*
_output_shapes
:
t
clip_by_norm_3/truedivRealDivclip_by_norm_3/mul_1clip_by_norm_3/Maximum*
T0*
_output_shapes
:
W
clip_by_norm_3Identityclip_by_norm_3/truediv*
T0*
_output_shapes
:
�
beta1_power/initial_valueConst*
valueB
 *fff?*%
_class
loc:@ending/output/bias*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
shared_name *%
_class
loc:@ending/output/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*%
_class
loc:@ending/output/bias*
validate_shape(*
_output_shapes
: *
use_locking(
q
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*%
_class
loc:@ending/output/bias
�
beta2_power/initial_valueConst*
valueB
 *w�?*%
_class
loc:@ending/output/bias*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
shared_name *%
_class
loc:@ending/output/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
use_locking(*
T0*%
_class
loc:@ending/output/bias*
validate_shape(*
_output_shapes
: 
q
beta2_power/readIdentitybeta2_power*
T0*%
_class
loc:@ending/output/bias*
_output_shapes
: 
�
Fending/rnn/sentence_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
valueB"�  �  *
dtype0*
_output_shapes
:
�
<ending/rnn/sentence_cell/kernel/Adam/Initializer/zeros/ConstConst*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6ending/rnn/sentence_cell/kernel/Adam/Initializer/zerosFillFending/rnn/sentence_cell/kernel/Adam/Initializer/zeros/shape_as_tensor<ending/rnn/sentence_cell/kernel/Adam/Initializer/zeros/Const*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*

index_type0* 
_output_shapes
:
�-�
�
$ending/rnn/sentence_cell/kernel/Adam
VariableV2*
shared_name *2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
	container *
shape:
�-�*
dtype0* 
_output_shapes
:
�-�
�
+ending/rnn/sentence_cell/kernel/Adam/AssignAssign$ending/rnn/sentence_cell/kernel/Adam6ending/rnn/sentence_cell/kernel/Adam/Initializer/zeros*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
validate_shape(* 
_output_shapes
:
�-�*
use_locking(
�
)ending/rnn/sentence_cell/kernel/Adam/readIdentity$ending/rnn/sentence_cell/kernel/Adam*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel* 
_output_shapes
:
�-�
�
Hending/rnn/sentence_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
valueB"�  �  *
dtype0*
_output_shapes
:
�
>ending/rnn/sentence_cell/kernel/Adam_1/Initializer/zeros/ConstConst*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
8ending/rnn/sentence_cell/kernel/Adam_1/Initializer/zerosFillHending/rnn/sentence_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensor>ending/rnn/sentence_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*

index_type0* 
_output_shapes
:
�-�
�
&ending/rnn/sentence_cell/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
�-�*
shared_name *2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
	container *
shape:
�-�
�
-ending/rnn/sentence_cell/kernel/Adam_1/AssignAssign&ending/rnn/sentence_cell/kernel/Adam_18ending/rnn/sentence_cell/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
validate_shape(* 
_output_shapes
:
�-�
�
+ending/rnn/sentence_cell/kernel/Adam_1/readIdentity&ending/rnn/sentence_cell/kernel/Adam_1*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel* 
_output_shapes
:
�-�
�
Dending/rnn/sentence_cell/bias/Adam/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
valueB:�*
dtype0*
_output_shapes
:
�
:ending/rnn/sentence_cell/bias/Adam/Initializer/zeros/ConstConst*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
4ending/rnn/sentence_cell/bias/Adam/Initializer/zerosFillDending/rnn/sentence_cell/bias/Adam/Initializer/zeros/shape_as_tensor:ending/rnn/sentence_cell/bias/Adam/Initializer/zeros/Const*
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*

index_type0*
_output_shapes	
:�
�
"ending/rnn/sentence_cell/bias/Adam
VariableV2*
shared_name *0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
)ending/rnn/sentence_cell/bias/Adam/AssignAssign"ending/rnn/sentence_cell/bias/Adam4ending/rnn/sentence_cell/bias/Adam/Initializer/zeros*
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
'ending/rnn/sentence_cell/bias/Adam/readIdentity"ending/rnn/sentence_cell/bias/Adam*
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
_output_shapes	
:�
�
Fending/rnn/sentence_cell/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
valueB:�*
dtype0*
_output_shapes
:
�
<ending/rnn/sentence_cell/bias/Adam_1/Initializer/zeros/ConstConst*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
valueB
 *    *
dtype0*
_output_shapes
: 
�
6ending/rnn/sentence_cell/bias/Adam_1/Initializer/zerosFillFending/rnn/sentence_cell/bias/Adam_1/Initializer/zeros/shape_as_tensor<ending/rnn/sentence_cell/bias/Adam_1/Initializer/zeros/Const*
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*

index_type0*
_output_shapes	
:�
�
$ending/rnn/sentence_cell/bias/Adam_1
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
	container 
�
+ending/rnn/sentence_cell/bias/Adam_1/AssignAssign$ending/rnn/sentence_cell/bias/Adam_16ending/rnn/sentence_cell/bias/Adam_1/Initializer/zeros*
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
)ending/rnn/sentence_cell/bias/Adam_1/readIdentity$ending/rnn/sentence_cell/bias/Adam_1*
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
_output_shapes	
:�
�
;ending/output/kernel/Adam/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@ending/output/kernel*
valueB"�     *
dtype0*
_output_shapes
:
�
1ending/output/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *'
_class
loc:@ending/output/kernel*
valueB
 *    
�
+ending/output/kernel/Adam/Initializer/zerosFill;ending/output/kernel/Adam/Initializer/zeros/shape_as_tensor1ending/output/kernel/Adam/Initializer/zeros/Const*
T0*'
_class
loc:@ending/output/kernel*

index_type0*
_output_shapes
:	�
�
ending/output/kernel/Adam
VariableV2*
shared_name *'
_class
loc:@ending/output/kernel*
	container *
shape:	�*
dtype0*
_output_shapes
:	�
�
 ending/output/kernel/Adam/AssignAssignending/output/kernel/Adam+ending/output/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0*'
_class
loc:@ending/output/kernel
�
ending/output/kernel/Adam/readIdentityending/output/kernel/Adam*
T0*'
_class
loc:@ending/output/kernel*
_output_shapes
:	�
�
=ending/output/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@ending/output/kernel*
valueB"�     *
dtype0*
_output_shapes
:
�
3ending/output/kernel/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *'
_class
loc:@ending/output/kernel*
valueB
 *    
�
-ending/output/kernel/Adam_1/Initializer/zerosFill=ending/output/kernel/Adam_1/Initializer/zeros/shape_as_tensor3ending/output/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes
:	�*
T0*'
_class
loc:@ending/output/kernel*

index_type0
�
ending/output/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	�*
shared_name *'
_class
loc:@ending/output/kernel*
	container *
shape:	�
�
"ending/output/kernel/Adam_1/AssignAssignending/output/kernel/Adam_1-ending/output/kernel/Adam_1/Initializer/zeros*
T0*'
_class
loc:@ending/output/kernel*
validate_shape(*
_output_shapes
:	�*
use_locking(
�
 ending/output/kernel/Adam_1/readIdentityending/output/kernel/Adam_1*
T0*'
_class
loc:@ending/output/kernel*
_output_shapes
:	�
�
)ending/output/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:*%
_class
loc:@ending/output/bias*
valueB*    
�
ending/output/bias/Adam
VariableV2*
dtype0*
_output_shapes
:*
shared_name *%
_class
loc:@ending/output/bias*
	container *
shape:
�
ending/output/bias/Adam/AssignAssignending/output/bias/Adam)ending/output/bias/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@ending/output/bias*
validate_shape(*
_output_shapes
:
�
ending/output/bias/Adam/readIdentityending/output/bias/Adam*
_output_shapes
:*
T0*%
_class
loc:@ending/output/bias
�
+ending/output/bias/Adam_1/Initializer/zerosConst*%
_class
loc:@ending/output/bias*
valueB*    *
dtype0*
_output_shapes
:
�
ending/output/bias/Adam_1
VariableV2*
shared_name *%
_class
loc:@ending/output/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
 ending/output/bias/Adam_1/AssignAssignending/output/bias/Adam_1+ending/output/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@ending/output/bias*
validate_shape(*
_output_shapes
:
�
ending/output/bias/Adam_1/readIdentityending/output/bias/Adam_1*
T0*%
_class
loc:@ending/output/bias*
_output_shapes
:
W
Adam/learning_rateConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
5Adam/update_ending/rnn/sentence_cell/kernel/ApplyAdam	ApplyAdamending/rnn/sentence_cell/kernel$ending/rnn/sentence_cell/kernel/Adam&ending/rnn/sentence_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonclip_by_norm*
use_locking( *
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
use_nesterov( * 
_output_shapes
:
�-�
�
3Adam/update_ending/rnn/sentence_cell/bias/ApplyAdam	ApplyAdamending/rnn/sentence_cell/bias"ending/rnn/sentence_cell/bias/Adam$ending/rnn/sentence_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonclip_by_norm_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias
�
*Adam/update_ending/output/kernel/ApplyAdam	ApplyAdamending/output/kernelending/output/kernel/Adamending/output/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonclip_by_norm_2*
use_nesterov( *
_output_shapes
:	�*
use_locking( *
T0*'
_class
loc:@ending/output/kernel
�
(Adam/update_ending/output/bias/ApplyAdam	ApplyAdamending/output/biasending/output/bias/Adamending/output/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonclip_by_norm_3*
T0*%
_class
loc:@ending/output/bias*
use_nesterov( *
_output_shapes
:*
use_locking( 
�
Adam/mulMulbeta1_power/read
Adam/beta1)^Adam/update_ending/output/bias/ApplyAdam+^Adam/update_ending/output/kernel/ApplyAdam4^Adam/update_ending/rnn/sentence_cell/bias/ApplyAdam6^Adam/update_ending/rnn/sentence_cell/kernel/ApplyAdam*
T0*%
_class
loc:@ending/output/bias*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*%
_class
loc:@ending/output/bias*
validate_shape(*
_output_shapes
: 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2)^Adam/update_ending/output/bias/ApplyAdam+^Adam/update_ending/output/kernel/ApplyAdam4^Adam/update_ending/rnn/sentence_cell/bias/ApplyAdam6^Adam/update_ending/rnn/sentence_cell/kernel/ApplyAdam*
T0*%
_class
loc:@ending/output/bias*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*%
_class
loc:@ending/output/bias*
validate_shape(*
_output_shapes
: 
�
Adam/updateNoOp^Adam/Assign^Adam/Assign_1)^Adam/update_ending/output/bias/ApplyAdam+^Adam/update_ending/output/kernel/ApplyAdam4^Adam/update_ending/rnn/sentence_cell/bias/ApplyAdam6^Adam/update_ending/rnn/sentence_cell/kernel/ApplyAdam
z

Adam/valueConst^Adam/update*
value	B :*
_class
loc:@global_step*
dtype0*
_output_shapes
: 
~
Adam	AssignAddglobal_step
Adam/value*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@global_step
N
	loss/tagsConst*
dtype0*
_output_shapes
: *
valueB
 Bloss
G
lossScalarSummary	loss/tagsMean*
_output_shapes
: *
T0
V
accuracy/tagsConst*
valueB Baccuracy*
dtype0*
_output_shapes
: 
Q
accuracyScalarSummaryaccuracy/tagsMean_1*
_output_shapes
: *
T0
S
Merge/MergeSummaryMergeSummarylossaccuracy*
N*
_output_shapes
: 
U
Merge_1/MergeSummaryMergeSummarylossaccuracy*
N*
_output_shapes
: �o
�*
�
*Dataset_map_<class 'functools.partial'>_64
arg0
arg1
concat_1
cast2DWrapper for passing nested structures to and from tf.data functions.�A
strided_slice/stackConst*
valueB: *
dtype0C
strided_slice/stack_1Const*
valueB:*
dtype0C
strided_slice/stack_2Const*
dtype0*
valueB:�
strided_sliceStridedSlicearg1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0B
random_uniform/shapeConst*
valueB:*
dtype0?
random_uniform/minConst*
valueB
 *    *
dtype0?
random_uniform/maxConst*
valueB
 *  �?*
dtype0{
random_uniform/RandomUniformRandomUniformrandom_uniform/shape:output:0*
dtype0*
seed2 *

seed *
T0\
random_uniform/subSubrandom_uniform/max:output:0random_uniform/min:output:0*
T0a
random_uniform/mulMul%random_uniform/RandomUniform:output:0random_uniform/sub:z:0*
T0S
random_uniformAddrandom_uniform/mul:z:0random_uniform/min:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
dtype0*
valueB:E
strided_slice_1/stack_2Const*
valueB:*
dtype0�
strided_slice_1StridedSlicerandom_uniform:z:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask 3
Less/yConst*
valueB
 *UUU?*
dtype0@
LessLessstrided_slice_1:output:0Less/y:output:0*
T02
cond/SwitchSwitchLess:z:0Less:z:0*
T0
=
cond/switch_tIdentitycond/Switch:output_true:0*
T0
>
cond/switch_fIdentitycond/Switch:output_false:0*
T0
+
cond/pred_idIdentityLess:z:0*
T0
�
cond/OneShotIteratorOneShotIterator^cond/switch_t*0
dataset_factoryR
_make_dataset_KVahewP2F78*
output_types	
2*
shared_name *6
output_shapes%
#:�%:�%:�%:�%:�%*
	container T
cond/IteratorToStringHandleIteratorToStringHandlecond/OneShotIterator:handle:0�
cond/IteratorGetNextIteratorGetNextcond/OneShotIterator:handle:0*6
output_shapes%
#:�%:�%:�%:�%:�%*
output_types	
2S

cond/stackPack!cond/IteratorGetNext:components:4*
N*
T0*

axis W
cond/random_uniform/shapeConst^cond/switch_f*
valueB:*
dtype0Q
cond/random_uniform/minConst^cond/switch_f*
value	B : *
dtype0Q
cond/random_uniform/maxConst^cond/switch_f*
value	B :*
dtype0�
cond/random_uniformRandomUniformInt"cond/random_uniform/shape:output:0 cond/random_uniform/min:output:0 cond/random_uniform/max:output:0*
T0*
seed2 *

Tout0*

seed L
cond/GatherV2/axisConst^cond/switch_f*
value	B : *
dtype0�
cond/GatherV2GatherV2#cond/GatherV2/Switch:output_false:0cond/random_uniform:output:0cond/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0]
cond/GatherV2/SwitchSwitcharg0cond/pred_id:output:0*
T0*
_class
	loc:@arg0R

cond/MergeMergecond/GatherV2:output:0cond/stack:output:0*
T0*
N8
ExpandDims/dimConst*
value	B : *
dtype0^

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*

Tdim0*
T05
concat/axisConst*
value	B : *
dtype0p
concatConcatV2ExpandDims:output:0cond/Merge:output:0concat/axis:output:0*
T0*
N*

Tidx0:
one_hot/on_valueConst*
value	B :*
dtype0;
one_hot/off_valueConst*
value	B : *
dtype09
one_hot/indicesConst*
value	B	 R *
dtype0	7
one_hot/depthConst*
value	B :*
dtype0�
one_hotOneHotone_hot/indices:output:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
axis���������*
TI0	5
range/startConst*
dtype0*
value	B : 5
range/limitConst*
dtype0*
value	B :5
range/deltaConst*
value	B :*
dtype0\
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*

Tidx0P
RandomShuffleRandomShufflerange:output:0*
T0*
seed2 *

seed 7
GatherV2/axisConst*
value	B : *
dtype0�
GatherV2GatherV2concat:output:0RandomShuffle:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams09
GatherV2_1/axisConst*
value	B : *
dtype0�

GatherV2_1GatherV2one_hot:output:0RandomShuffle:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0:
ArgMax/dimensionConst*
dtype0*
value	B : h
ArgMaxArgMaxGatherV2_1:output:0ArgMax/dimension:output:0*
T0*
output_type0	*

Tidx0E
CastCastArgMax:output:0*

SrcT0	*
Truncate( *

DstT07
concat_1/axisConst*
dtype0*
value	B : e

concat_1_0ConcatV2arg0GatherV2:output:0concat_1/axis:output:0*
T0*
N*

Tidx0"
castCast:y:0"
concat_1concat_1_0:output:0
�
�
"Dataset_flat_map_read_one_file_151
arg0
tfrecorddataset2DWrapper for passing nested structures to and from tf.data functions.�9
compression_typeConst*
dtype0*
valueB B 7
buffer_sizeConst*
valueB		 R��*
dtype0	Y
TFRecordDatasetTFRecordDatasetarg0compression_type:output:0buffer_size:output:0"+
tfrecorddatasetTFRecordDataset:handle:0
�
�
Dataset_map_tensorize_dict_172
arg0
arg1
arg2
arg3
arg4
arg5	
stack2DWrapper for passing nested structures to and from tf.data functions.Q
stack_0Packarg0arg1arg2arg3arg4arg5*
T0*

axis *
N"
stackstack_0:output:0
�
�
Dataset_map_extract_fn_35
arg0)
%parsesingleexample_parsesingleexample+
'parsesingleexample_parsesingleexample_0+
'parsesingleexample_parsesingleexample_1+
'parsesingleexample_parsesingleexample_2+
'parsesingleexample_parsesingleexample_32DWrapper for passing nested structures to and from tf.data functions.A
ParseSingleExample/ConstConst*
valueB *
dtype0C
ParseSingleExample/Const_1Const*
dtype0*
valueB C
ParseSingleExample/Const_2Const*
valueB *
dtype0C
ParseSingleExample/Const_3Const*
valueB *
dtype0C
ParseSingleExample/Const_4Const*
dtype0*
valueB �
%ParseSingleExample/ParseSingleExampleParseSingleExamplearg0!ParseSingleExample/Const:output:0#ParseSingleExample/Const_1:output:0#ParseSingleExample/Const_2:output:0#ParseSingleExample/Const_3:output:0#ParseSingleExample/Const_4:output:0*

num_sparse *G

dense_keys9
7	sentence1	sentence2	sentence3	sentence4	sentence5*5
dense_shapes%
#:�%:�%:�%:�%:�%*
sparse_types
 *
sparse_keys
 *
Tdense	
2"_
'parsesingleexample_parsesingleexample_24ParseSingleExample/ParseSingleExample:dense_values:3"_
'parsesingleexample_parsesingleexample_34ParseSingleExample/ParseSingleExample:dense_values:4"]
%parsesingleexample_parsesingleexample4ParseSingleExample/ParseSingleExample:dense_values:0"_
'parsesingleexample_parsesingleexample_04ParseSingleExample/ParseSingleExample:dense_values:1"_
'parsesingleexample_parsesingleexample_14ParseSingleExample/ParseSingleExample:dense_values:2
�
Q
_make_dataset_KVahewP2F78
modeldataset2Factory function for a dataset.�{
optimizationsConst*
dtype0*V
valueMBKBmap_and_batch_fusionBnoop_eliminationBshuffle_and_repeat_fusion�
TensorSliceDataset/ConstConst*�
value�B~ Bx/Users/arthur/Documents/natural_lanugage_understanding/nlu_project2/data/processed/train_stories_skip_thoughts.tfrecords*
dtype0^
'TensorSliceDataset/flat_filenames/shapeConst*
valueB:
���������*
dtype0�
!TensorSliceDataset/flat_filenamesReshape!TensorSliceDataset/Const:output:00TensorSliceDataset/flat_filenames/shape:output:0*
T0*
Tshape0�
TensorSliceDatasetTensorSliceDataset*TensorSliceDataset/flat_filenames:output:0*
output_shapes
: *
Toutput_types
2�
FlatMapDatasetFlatMapDatasetTensorSliceDataset:handle:0*)
f$R"
 Dataset_flat_map_read_one_file_4*
output_types
2*

Targuments
 *
output_shapes
: �

MapDataset
MapDatasetFlatMapDataset:handle:0*
output_types	
2*
use_inter_op_parallelism(*

Targuments
 *
preserve_cardinality( *6
output_shapes%
#:�%:�%:�%:�%:�%*"
fR
Dataset_map_extract_fn_10�
OptimizeDatasetOptimizeDatasetMapDataset:handle:0optimizations:output:0*6
output_shapes%
#:�%:�%:�%:�%:�%*
output_types	
2�
ModelDatasetModelDatasetOptimizeDataset:handle:0*6
output_shapes%
#:�%:�%:�%:�%:�%*
output_types	
2"%
modeldatasetModelDataset:handle:0
�
�
 Dataset_flat_map_read_one_file_4
arg0
tfrecorddataset2DWrapper for passing nested structures to and from tf.data functions.�9
compression_typeConst*
valueB B *
dtype07
buffer_sizeConst*
valueB		 R��*
dtype0	Y
TFRecordDatasetTFRecordDatasetarg0compression_type:output:0buffer_size:output:0"+
tfrecorddatasetTFRecordDataset:handle:0
�
�
!Dataset_flat_map_read_one_file_29
arg0
tfrecorddataset2DWrapper for passing nested structures to and from tf.data functions.�9
compression_typeConst*
valueB B *
dtype07
buffer_sizeConst*
dtype0	*
valueB		 R��Y
TFRecordDatasetTFRecordDatasetarg0compression_type:output:0buffer_size:output:0"+
tfrecorddatasetTFRecordDataset:handle:0
�
�
Dataset_map_extract_fn_157
arg0)
%parsesingleexample_parsesingleexample+
'parsesingleexample_parsesingleexample_0+
'parsesingleexample_parsesingleexample_1+
'parsesingleexample_parsesingleexample_2+
'parsesingleexample_parsesingleexample_3+
'parsesingleexample_parsesingleexample_42DWrapper for passing nested structures to and from tf.data functions.A
ParseSingleExample/ConstConst*
valueB *
dtype0C
ParseSingleExample/Const_1Const*
valueB *
dtype0C
ParseSingleExample/Const_2Const*
dtype0*
valueB C
ParseSingleExample/Const_3Const*
valueB *
dtype0C
ParseSingleExample/Const_4Const*
dtype0*
valueB C
ParseSingleExample/Const_5Const*
dtype0*
valueB �
%ParseSingleExample/ParseSingleExampleParseSingleExamplearg0!ParseSingleExample/Const:output:0#ParseSingleExample/Const_1:output:0#ParseSingleExample/Const_2:output:0#ParseSingleExample/Const_3:output:0#ParseSingleExample/Const_4:output:0#ParseSingleExample/Const_5:output:0*
sparse_types
 *<
dense_shapes,
*:�%:�%:�%:�%:�%:�%*
sparse_keys
 *
Tdense

2*

num_sparse *N

dense_keys@
>ending1ending2	sentence1	sentence2	sentence3	sentence4"_
'parsesingleexample_parsesingleexample_24ParseSingleExample/ParseSingleExample:dense_values:3"_
'parsesingleexample_parsesingleexample_34ParseSingleExample/ParseSingleExample:dense_values:4"_
'parsesingleexample_parsesingleexample_44ParseSingleExample/ParseSingleExample:dense_values:5"]
%parsesingleexample_parsesingleexample4ParseSingleExample/ParseSingleExample:dense_values:0"_
'parsesingleexample_parsesingleexample_04ParseSingleExample/ParseSingleExample:dense_values:1"_
'parsesingleexample_parsesingleexample_14ParseSingleExample/ParseSingleExample:dense_values:2
�
�
Dataset_map_extract_fn_10
arg0)
%parsesingleexample_parsesingleexample+
'parsesingleexample_parsesingleexample_0+
'parsesingleexample_parsesingleexample_1+
'parsesingleexample_parsesingleexample_2+
'parsesingleexample_parsesingleexample_32DWrapper for passing nested structures to and from tf.data functions.A
ParseSingleExample/ConstConst*
valueB *
dtype0C
ParseSingleExample/Const_1Const*
valueB *
dtype0C
ParseSingleExample/Const_2Const*
dtype0*
valueB C
ParseSingleExample/Const_3Const*
valueB *
dtype0C
ParseSingleExample/Const_4Const*
valueB *
dtype0�
%ParseSingleExample/ParseSingleExampleParseSingleExamplearg0!ParseSingleExample/Const:output:0#ParseSingleExample/Const_1:output:0#ParseSingleExample/Const_2:output:0#ParseSingleExample/Const_3:output:0#ParseSingleExample/Const_4:output:0*
Tdense	
2*

num_sparse *G

dense_keys9
7	sentence1	sentence2	sentence3	sentence4	sentence5*5
dense_shapes%
#:�%:�%:�%:�%:�%*
sparse_types
 *
sparse_keys
 "_
'parsesingleexample_parsesingleexample_24ParseSingleExample/ParseSingleExample:dense_values:3"_
'parsesingleexample_parsesingleexample_34ParseSingleExample/ParseSingleExample:dense_values:4"]
%parsesingleexample_parsesingleexample4ParseSingleExample/ParseSingleExample:dense_values:0"_
'parsesingleexample_parsesingleexample_04ParseSingleExample/ParseSingleExample:dense_values:1"_
'parsesingleexample_parsesingleexample_14ParseSingleExample/ParseSingleExample:dense_values:2
�
�
,Dataset_map_split_skip_thoughts_sentences_48
arg0
arg1
arg2
arg3
arg4
packed_6
packed_72DWrapper for passing nested structures to and from tf.data functions.D
packedPackarg0arg1arg2arg3*
T0*

axis *
N4
packed_1Packarg4*
T0*

axis *
NF
packed_2Packarg0arg1arg2arg3*
T0*

axis *
NF
packed_3Packarg0arg1arg2arg3*
T0*

axis *
N4
packed_4Packarg4*
T0*

axis *
N4
packed_5Packarg4*
N*
T0*

axis H

packed_6_0Packarg0arg1arg2arg3*
N*
T0*

axis 6

packed_7_0Packarg4*
T0*

axis *
N"
packed_7packed_7_0:output:0"
packed_6packed_6_0:output:0"3Et�kh     �]fE	�Ӣ�`:�AJ��
�5�5
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
�
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
s
	AssignAdd
ref"T�

value"T

output_ref"T�" 
Ttype:
2	"
use_lockingbool( 
�
BatchDatasetV2
input_dataset

batch_size	
drop_remainder


handle"
output_types
list(type)(0" 
output_shapeslist(shape)(0
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	
�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
�
FlatMapDataset
input_dataset
other_arguments2
Targuments

handle"	
ffunc"

Targuments
list(type)("
output_types
list(type)(0" 
output_shapeslist(shape)(0
,
Floor
x"T
y"T"
Ttype:
2
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
�
IteratorFromStringHandleV2
string_handle
resource_handle" 
output_types
list(type)
 (""
output_shapeslist(shape)
 (�
�
IteratorGetNext
iterator

components2output_types"
output_types
list(type)(0" 
output_shapeslist(shape)(0�
C
IteratorToStringHandle
resource_handle
string_handle�
�

IteratorV2

handle"
shared_namestring"
	containerstring"
output_types
list(type)(0" 
output_shapeslist(shape)(0�
,
MakeIterator
dataset
iterator�
�

MapDataset
input_dataset
other_arguments2
Targuments

handle"	
ffunc"

Targuments
list(type)("
output_types
list(type)(0" 
output_shapeslist(shape)(0"$
use_inter_op_parallelismbool(" 
preserve_cardinalitybool( 
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	�
�
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
8
MergeSummary
inputs*N
summary"
Nint(0
o
ModelDataset
input_dataset

handle"
output_types
list(type)(0" 
output_shapeslist(shape)(0
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
�
OptimizeDataset
input_dataset
optimizations

handle"
output_types
list(type)(0" 
output_shapeslist(shape)(0
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
ParallelMapDataset
input_dataset
other_arguments2
Targuments
num_parallel_calls

handle"	
ffunc"

Targuments
list(type)("
output_types
list(type)(0" 
output_shapeslist(shape)(0"$
use_inter_op_parallelismbool("
sloppybool( " 
preserve_cardinalitybool( 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
L
PreventGradient

input"T
output"T"	
Ttype"
messagestring 
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
{
RepeatDataset
input_dataset	
count	

handle"
output_types
list(type)(0" 
output_shapeslist(shape)(0
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
0
Round
x"T
y"T"
Ttype:

2	
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
�
ShuffleDataset
input_dataset
buffer_size	
seed		
seed2	

handle"$
reshuffle_each_iterationbool("
output_types
list(type)(0" 
output_shapeslist(shape)(0
0
Sigmoid
x"T
y"T"
Ttype:

2
=
SigmoidGrad
y"T
dy"T
z"T"
Ttype:

2
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
�
#SparseSoftmaxCrossEntropyWithLogits
features"T
labels"Tlabels	
loss"T
backprop"T"
Ttype:
2"
Tlabelstype0	:
2	
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
-
Sqrt
x"T
y"T"
Ttype:

2
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
�
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
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
�
TensorSliceDataset

components2Toutput_types

handle"
Toutput_types
list(type)(0" 
output_shapeslist(shape)(0�
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
Unpack

value"T
output"T*num"
numint("	
Ttype"
axisint 
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype


ZipDataset
input_datasets*N

handle"
output_types
list(type)(0" 
output_shapeslist(shape)(0"
Nint(0*1.13.12
b'unknown'��
�
ConstConst*�
value�B~ Bx/Users/arthur/Documents/natural_lanugage_understanding/nlu_project2/data/processed/train_stories_skip_thoughts.tfrecords*
dtype0*
_output_shapes
: 
g
flat_filenames/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
i
flat_filenamesReshapeConstflat_filenames/shape*
_output_shapes
:*
T0*
Tshape0
v
PlaceholderPlaceholder*
dtype0*+
_output_shapes
:���������* 
shape:���������
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:���������*
shape:���������
N
Placeholder_2Placeholder*
dtype0*
_output_shapes
: *
shape: 
�
Const_1Const*�
value�B~ Bx/Users/arthur/Documents/natural_lanugage_understanding/nlu_project2/data/processed/train_stories_skip_thoughts.tfrecords*
dtype0*
_output_shapes
: 
i
flat_filenames_1/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
o
flat_filenames_1ReshapeConst_1flat_filenames_1/shape*
_output_shapes
:*
T0*
Tshape0
T
num_parallel_callsConst*
value	B :*
dtype0*
_output_shapes
: 
V
num_parallel_calls_1Const*
value	B :*
dtype0*
_output_shapes
: 
H
countConst*
dtype0	*
_output_shapes
: *
value
B	 R�'
N
buffer_sizeConst*
value
B	 R�'*
dtype0	*
_output_shapes
: 
F
seedConst*
value	B	 R *
dtype0	*
_output_shapes
: 
G
seed2Const*
value	B	 R *
dtype0	*
_output_shapes
: 
M

batch_sizeConst*
dtype0	*
_output_shapes
: *
value
B	 R�
P
drop_remainderConst*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
Const_2Const*�
valueB} Bw/Users/arthur/Documents/natural_lanugage_understanding/nlu_project2/data/processed/eval_stories_skip_thoughts.tfrecords*
dtype0*
_output_shapes
: 
i
flat_filenames_2/shapeConst*
valueB:
���������*
dtype0*
_output_shapes
:
o
flat_filenames_2ReshapeConst_2flat_filenames_2/shape*
T0*
Tshape0*
_output_shapes
:
V
num_parallel_calls_2Const*
value	B :*
dtype0*
_output_shapes
: 
P
buffer_size_1Const*
value
B	 R�'*
dtype0	*
_output_shapes
: 
H
seed_1Const*
value	B	 R *
dtype0	*
_output_shapes
: 
I
seed2_1Const*
value	B	 R *
dtype0	*
_output_shapes
: 
J
count_1Const*
value
B	 R�*
dtype0	*
_output_shapes
: 
O
batch_size_1Const*
value
B	 R�*
dtype0	*
_output_shapes
: 
R
drop_remainder_1Const*
value	B
 Z*
dtype0
*
_output_shapes
: 
�
optimizationsConst*
dtype0*
_output_shapes
:*V
valueMBKBmap_and_batch_fusionBnoop_eliminationBshuffle_and_repeat_fusion
�

IteratorV2
IteratorV2*
output_types
2*
shared_name **
output_shapes
:��%:�*
	container *
_output_shapes
: 
�
TensorSliceDatasetTensorSliceDatasetflat_filenames_1*
output_shapes
: *
_class
loc:@IteratorV2*
Toutput_types
2*
_output_shapes
: 
�
FlatMapDatasetFlatMapDatasetTensorSliceDataset*
_output_shapes
: *
output_shapes
: *
_class
loc:@IteratorV2**
f%R#
!Dataset_flat_map_read_one_file_29*
output_types
2*

Targuments
 
�

MapDataset
MapDatasetFlatMapDataset*

Targuments
 *
_output_shapes
: *
preserve_cardinality( *6
output_shapes%
#:�%:�%:�%:�%:�%*
_class
loc:@IteratorV2*"
fR
Dataset_map_extract_fn_35*
output_types	
2*
use_inter_op_parallelism(
�
ParallelMapDatasetParallelMapDataset
MapDatasetnum_parallel_calls*
preserve_cardinality( *
_output_shapes
: *)
output_shapes
:	�%:	�%*
_class
loc:@IteratorV2*5
f0R.
,Dataset_map_split_skip_thoughts_sentences_48*
sloppy( *
use_inter_op_parallelism(*
output_types
2*

Targuments
 
�
ParallelMapDataset_1ParallelMapDatasetParallelMapDatasetnum_parallel_calls_1*
preserve_cardinality( *
_output_shapes
: * 
output_shapes
:	�%: *
_class
loc:@IteratorV2*3
f.R,
*Dataset_map_<class 'functools.partial'>_64*
sloppy( *
use_inter_op_parallelism(*
output_types
2*

Targuments
 
�
RepeatDatasetRepeatDatasetParallelMapDataset_1count* 
output_shapes
:	�%: *
_class
loc:@IteratorV2*
_output_shapes
: *
output_types
2
�
ShuffleDatasetShuffleDatasetRepeatDatasetbuffer_sizeseedseed2* 
output_shapes
:	�%: *
_class
loc:@IteratorV2*
reshuffle_each_iteration(*
_output_shapes
: *
output_types
2
�
BatchDatasetV2BatchDatasetV2ShuffleDataset
batch_sizedrop_remainder*
_output_shapes
: *
output_types
2**
output_shapes
:��%:�*
_class
loc:@IteratorV2
�
OptimizeDatasetOptimizeDatasetBatchDatasetV2optimizations*
_output_shapes
: *
output_types
2**
output_shapes
:��%:�*
_class
loc:@IteratorV2
�
ModelDatasetModelDatasetOptimizeDataset**
output_shapes
:��%:�*
_class
loc:@IteratorV2*
_output_shapes
: *
output_types
2
U
MakeIteratorMakeIteratorModelDataset
IteratorV2*
_class
loc:@IteratorV2
T
IteratorToStringHandleIteratorToStringHandle
IteratorV2*
_output_shapes
: 
�
optimizations_1Const*V
valueMBKBmap_and_batch_fusionBnoop_eliminationBshuffle_and_repeat_fusion*
dtype0*
_output_shapes
:
�
IteratorV2_1
IteratorV2**
output_shapes
:��%:�*
	container *
_output_shapes
: *
output_types
2*
shared_name 
�
TensorSliceDataset_1TensorSliceDatasetflat_filenames_2*
output_shapes
: *
_class
loc:@IteratorV2_1*
Toutput_types
2*
_output_shapes
: 
�
FlatMapDataset_1FlatMapDatasetTensorSliceDataset_1*
output_types
2*

Targuments
 *
_output_shapes
: *
output_shapes
: *
_class
loc:@IteratorV2_1*+
f&R$
"Dataset_flat_map_read_one_file_151
�
MapDataset_1
MapDatasetFlatMapDataset_1*
preserve_cardinality( *
_output_shapes
: *=
output_shapes,
*:�%:�%:�%:�%:�%:�%*
_class
loc:@IteratorV2_1*#
fR
Dataset_map_extract_fn_157*
use_inter_op_parallelism(*
output_types

2*

Targuments
 
�
ParallelMapDataset_2ParallelMapDatasetMapDataset_1num_parallel_calls_2*
output_shapes
:	�%*
_class
loc:@IteratorV2_1*'
f"R 
Dataset_map_tensorize_dict_172*
sloppy( *
output_types
2*
use_inter_op_parallelism(*

Targuments
 *
_output_shapes
: *
preserve_cardinality( 
�
TensorSliceDataset_2TensorSliceDatasetPlaceholder_1*
output_shapes
: *
_class
loc:@IteratorV2_1*
Toutput_types
2*
_output_shapes
: 
�

ZipDataset
ZipDatasetParallelMapDataset_2TensorSliceDataset_2*
output_types
2* 
output_shapes
:	�%: *
_class
loc:@IteratorV2_1*
N*
_output_shapes
: 
�
ShuffleDataset_1ShuffleDataset
ZipDatasetbuffer_size_1seed_1seed2_1* 
output_shapes
:	�%: *
_class
loc:@IteratorV2_1*
reshuffle_each_iteration(*
_output_shapes
: *
output_types
2
�
RepeatDataset_1RepeatDatasetShuffleDataset_1count_1* 
output_shapes
:	�%: *
_class
loc:@IteratorV2_1*
_output_shapes
: *
output_types
2
�
BatchDatasetV2_1BatchDatasetV2RepeatDataset_1batch_size_1drop_remainder_1**
output_shapes
:��%:�*
_class
loc:@IteratorV2_1*
_output_shapes
: *
output_types
2
�
OptimizeDataset_1OptimizeDatasetBatchDatasetV2_1optimizations_1*
output_types
2**
output_shapes
:��%:�*
_class
loc:@IteratorV2_1*
_output_shapes
: 
�
ModelDataset_1ModelDatasetOptimizeDataset_1*
_output_shapes
: *
output_types
2**
output_shapes
:��%:�*
_class
loc:@IteratorV2_1
]
MakeIterator_1MakeIteratorModelDataset_1IteratorV2_1*
_class
loc:@IteratorV2_1
X
IteratorToStringHandle_1IteratorToStringHandleIteratorV2_1*
_output_shapes
: 
�
IteratorFromStringHandleV2IteratorFromStringHandleV2Placeholder_2"/device:CPU:0**
output_shapes
:��%:�*
_output_shapes
: *
output_types
2
f
IteratorToStringHandle_2IteratorToStringHandleIteratorFromStringHandleV2*
_output_shapes
: 
�
IteratorGetNextIteratorGetNextIteratorFromStringHandleV2*
output_types
2**
output_shapes
:��%:�*+
_output_shapes
:��%:�
]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_sliceStridedSliceIteratorGetNextstrided_slice/stackstrided_slice/stack_1strided_slice/stack_2*
T0*
Index0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:	�%
_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
strided_slice_1StridedSliceIteratorGetNext:1strided_slice_1/stackstrided_slice_1/stack_1strided_slice_1/stack_2*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
_output_shapes
: *
T0*
Index0*
shrink_axis_mask
�
TensorSliceDataset_3TensorSliceDatasetflat_filenames_1"/device:CPU:0*
output_shapes
: *-
_class#
!loc:@IteratorFromStringHandleV2*
Toutput_types
2*
_output_shapes
: 
�
FlatMapDataset_2FlatMapDatasetTensorSliceDataset_3"/device:CPU:0*
output_shapes
: *-
_class#
!loc:@IteratorFromStringHandleV2**
f%R#
!Dataset_flat_map_read_one_file_29*
output_types
2*

Targuments
 *
_output_shapes
: 
�
MapDataset_2
MapDatasetFlatMapDataset_2"/device:CPU:0*
preserve_cardinality( *
_output_shapes
: *6
output_shapes%
#:�%:�%:�%:�%:�%*-
_class#
!loc:@IteratorFromStringHandleV2*"
fR
Dataset_map_extract_fn_35*
use_inter_op_parallelism(*
output_types	
2*

Targuments
 
�
ParallelMapDataset_3ParallelMapDatasetMapDataset_2num_parallel_calls"/device:CPU:0*)
output_shapes
:	�%:	�%*-
_class#
!loc:@IteratorFromStringHandleV2*5
f0R.
,Dataset_map_split_skip_thoughts_sentences_48*
sloppy( *
output_types
2*
use_inter_op_parallelism(*

Targuments
 *
preserve_cardinality( *
_output_shapes
: 
�
ParallelMapDataset_4ParallelMapDatasetParallelMapDataset_3num_parallel_calls_1"/device:CPU:0* 
output_shapes
:	�%: *-
_class#
!loc:@IteratorFromStringHandleV2*3
f.R,
*Dataset_map_<class 'functools.partial'>_64*
sloppy( *
output_types
2*
use_inter_op_parallelism(*

Targuments
 *
preserve_cardinality( *
_output_shapes
: 
�
RepeatDataset_2RepeatDatasetParallelMapDataset_4count"/device:CPU:0* 
output_shapes
:	�%: *-
_class#
!loc:@IteratorFromStringHandleV2*
_output_shapes
: *
output_types
2
�
ShuffleDataset_2ShuffleDatasetRepeatDataset_2buffer_sizeseedseed2"/device:CPU:0* 
output_shapes
:	�%: *-
_class#
!loc:@IteratorFromStringHandleV2*
reshuffle_each_iteration(*
_output_shapes
: *
output_types
2
�
BatchDatasetV2_2BatchDatasetV2ShuffleDataset_2
batch_sizedrop_remainder"/device:CPU:0*
_output_shapes
: *
output_types
2**
output_shapes
:��%:�*-
_class#
!loc:@IteratorFromStringHandleV2
�
train_datasetMakeIteratorBatchDatasetV2_2IteratorFromStringHandleV2"/device:CPU:0*-
_class#
!loc:@IteratorFromStringHandleV2
�
TensorSliceDataset_4TensorSliceDatasetflat_filenames_2"/device:CPU:0*
output_shapes
: *-
_class#
!loc:@IteratorFromStringHandleV2*
Toutput_types
2*
_output_shapes
: 
�
FlatMapDataset_3FlatMapDatasetTensorSliceDataset_4"/device:CPU:0*
output_types
2*

Targuments
 *
_output_shapes
: *
output_shapes
: *-
_class#
!loc:@IteratorFromStringHandleV2*+
f&R$
"Dataset_flat_map_read_one_file_151
�
MapDataset_3
MapDatasetFlatMapDataset_3"/device:CPU:0*
_output_shapes
: *
preserve_cardinality( *=
output_shapes,
*:�%:�%:�%:�%:�%:�%*-
_class#
!loc:@IteratorFromStringHandleV2*#
fR
Dataset_map_extract_fn_157*
use_inter_op_parallelism(*
output_types

2*

Targuments
 
�
ParallelMapDataset_5ParallelMapDatasetMapDataset_3num_parallel_calls_2"/device:CPU:0*

Targuments
 *
_output_shapes
: *
preserve_cardinality( *
output_shapes
:	�%*-
_class#
!loc:@IteratorFromStringHandleV2*'
f"R 
Dataset_map_tensorize_dict_172*
sloppy( *
use_inter_op_parallelism(*
output_types
2
�
TensorSliceDataset_5TensorSliceDatasetPlaceholder_1"/device:CPU:0*
output_shapes
: *-
_class#
!loc:@IteratorFromStringHandleV2*
Toutput_types
2*
_output_shapes
: 
�
ZipDataset_1
ZipDatasetParallelMapDataset_5TensorSliceDataset_5"/device:CPU:0* 
output_shapes
:	�%: *-
_class#
!loc:@IteratorFromStringHandleV2*
N*
_output_shapes
: *
output_types
2
�
ShuffleDataset_3ShuffleDatasetZipDataset_1buffer_size_1seed_1seed2_1"/device:CPU:0*
reshuffle_each_iteration(*
_output_shapes
: *
output_types
2* 
output_shapes
:	�%: *-
_class#
!loc:@IteratorFromStringHandleV2
�
RepeatDataset_3RepeatDatasetShuffleDataset_3count_1"/device:CPU:0* 
output_shapes
:	�%: *-
_class#
!loc:@IteratorFromStringHandleV2*
_output_shapes
: *
output_types
2
�
BatchDatasetV2_3BatchDatasetV2RepeatDataset_3batch_size_1drop_remainder_1"/device:CPU:0**
output_shapes
:��%:�*-
_class#
!loc:@IteratorFromStringHandleV2*
_output_shapes
: *
output_types
2
�
test_datasetMakeIteratorBatchDatasetV2_3IteratorFromStringHandleV2"/device:CPU:0*-
_class#
!loc:@IteratorFromStringHandleV2
v
!split_endings/strided_slice/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
x
#split_endings/strided_slice/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
x
#split_endings/strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
�
split_endings/strided_sliceStridedSliceIteratorGetNext!split_endings/strided_slice/stack#split_endings/strided_slice/stack_1#split_endings/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*$
_output_shapes
:��%*
Index0*
T0
x
#split_endings/strided_slice_1/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
z
%split_endings/strided_slice_1/stack_1Const*!
valueB"            *
dtype0*
_output_shapes
:
z
%split_endings/strided_slice_1/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
�
split_endings/strided_slice_1StridedSliceIteratorGetNext#split_endings/strided_slice_1/stack%split_endings/strided_slice_1/stack_1%split_endings/strided_slice_1/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*$
_output_shapes
:��%*
T0*
Index0
|
'ending/sentence_rnn/strided_slice/stackConst*!
valueB"            *
dtype0*
_output_shapes
:
~
)ending/sentence_rnn/strided_slice/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
~
)ending/sentence_rnn/strided_slice/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
�
!ending/sentence_rnn/strided_sliceStridedSlicesplit_endings/strided_slice_1'ending/sentence_rnn/strided_slice/stack)ending/sentence_rnn/strided_slice/stack_1)ending/sentence_rnn/strided_slice/stack_2*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*$
_output_shapes
:��%*
T0*
Index0
a
ending/sentence_rnn/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
ending/sentence_rnn/concatConcatV2split_endings/strided_slice!ending/sentence_rnn/strided_sliceending/sentence_rnn/concat/axis*

Tidx0*
T0*
N*$
_output_shapes
:��%
�
ending/sentence_rnn/unstackUnpackending/sentence_rnn/concat*P
_output_shapes>
<:
��%:
��%:
��%:
��%:
��%*
T0*	
num*

axis
z
/ending/sentence_rnn/rnn/LSTMCellZeroState/ConstConst*
valueB:�*
dtype0*
_output_shapes
:
|
1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_1Const*
dtype0*
_output_shapes
:*
valueB:�
w
5ending/sentence_rnn/rnn/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
0ending/sentence_rnn/rnn/LSTMCellZeroState/concatConcatV2/ending/sentence_rnn/rnn/LSTMCellZeroState/Const1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_15ending/sentence_rnn/rnn/LSTMCellZeroState/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
z
5ending/sentence_rnn/rnn/LSTMCellZeroState/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/ending/sentence_rnn/rnn/LSTMCellZeroState/zerosFill0ending/sentence_rnn/rnn/LSTMCellZeroState/concat5ending/sentence_rnn/rnn/LSTMCellZeroState/zeros/Const*
T0*

index_type0* 
_output_shapes
:
��
|
1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:
|
1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_3Const*
valueB:�*
dtype0*
_output_shapes
:
|
1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_4Const*
valueB:�*
dtype0*
_output_shapes
:
|
1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_5Const*
valueB:�*
dtype0*
_output_shapes
:
y
7ending/sentence_rnn/rnn/LSTMCellZeroState/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
�
2ending/sentence_rnn/rnn/LSTMCellZeroState/concat_1ConcatV21ending/sentence_rnn/rnn/LSTMCellZeroState/Const_41ending/sentence_rnn/rnn/LSTMCellZeroState/Const_57ending/sentence_rnn/rnn/LSTMCellZeroState/concat_1/axis*
T0*
N*
_output_shapes
:*

Tidx0
|
7ending/sentence_rnn/rnn/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
1ending/sentence_rnn/rnn/LSTMCellZeroState/zeros_1Fill2ending/sentence_rnn/rnn/LSTMCellZeroState/concat_17ending/sentence_rnn/rnn/LSTMCellZeroState/zeros_1/Const* 
_output_shapes
:
��*
T0*

index_type0
|
1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_6Const*
valueB:�*
dtype0*
_output_shapes
:
|
1ending/sentence_rnn/rnn/LSTMCellZeroState/Const_7Const*
valueB:�*
dtype0*
_output_shapes
:
�
@ending/rnn/sentence_cell/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
valueB"�  �  
�
>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/minConst*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
valueB
 *�ʼ*
dtype0*
_output_shapes
: 
�
>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
valueB
 *��<
�
Hending/rnn/sentence_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniform@ending/rnn/sentence_cell/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
�-�*

seed**
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
seed2x
�
>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/subSub>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/max>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel
�
>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/mulMulHending/rnn/sentence_cell/kernel/Initializer/random_uniform/RandomUniform>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel* 
_output_shapes
:
�-�
�
:ending/rnn/sentence_cell/kernel/Initializer/random_uniformAdd>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/mul>ending/rnn/sentence_cell/kernel/Initializer/random_uniform/min* 
_output_shapes
:
�-�*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel
�
ending/rnn/sentence_cell/kernel
VariableV2*
	container *
shape:
�-�*
dtype0* 
_output_shapes
:
�-�*
shared_name *2
_class(
&$loc:@ending/rnn/sentence_cell/kernel
�
&ending/rnn/sentence_cell/kernel/AssignAssignending/rnn/sentence_cell/kernel:ending/rnn/sentence_cell/kernel/Initializer/random_uniform*
use_locking(*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
validate_shape(* 
_output_shapes
:
�-�
|
$ending/rnn/sentence_cell/kernel/readIdentityending/rnn/sentence_cell/kernel*
T0* 
_output_shapes
:
�-�
�
?ending/rnn/sentence_cell/bias/Initializer/zeros/shape_as_tensorConst*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
valueB:�*
dtype0*
_output_shapes
:
�
5ending/rnn/sentence_cell/bias/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
valueB
 *    
�
/ending/rnn/sentence_cell/bias/Initializer/zerosFill?ending/rnn/sentence_cell/bias/Initializer/zeros/shape_as_tensor5ending/rnn/sentence_cell/bias/Initializer/zeros/Const*
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*

index_type0*
_output_shapes	
:�
�
ending/rnn/sentence_cell/bias
VariableV2*
shared_name *0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
$ending/rnn/sentence_cell/bias/AssignAssignending/rnn/sentence_cell/bias/ending/rnn/sentence_cell/bias/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
validate_shape(*
_output_shapes	
:�
s
"ending/rnn/sentence_cell/bias/readIdentityending/rnn/sentence_cell/bias*
T0*
_output_shapes	
:�
s
1ending/sentence_rnn/rnn/sentence_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
,ending/sentence_rnn/rnn/sentence_cell/concatConcatV2ending/sentence_rnn/unstack1ending/sentence_rnn/rnn/LSTMCellZeroState/zeros_11ending/sentence_rnn/rnn/sentence_cell/concat/axis*

Tidx0*
T0*
N* 
_output_shapes
:
��-
�
,ending/sentence_rnn/rnn/sentence_cell/MatMulMatMul,ending/sentence_rnn/rnn/sentence_cell/concat$ending/rnn/sentence_cell/kernel/read*
transpose_a( * 
_output_shapes
:
��*
transpose_b( *
T0
�
-ending/sentence_rnn/rnn/sentence_cell/BiasAddBiasAdd,ending/sentence_rnn/rnn/sentence_cell/MatMul"ending/rnn/sentence_cell/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
��
m
+ending/sentence_rnn/rnn/sentence_cell/ConstConst*
dtype0*
_output_shapes
: *
value	B :
w
5ending/sentence_rnn/rnn/sentence_cell/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
+ending/sentence_rnn/rnn/sentence_cell/splitSplit5ending/sentence_rnn/rnn/sentence_cell/split/split_dim-ending/sentence_rnn/rnn/sentence_cell/BiasAdd*
T0*
	num_split*D
_output_shapes2
0:
��:
��:
��:
��
p
+ending/sentence_rnn/rnn/sentence_cell/add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
)ending/sentence_rnn/rnn/sentence_cell/addAdd-ending/sentence_rnn/rnn/sentence_cell/split:2+ending/sentence_rnn/rnn/sentence_cell/add/y*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn/sentence_cell/SigmoidSigmoid)ending/sentence_rnn/rnn/sentence_cell/add*
T0* 
_output_shapes
:
��
�
)ending/sentence_rnn/rnn/sentence_cell/mulMul-ending/sentence_rnn/rnn/sentence_cell/Sigmoid/ending/sentence_rnn/rnn/LSTMCellZeroState/zeros*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_1Sigmoid+ending/sentence_rnn/rnn/sentence_cell/split* 
_output_shapes
:
��*
T0
�
*ending/sentence_rnn/rnn/sentence_cell/TanhTanh-ending/sentence_rnn/rnn/sentence_cell/split:1*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_1Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_1*ending/sentence_rnn/rnn/sentence_cell/Tanh* 
_output_shapes
:
��*
T0
�
+ending/sentence_rnn/rnn/sentence_cell/add_1Add)ending/sentence_rnn/rnn/sentence_cell/mul+ending/sentence_rnn/rnn/sentence_cell/mul_1* 
_output_shapes
:
��*
T0
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_2Sigmoid-ending/sentence_rnn/rnn/sentence_cell/split:3* 
_output_shapes
:
��*
T0
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_1Tanh+ending/sentence_rnn/rnn/sentence_cell/add_1*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_2Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_2,ending/sentence_rnn/rnn/sentence_cell/Tanh_1*
T0* 
_output_shapes
:
��
u
3ending/sentence_rnn/rnn/sentence_cell/concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
.ending/sentence_rnn/rnn/sentence_cell/concat_1ConcatV2ending/sentence_rnn/unstack:1+ending/sentence_rnn/rnn/sentence_cell/mul_23ending/sentence_rnn/rnn/sentence_cell/concat_1/axis*

Tidx0*
T0*
N* 
_output_shapes
:
��-
�
.ending/sentence_rnn/rnn/sentence_cell/MatMul_1MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_1$ending/rnn/sentence_cell/kernel/read*
transpose_b( *
T0*
transpose_a( * 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1BiasAdd.ending/sentence_rnn/rnn/sentence_cell/MatMul_1"ending/rnn/sentence_cell/bias/read*
data_formatNHWC* 
_output_shapes
:
��*
T0
o
-ending/sentence_rnn/rnn/sentence_cell/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
y
7ending/sentence_rnn/rnn/sentence_cell/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn/sentence_cell/split_1Split7ending/sentence_rnn/rnn/sentence_cell/split_1/split_dim/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1*
T0*
	num_split*D
_output_shapes2
0:
��:
��:
��:
��
r
-ending/sentence_rnn/rnn/sentence_cell/add_2/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+ending/sentence_rnn/rnn/sentence_cell/add_2Add/ending/sentence_rnn/rnn/sentence_cell/split_1:2-ending/sentence_rnn/rnn/sentence_cell/add_2/y*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_3Sigmoid+ending/sentence_rnn/rnn/sentence_cell/add_2*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_3Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_3+ending/sentence_rnn/rnn/sentence_cell/add_1*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_4Sigmoid-ending/sentence_rnn/rnn/sentence_cell/split_1*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_2Tanh/ending/sentence_rnn/rnn/sentence_cell/split_1:1*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_4Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_4,ending/sentence_rnn/rnn/sentence_cell/Tanh_2* 
_output_shapes
:
��*
T0
�
+ending/sentence_rnn/rnn/sentence_cell/add_3Add+ending/sentence_rnn/rnn/sentence_cell/mul_3+ending/sentence_rnn/rnn/sentence_cell/mul_4*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_5Sigmoid/ending/sentence_rnn/rnn/sentence_cell/split_1:3* 
_output_shapes
:
��*
T0
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_3Tanh+ending/sentence_rnn/rnn/sentence_cell/add_3*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_5Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_5,ending/sentence_rnn/rnn/sentence_cell/Tanh_3*
T0* 
_output_shapes
:
��
u
3ending/sentence_rnn/rnn/sentence_cell/concat_2/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
.ending/sentence_rnn/rnn/sentence_cell/concat_2ConcatV2ending/sentence_rnn/unstack:2+ending/sentence_rnn/rnn/sentence_cell/mul_53ending/sentence_rnn/rnn/sentence_cell/concat_2/axis*

Tidx0*
T0*
N* 
_output_shapes
:
��-
�
.ending/sentence_rnn/rnn/sentence_cell/MatMul_2MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_2$ending/rnn/sentence_cell/kernel/read*
T0*
transpose_a( * 
_output_shapes
:
��*
transpose_b( 
�
/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2BiasAdd.ending/sentence_rnn/rnn/sentence_cell/MatMul_2"ending/rnn/sentence_cell/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
��
o
-ending/sentence_rnn/rnn/sentence_cell/Const_2Const*
value	B :*
dtype0*
_output_shapes
: 
y
7ending/sentence_rnn/rnn/sentence_cell/split_2/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn/sentence_cell/split_2Split7ending/sentence_rnn/rnn/sentence_cell/split_2/split_dim/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2*
T0*
	num_split*D
_output_shapes2
0:
��:
��:
��:
��
r
-ending/sentence_rnn/rnn/sentence_cell/add_4/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+ending/sentence_rnn/rnn/sentence_cell/add_4Add/ending/sentence_rnn/rnn/sentence_cell/split_2:2-ending/sentence_rnn/rnn/sentence_cell/add_4/y* 
_output_shapes
:
��*
T0
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_6Sigmoid+ending/sentence_rnn/rnn/sentence_cell/add_4*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_6Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_6+ending/sentence_rnn/rnn/sentence_cell/add_3*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_7Sigmoid-ending/sentence_rnn/rnn/sentence_cell/split_2*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_4Tanh/ending/sentence_rnn/rnn/sentence_cell/split_2:1* 
_output_shapes
:
��*
T0
�
+ending/sentence_rnn/rnn/sentence_cell/mul_7Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_7,ending/sentence_rnn/rnn/sentence_cell/Tanh_4*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/add_5Add+ending/sentence_rnn/rnn/sentence_cell/mul_6+ending/sentence_rnn/rnn/sentence_cell/mul_7*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_8Sigmoid/ending/sentence_rnn/rnn/sentence_cell/split_2:3*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_5Tanh+ending/sentence_rnn/rnn/sentence_cell/add_5*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_8Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_8,ending/sentence_rnn/rnn/sentence_cell/Tanh_5*
T0* 
_output_shapes
:
��
u
3ending/sentence_rnn/rnn/sentence_cell/concat_3/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
.ending/sentence_rnn/rnn/sentence_cell/concat_3ConcatV2ending/sentence_rnn/unstack:3+ending/sentence_rnn/rnn/sentence_cell/mul_83ending/sentence_rnn/rnn/sentence_cell/concat_3/axis*
T0*
N* 
_output_shapes
:
��-*

Tidx0
�
.ending/sentence_rnn/rnn/sentence_cell/MatMul_3MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_3$ending/rnn/sentence_cell/kernel/read*
transpose_b( *
T0*
transpose_a( * 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3BiasAdd.ending/sentence_rnn/rnn/sentence_cell/MatMul_3"ending/rnn/sentence_cell/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
��
o
-ending/sentence_rnn/rnn/sentence_cell/Const_3Const*
value	B :*
dtype0*
_output_shapes
: 
y
7ending/sentence_rnn/rnn/sentence_cell/split_3/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn/sentence_cell/split_3Split7ending/sentence_rnn/rnn/sentence_cell/split_3/split_dim/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3*
T0*
	num_split*D
_output_shapes2
0:
��:
��:
��:
��
r
-ending/sentence_rnn/rnn/sentence_cell/add_6/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+ending/sentence_rnn/rnn/sentence_cell/add_6Add/ending/sentence_rnn/rnn/sentence_cell/split_3:2-ending/sentence_rnn/rnn/sentence_cell/add_6/y* 
_output_shapes
:
��*
T0
�
/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_9Sigmoid+ending/sentence_rnn/rnn/sentence_cell/add_6*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/mul_9Mul/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_9+ending/sentence_rnn/rnn/sentence_cell/add_5*
T0* 
_output_shapes
:
��
�
0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_10Sigmoid-ending/sentence_rnn/rnn/sentence_cell/split_3*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_6Tanh/ending/sentence_rnn/rnn/sentence_cell/split_3:1*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/mul_10Mul0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_10,ending/sentence_rnn/rnn/sentence_cell/Tanh_6* 
_output_shapes
:
��*
T0
�
+ending/sentence_rnn/rnn/sentence_cell/add_7Add+ending/sentence_rnn/rnn/sentence_cell/mul_9,ending/sentence_rnn/rnn/sentence_cell/mul_10* 
_output_shapes
:
��*
T0
�
0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_11Sigmoid/ending/sentence_rnn/rnn/sentence_cell/split_3:3*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_7Tanh+ending/sentence_rnn/rnn/sentence_cell/add_7*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/mul_11Mul0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_11,ending/sentence_rnn/rnn/sentence_cell/Tanh_7*
T0* 
_output_shapes
:
��
u
3ending/sentence_rnn/rnn/sentence_cell/concat_4/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
.ending/sentence_rnn/rnn/sentence_cell/concat_4ConcatV2ending/sentence_rnn/unstack:4,ending/sentence_rnn/rnn/sentence_cell/mul_113ending/sentence_rnn/rnn/sentence_cell/concat_4/axis*
T0*
N* 
_output_shapes
:
��-*

Tidx0
�
.ending/sentence_rnn/rnn/sentence_cell/MatMul_4MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_4$ending/rnn/sentence_cell/kernel/read*
T0*
transpose_a( * 
_output_shapes
:
��*
transpose_b( 
�
/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4BiasAdd.ending/sentence_rnn/rnn/sentence_cell/MatMul_4"ending/rnn/sentence_cell/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
��
o
-ending/sentence_rnn/rnn/sentence_cell/Const_4Const*
value	B :*
dtype0*
_output_shapes
: 
y
7ending/sentence_rnn/rnn/sentence_cell/split_4/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn/sentence_cell/split_4Split7ending/sentence_rnn/rnn/sentence_cell/split_4/split_dim/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4*
T0*
	num_split*D
_output_shapes2
0:
��:
��:
��:
��
r
-ending/sentence_rnn/rnn/sentence_cell/add_8/yConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
+ending/sentence_rnn/rnn/sentence_cell/add_8Add/ending/sentence_rnn/rnn/sentence_cell/split_4:2-ending/sentence_rnn/rnn/sentence_cell/add_8/y*
T0* 
_output_shapes
:
��
�
0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_12Sigmoid+ending/sentence_rnn/rnn/sentence_cell/add_8*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/mul_12Mul0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_12+ending/sentence_rnn/rnn/sentence_cell/add_7*
T0* 
_output_shapes
:
��
�
0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_13Sigmoid-ending/sentence_rnn/rnn/sentence_cell/split_4*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_8Tanh/ending/sentence_rnn/rnn/sentence_cell/split_4:1*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/mul_13Mul0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_13,ending/sentence_rnn/rnn/sentence_cell/Tanh_8*
T0* 
_output_shapes
:
��
�
+ending/sentence_rnn/rnn/sentence_cell/add_9Add,ending/sentence_rnn/rnn/sentence_cell/mul_12,ending/sentence_rnn/rnn/sentence_cell/mul_13*
T0* 
_output_shapes
:
��
�
0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_14Sigmoid/ending/sentence_rnn/rnn/sentence_cell/split_4:3*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/Tanh_9Tanh+ending/sentence_rnn/rnn/sentence_cell/add_9*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn/sentence_cell/mul_14Mul0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_14,ending/sentence_rnn/rnn/sentence_cell/Tanh_9*
T0* 
_output_shapes
:
��
d
"ending/sentence_rnn/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
value	B :
�
ending/sentence_rnn/ExpandDims
ExpandDims+ending/sentence_rnn/rnn/sentence_cell/mul_2"ending/sentence_rnn/ExpandDims/dim*

Tdim0*
T0*$
_output_shapes
:��
f
$ending/sentence_rnn/ExpandDims_1/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
 ending/sentence_rnn/ExpandDims_1
ExpandDims+ending/sentence_rnn/rnn/sentence_cell/mul_5$ending/sentence_rnn/ExpandDims_1/dim*

Tdim0*
T0*$
_output_shapes
:��
f
$ending/sentence_rnn/ExpandDims_2/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
 ending/sentence_rnn/ExpandDims_2
ExpandDims+ending/sentence_rnn/rnn/sentence_cell/mul_8$ending/sentence_rnn/ExpandDims_2/dim*

Tdim0*
T0*$
_output_shapes
:��
f
$ending/sentence_rnn/ExpandDims_3/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
 ending/sentence_rnn/ExpandDims_3
ExpandDims,ending/sentence_rnn/rnn/sentence_cell/mul_11$ending/sentence_rnn/ExpandDims_3/dim*
T0*$
_output_shapes
:��*

Tdim0
f
$ending/sentence_rnn/ExpandDims_4/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
 ending/sentence_rnn/ExpandDims_4
ExpandDims,ending/sentence_rnn/rnn/sentence_cell/mul_14$ending/sentence_rnn/ExpandDims_4/dim*

Tdim0*
T0*$
_output_shapes
:��
c
!ending/sentence_rnn/concat_1/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
ending/sentence_rnn/concat_1ConcatV2ending/sentence_rnn/ExpandDims ending/sentence_rnn/ExpandDims_1 ending/sentence_rnn/ExpandDims_2 ending/sentence_rnn/ExpandDims_3 ending/sentence_rnn/ExpandDims_4!ending/sentence_rnn/concat_1/axis*

Tidx0*
T0*
N*$
_output_shapes
:��
i
'ending/sentence_rnn/concat_2/concat_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
ending/sentence_rnn/concat_2Identity+ending/sentence_rnn/rnn/sentence_cell/add_9* 
_output_shapes
:
��*
T0
e
 ending/sentence_rnn/dropout/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
r
!ending/sentence_rnn/dropout/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�   �  
f
!ending/sentence_rnn/dropout/sub/xConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
ending/sentence_rnn/dropout/subSub!ending/sentence_rnn/dropout/sub/x ending/sentence_rnn/dropout/rate*
T0*
_output_shapes
: 
s
.ending/sentence_rnn/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
s
.ending/sentence_rnn/dropout/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
8ending/sentence_rnn/dropout/random_uniform/RandomUniformRandomUniform!ending/sentence_rnn/dropout/Shape*
dtype0*
seed2�* 
_output_shapes
:
��*

seed**
T0
�
.ending/sentence_rnn/dropout/random_uniform/subSub.ending/sentence_rnn/dropout/random_uniform/max.ending/sentence_rnn/dropout/random_uniform/min*
T0*
_output_shapes
: 
�
.ending/sentence_rnn/dropout/random_uniform/mulMul8ending/sentence_rnn/dropout/random_uniform/RandomUniform.ending/sentence_rnn/dropout/random_uniform/sub*
T0* 
_output_shapes
:
��
�
*ending/sentence_rnn/dropout/random_uniformAdd.ending/sentence_rnn/dropout/random_uniform/mul.ending/sentence_rnn/dropout/random_uniform/min*
T0* 
_output_shapes
:
��
�
ending/sentence_rnn/dropout/addAddending/sentence_rnn/dropout/sub*ending/sentence_rnn/dropout/random_uniform*
T0* 
_output_shapes
:
��
v
!ending/sentence_rnn/dropout/FloorFloorending/sentence_rnn/dropout/add* 
_output_shapes
:
��*
T0
�
#ending/sentence_rnn/dropout/truedivRealDivending/sentence_rnn/concat_2ending/sentence_rnn/dropout/sub*
T0* 
_output_shapes
:
��
�
ending/sentence_rnn/dropout/mulMul#ending/sentence_rnn/dropout/truediv!ending/sentence_rnn/dropout/Floor*
T0* 
_output_shapes
:
��
�
5ending/output/kernel/Initializer/random_uniform/shapeConst*'
_class
loc:@ending/output/kernel*
valueB"�     *
dtype0*
_output_shapes
:
�
3ending/output/kernel/Initializer/random_uniform/minConst*'
_class
loc:@ending/output/kernel*
valueB
 *⎞�*
dtype0*
_output_shapes
: 
�
3ending/output/kernel/Initializer/random_uniform/maxConst*'
_class
loc:@ending/output/kernel*
valueB
 *⎞=*
dtype0*
_output_shapes
: 
�
=ending/output/kernel/Initializer/random_uniform/RandomUniformRandomUniform5ending/output/kernel/Initializer/random_uniform/shape*
seed2�*
dtype0*
_output_shapes
:	�*

seed**
T0*'
_class
loc:@ending/output/kernel
�
3ending/output/kernel/Initializer/random_uniform/subSub3ending/output/kernel/Initializer/random_uniform/max3ending/output/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@ending/output/kernel*
_output_shapes
: 
�
3ending/output/kernel/Initializer/random_uniform/mulMul=ending/output/kernel/Initializer/random_uniform/RandomUniform3ending/output/kernel/Initializer/random_uniform/sub*
_output_shapes
:	�*
T0*'
_class
loc:@ending/output/kernel
�
/ending/output/kernel/Initializer/random_uniformAdd3ending/output/kernel/Initializer/random_uniform/mul3ending/output/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@ending/output/kernel*
_output_shapes
:	�
�
ending/output/kernel
VariableV2*
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name *'
_class
loc:@ending/output/kernel*
	container 
�
ending/output/kernel/AssignAssignending/output/kernel/ending/output/kernel/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@ending/output/kernel*
validate_shape(*
_output_shapes
:	�
�
ending/output/kernel/readIdentityending/output/kernel*
T0*'
_class
loc:@ending/output/kernel*
_output_shapes
:	�
�
$ending/output/bias/Initializer/zerosConst*%
_class
loc:@ending/output/bias*
valueB*    *
dtype0*
_output_shapes
:
�
ending/output/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *%
_class
loc:@ending/output/bias
�
ending/output/bias/AssignAssignending/output/bias$ending/output/bias/Initializer/zeros*
T0*%
_class
loc:@ending/output/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
ending/output/bias/readIdentityending/output/bias*
T0*%
_class
loc:@ending/output/bias*
_output_shapes
:
�
$ending/sentence_rnn/fc/output/MatMulMatMulending/sentence_rnn/dropout/mulending/output/kernel/read*
T0*
transpose_a( *
_output_shapes
:	�*
transpose_b( 
�
%ending/sentence_rnn/fc/output/BiasAddBiasAdd$ending/sentence_rnn/fc/output/MatMulending/output/bias/read*
T0*
data_formatNHWC*
_output_shapes
:	�
~
)ending/sentence_rnn/strided_slice_1/stackConst*!
valueB"           *
dtype0*
_output_shapes
:
�
+ending/sentence_rnn/strided_slice_1/stack_1Const*!
valueB"           *
dtype0*
_output_shapes
:
�
+ending/sentence_rnn/strided_slice_1/stack_2Const*!
valueB"         *
dtype0*
_output_shapes
:
�
#ending/sentence_rnn/strided_slice_1StridedSlicesplit_endings/strided_slice_1)ending/sentence_rnn/strided_slice_1/stack+ending/sentence_rnn/strided_slice_1/stack_1+ending/sentence_rnn/strided_slice_1/stack_2*

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask*$
_output_shapes
:��%*
T0*
Index0*
shrink_axis_mask 
c
!ending/sentence_rnn/concat_3/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
ending/sentence_rnn/concat_3ConcatV2split_endings/strided_slice#ending/sentence_rnn/strided_slice_1!ending/sentence_rnn/concat_3/axis*
T0*
N*$
_output_shapes
:��%*

Tidx0
�
ending/sentence_rnn/unstack_1Unpackending/sentence_rnn/concat_3*
T0*	
num*

axis*P
_output_shapes>
<:
��%:
��%:
��%:
��%:
��%
|
1ending/sentence_rnn/rnn_1/LSTMCellZeroState/ConstConst*
dtype0*
_output_shapes
:*
valueB:�
~
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_1Const*
valueB:�*
dtype0*
_output_shapes
:
y
7ending/sentence_rnn/rnn_1/LSTMCellZeroState/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
2ending/sentence_rnn/rnn_1/LSTMCellZeroState/concatConcatV21ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_17ending/sentence_rnn/rnn_1/LSTMCellZeroState/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
|
7ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    
�
1ending/sentence_rnn/rnn_1/LSTMCellZeroState/zerosFill2ending/sentence_rnn/rnn_1/LSTMCellZeroState/concat7ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros/Const*
T0*

index_type0* 
_output_shapes
:
��
~
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_2Const*
valueB:�*
dtype0*
_output_shapes
:
~
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_3Const*
dtype0*
_output_shapes
:*
valueB:�
~
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_4Const*
valueB:�*
dtype0*
_output_shapes
:
~
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_5Const*
valueB:�*
dtype0*
_output_shapes
:
{
9ending/sentence_rnn/rnn_1/LSTMCellZeroState/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
4ending/sentence_rnn/rnn_1/LSTMCellZeroState/concat_1ConcatV23ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_43ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_59ending/sentence_rnn/rnn_1/LSTMCellZeroState/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
~
9ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
�
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros_1Fill4ending/sentence_rnn/rnn_1/LSTMCellZeroState/concat_19ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0* 
_output_shapes
:
��
~
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_6Const*
valueB:�*
dtype0*
_output_shapes
:
~
3ending/sentence_rnn/rnn_1/LSTMCellZeroState/Const_7Const*
valueB:�*
dtype0*
_output_shapes
:
u
3ending/sentence_rnn/rnn_1/sentence_cell/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
.ending/sentence_rnn/rnn_1/sentence_cell/concatConcatV2ending/sentence_rnn/unstack_13ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros_13ending/sentence_rnn/rnn_1/sentence_cell/concat/axis*

Tidx0*
T0*
N* 
_output_shapes
:
��-
�
.ending/sentence_rnn/rnn_1/sentence_cell/MatMulMatMul.ending/sentence_rnn/rnn_1/sentence_cell/concat$ending/rnn/sentence_cell/kernel/read*
transpose_a( * 
_output_shapes
:
��*
transpose_b( *
T0
�
/ending/sentence_rnn/rnn_1/sentence_cell/BiasAddBiasAdd.ending/sentence_rnn/rnn_1/sentence_cell/MatMul"ending/rnn/sentence_cell/bias/read*
data_formatNHWC* 
_output_shapes
:
��*
T0
o
-ending/sentence_rnn/rnn_1/sentence_cell/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
y
7ending/sentence_rnn/rnn_1/sentence_cell/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn_1/sentence_cell/splitSplit7ending/sentence_rnn/rnn_1/sentence_cell/split/split_dim/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd*
T0*
	num_split*D
_output_shapes2
0:
��:
��:
��:
��
r
-ending/sentence_rnn/rnn_1/sentence_cell/add/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
+ending/sentence_rnn/rnn_1/sentence_cell/addAdd/ending/sentence_rnn/rnn_1/sentence_cell/split:2-ending/sentence_rnn/rnn_1/sentence_cell/add/y*
T0* 
_output_shapes
:
��
�
/ending/sentence_rnn/rnn_1/sentence_cell/SigmoidSigmoid+ending/sentence_rnn/rnn_1/sentence_cell/add* 
_output_shapes
:
��*
T0
�
+ending/sentence_rnn/rnn_1/sentence_cell/mulMul/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid1ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros*
T0* 
_output_shapes
:
��
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_1Sigmoid-ending/sentence_rnn/rnn_1/sentence_cell/split*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/rnn_1/sentence_cell/TanhTanh/ending/sentence_rnn/rnn_1/sentence_cell/split:1*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_1Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_1,ending/sentence_rnn/rnn_1/sentence_cell/Tanh*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_1Add+ending/sentence_rnn/rnn_1/sentence_cell/mul-ending/sentence_rnn/rnn_1/sentence_cell/mul_1* 
_output_shapes
:
��*
T0
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_2Sigmoid/ending/sentence_rnn/rnn_1/sentence_cell/split:3* 
_output_shapes
:
��*
T0
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_1Tanh-ending/sentence_rnn/rnn_1/sentence_cell/add_1*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_2Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_2.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_1* 
_output_shapes
:
��*
T0
w
5ending/sentence_rnn/rnn_1/sentence_cell/concat_1/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
0ending/sentence_rnn/rnn_1/sentence_cell/concat_1ConcatV2ending/sentence_rnn/unstack_1:1-ending/sentence_rnn/rnn_1/sentence_cell/mul_25ending/sentence_rnn/rnn_1/sentence_cell/concat_1/axis*
T0*
N* 
_output_shapes
:
��-*

Tidx0
�
0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_1$ending/rnn/sentence_cell/kernel/read*
T0*
transpose_a( * 
_output_shapes
:
��*
transpose_b( 
�
1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1BiasAdd0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1"ending/rnn/sentence_cell/bias/read*
data_formatNHWC* 
_output_shapes
:
��*
T0
q
/ending/sentence_rnn/rnn_1/sentence_cell/Const_1Const*
dtype0*
_output_shapes
: *
value	B :
{
9ending/sentence_rnn/rnn_1/sentence_cell/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
/ending/sentence_rnn/rnn_1/sentence_cell/split_1Split9ending/sentence_rnn/rnn_1/sentence_cell/split_1/split_dim1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1*
T0*
	num_split*D
_output_shapes2
0:
��:
��:
��:
��
t
/ending/sentence_rnn/rnn_1/sentence_cell/add_2/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_2Add1ending/sentence_rnn/rnn_1/sentence_cell/split_1:2/ending/sentence_rnn/rnn_1/sentence_cell/add_2/y*
T0* 
_output_shapes
:
��
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_3Sigmoid-ending/sentence_rnn/rnn_1/sentence_cell/add_2*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_3Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_3-ending/sentence_rnn/rnn_1/sentence_cell/add_1* 
_output_shapes
:
��*
T0
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_4Sigmoid/ending/sentence_rnn/rnn_1/sentence_cell/split_1*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_2Tanh1ending/sentence_rnn/rnn_1/sentence_cell/split_1:1* 
_output_shapes
:
��*
T0
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_4Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_4.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_2* 
_output_shapes
:
��*
T0
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_3Add-ending/sentence_rnn/rnn_1/sentence_cell/mul_3-ending/sentence_rnn/rnn_1/sentence_cell/mul_4*
T0* 
_output_shapes
:
��
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_5Sigmoid1ending/sentence_rnn/rnn_1/sentence_cell/split_1:3*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_3Tanh-ending/sentence_rnn/rnn_1/sentence_cell/add_3* 
_output_shapes
:
��*
T0
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_5Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_5.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_3* 
_output_shapes
:
��*
T0
w
5ending/sentence_rnn/rnn_1/sentence_cell/concat_2/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
0ending/sentence_rnn/rnn_1/sentence_cell/concat_2ConcatV2ending/sentence_rnn/unstack_1:2-ending/sentence_rnn/rnn_1/sentence_cell/mul_55ending/sentence_rnn/rnn_1/sentence_cell/concat_2/axis*
T0*
N* 
_output_shapes
:
��-*

Tidx0
�
0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_2$ending/rnn/sentence_cell/kernel/read*
transpose_a( * 
_output_shapes
:
��*
transpose_b( *
T0
�
1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2BiasAdd0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2"ending/rnn/sentence_cell/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
��
q
/ending/sentence_rnn/rnn_1/sentence_cell/Const_2Const*
value	B :*
dtype0*
_output_shapes
: 
{
9ending/sentence_rnn/rnn_1/sentence_cell/split_2/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
/ending/sentence_rnn/rnn_1/sentence_cell/split_2Split9ending/sentence_rnn/rnn_1/sentence_cell/split_2/split_dim1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2*
T0*
	num_split*D
_output_shapes2
0:
��:
��:
��:
��
t
/ending/sentence_rnn/rnn_1/sentence_cell/add_4/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_4Add1ending/sentence_rnn/rnn_1/sentence_cell/split_2:2/ending/sentence_rnn/rnn_1/sentence_cell/add_4/y*
T0* 
_output_shapes
:
��
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_6Sigmoid-ending/sentence_rnn/rnn_1/sentence_cell/add_4*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_6Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_6-ending/sentence_rnn/rnn_1/sentence_cell/add_3*
T0* 
_output_shapes
:
��
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_7Sigmoid/ending/sentence_rnn/rnn_1/sentence_cell/split_2* 
_output_shapes
:
��*
T0
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_4Tanh1ending/sentence_rnn/rnn_1/sentence_cell/split_2:1* 
_output_shapes
:
��*
T0
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_7Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_7.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_4*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_5Add-ending/sentence_rnn/rnn_1/sentence_cell/mul_6-ending/sentence_rnn/rnn_1/sentence_cell/mul_7* 
_output_shapes
:
��*
T0
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_8Sigmoid1ending/sentence_rnn/rnn_1/sentence_cell/split_2:3*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_5Tanh-ending/sentence_rnn/rnn_1/sentence_cell/add_5* 
_output_shapes
:
��*
T0
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_8Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_8.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_5*
T0* 
_output_shapes
:
��
w
5ending/sentence_rnn/rnn_1/sentence_cell/concat_3/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
0ending/sentence_rnn/rnn_1/sentence_cell/concat_3ConcatV2ending/sentence_rnn/unstack_1:3-ending/sentence_rnn/rnn_1/sentence_cell/mul_85ending/sentence_rnn/rnn_1/sentence_cell/concat_3/axis*
T0*
N* 
_output_shapes
:
��-*

Tidx0
�
0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_3$ending/rnn/sentence_cell/kernel/read*
transpose_b( *
T0*
transpose_a( * 
_output_shapes
:
��
�
1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3BiasAdd0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3"ending/rnn/sentence_cell/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
��
q
/ending/sentence_rnn/rnn_1/sentence_cell/Const_3Const*
value	B :*
dtype0*
_output_shapes
: 
{
9ending/sentence_rnn/rnn_1/sentence_cell/split_3/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
/ending/sentence_rnn/rnn_1/sentence_cell/split_3Split9ending/sentence_rnn/rnn_1/sentence_cell/split_3/split_dim1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3*
T0*
	num_split*D
_output_shapes2
0:
��:
��:
��:
��
t
/ending/sentence_rnn/rnn_1/sentence_cell/add_6/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_6Add1ending/sentence_rnn/rnn_1/sentence_cell/split_3:2/ending/sentence_rnn/rnn_1/sentence_cell/add_6/y*
T0* 
_output_shapes
:
��
�
1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_9Sigmoid-ending/sentence_rnn/rnn_1/sentence_cell/add_6*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/mul_9Mul1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_9-ending/sentence_rnn/rnn_1/sentence_cell/add_5*
T0* 
_output_shapes
:
��
�
2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_10Sigmoid/ending/sentence_rnn/rnn_1/sentence_cell/split_3*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_6Tanh1ending/sentence_rnn/rnn_1/sentence_cell/split_3:1*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/mul_10Mul2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_10.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_6*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_7Add-ending/sentence_rnn/rnn_1/sentence_cell/mul_9.ending/sentence_rnn/rnn_1/sentence_cell/mul_10* 
_output_shapes
:
��*
T0
�
2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_11Sigmoid1ending/sentence_rnn/rnn_1/sentence_cell/split_3:3*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_7Tanh-ending/sentence_rnn/rnn_1/sentence_cell/add_7*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/mul_11Mul2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_11.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_7*
T0* 
_output_shapes
:
��
w
5ending/sentence_rnn/rnn_1/sentence_cell/concat_4/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
0ending/sentence_rnn/rnn_1/sentence_cell/concat_4ConcatV2ending/sentence_rnn/unstack_1:4.ending/sentence_rnn/rnn_1/sentence_cell/mul_115ending/sentence_rnn/rnn_1/sentence_cell/concat_4/axis*
T0*
N* 
_output_shapes
:
��-*

Tidx0
�
0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_4$ending/rnn/sentence_cell/kernel/read*
transpose_b( *
T0*
transpose_a( * 
_output_shapes
:
��
�
1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4BiasAdd0ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4"ending/rnn/sentence_cell/bias/read*
T0*
data_formatNHWC* 
_output_shapes
:
��
q
/ending/sentence_rnn/rnn_1/sentence_cell/Const_4Const*
value	B :*
dtype0*
_output_shapes
: 
{
9ending/sentence_rnn/rnn_1/sentence_cell/split_4/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
�
/ending/sentence_rnn/rnn_1/sentence_cell/split_4Split9ending/sentence_rnn/rnn_1/sentence_cell/split_4/split_dim1ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4*
T0*
	num_split*D
_output_shapes2
0:
��:
��:
��:
��
t
/ending/sentence_rnn/rnn_1/sentence_cell/add_8/yConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_8Add1ending/sentence_rnn/rnn_1/sentence_cell/split_4:2/ending/sentence_rnn/rnn_1/sentence_cell/add_8/y*
T0* 
_output_shapes
:
��
�
2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_12Sigmoid-ending/sentence_rnn/rnn_1/sentence_cell/add_8*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/mul_12Mul2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_12-ending/sentence_rnn/rnn_1/sentence_cell/add_7* 
_output_shapes
:
��*
T0
�
2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_13Sigmoid/ending/sentence_rnn/rnn_1/sentence_cell/split_4*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_8Tanh1ending/sentence_rnn/rnn_1/sentence_cell/split_4:1* 
_output_shapes
:
��*
T0
�
.ending/sentence_rnn/rnn_1/sentence_cell/mul_13Mul2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_13.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_8*
T0* 
_output_shapes
:
��
�
-ending/sentence_rnn/rnn_1/sentence_cell/add_9Add.ending/sentence_rnn/rnn_1/sentence_cell/mul_12.ending/sentence_rnn/rnn_1/sentence_cell/mul_13* 
_output_shapes
:
��*
T0
�
2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_14Sigmoid1ending/sentence_rnn/rnn_1/sentence_cell/split_4:3*
T0* 
_output_shapes
:
��
�
.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_9Tanh-ending/sentence_rnn/rnn_1/sentence_cell/add_9* 
_output_shapes
:
��*
T0
�
.ending/sentence_rnn/rnn_1/sentence_cell/mul_14Mul2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_14.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_9*
T0* 
_output_shapes
:
��
f
$ending/sentence_rnn/ExpandDims_5/dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
 ending/sentence_rnn/ExpandDims_5
ExpandDims-ending/sentence_rnn/rnn_1/sentence_cell/mul_2$ending/sentence_rnn/ExpandDims_5/dim*$
_output_shapes
:��*

Tdim0*
T0
f
$ending/sentence_rnn/ExpandDims_6/dimConst*
dtype0*
_output_shapes
: *
value	B :
�
 ending/sentence_rnn/ExpandDims_6
ExpandDims-ending/sentence_rnn/rnn_1/sentence_cell/mul_5$ending/sentence_rnn/ExpandDims_6/dim*

Tdim0*
T0*$
_output_shapes
:��
f
$ending/sentence_rnn/ExpandDims_7/dimConst*
dtype0*
_output_shapes
: *
value	B :
�
 ending/sentence_rnn/ExpandDims_7
ExpandDims-ending/sentence_rnn/rnn_1/sentence_cell/mul_8$ending/sentence_rnn/ExpandDims_7/dim*

Tdim0*
T0*$
_output_shapes
:��
f
$ending/sentence_rnn/ExpandDims_8/dimConst*
dtype0*
_output_shapes
: *
value	B :
�
 ending/sentence_rnn/ExpandDims_8
ExpandDims.ending/sentence_rnn/rnn_1/sentence_cell/mul_11$ending/sentence_rnn/ExpandDims_8/dim*
T0*$
_output_shapes
:��*

Tdim0
f
$ending/sentence_rnn/ExpandDims_9/dimConst*
dtype0*
_output_shapes
: *
value	B :
�
 ending/sentence_rnn/ExpandDims_9
ExpandDims.ending/sentence_rnn/rnn_1/sentence_cell/mul_14$ending/sentence_rnn/ExpandDims_9/dim*

Tdim0*
T0*$
_output_shapes
:��
c
!ending/sentence_rnn/concat_4/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
ending/sentence_rnn/concat_4ConcatV2 ending/sentence_rnn/ExpandDims_5 ending/sentence_rnn/ExpandDims_6 ending/sentence_rnn/ExpandDims_7 ending/sentence_rnn/ExpandDims_8 ending/sentence_rnn/ExpandDims_9!ending/sentence_rnn/concat_4/axis*
T0*
N*$
_output_shapes
:��*

Tidx0
i
'ending/sentence_rnn/concat_5/concat_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
ending/sentence_rnn/concat_5Identity-ending/sentence_rnn/rnn_1/sentence_cell/add_9*
T0* 
_output_shapes
:
��
g
"ending/sentence_rnn/dropout_1/rateConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
t
#ending/sentence_rnn/dropout_1/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�   �  
h
#ending/sentence_rnn/dropout_1/sub/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
!ending/sentence_rnn/dropout_1/subSub#ending/sentence_rnn/dropout_1/sub/x"ending/sentence_rnn/dropout_1/rate*
T0*
_output_shapes
: 
u
0ending/sentence_rnn/dropout_1/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: 
u
0ending/sentence_rnn/dropout_1/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
:ending/sentence_rnn/dropout_1/random_uniform/RandomUniformRandomUniform#ending/sentence_rnn/dropout_1/Shape*

seed**
T0*
dtype0*
seed2�* 
_output_shapes
:
��
�
0ending/sentence_rnn/dropout_1/random_uniform/subSub0ending/sentence_rnn/dropout_1/random_uniform/max0ending/sentence_rnn/dropout_1/random_uniform/min*
T0*
_output_shapes
: 
�
0ending/sentence_rnn/dropout_1/random_uniform/mulMul:ending/sentence_rnn/dropout_1/random_uniform/RandomUniform0ending/sentence_rnn/dropout_1/random_uniform/sub*
T0* 
_output_shapes
:
��
�
,ending/sentence_rnn/dropout_1/random_uniformAdd0ending/sentence_rnn/dropout_1/random_uniform/mul0ending/sentence_rnn/dropout_1/random_uniform/min*
T0* 
_output_shapes
:
��
�
!ending/sentence_rnn/dropout_1/addAdd!ending/sentence_rnn/dropout_1/sub,ending/sentence_rnn/dropout_1/random_uniform*
T0* 
_output_shapes
:
��
z
#ending/sentence_rnn/dropout_1/FloorFloor!ending/sentence_rnn/dropout_1/add*
T0* 
_output_shapes
:
��
�
%ending/sentence_rnn/dropout_1/truedivRealDivending/sentence_rnn/concat_5!ending/sentence_rnn/dropout_1/sub*
T0* 
_output_shapes
:
��
�
!ending/sentence_rnn/dropout_1/mulMul%ending/sentence_rnn/dropout_1/truediv#ending/sentence_rnn/dropout_1/Floor* 
_output_shapes
:
��*
T0
�
&ending/sentence_rnn/fc_1/output/MatMulMatMul!ending/sentence_rnn/dropout_1/mulending/output/kernel/read*
T0*
transpose_a( *
_output_shapes
:	�*
transpose_b( 
�
'ending/sentence_rnn/fc_1/output/BiasAddBiasAdd&ending/sentence_rnn/fc_1/output/MatMulending/output/bias/read*
T0*
data_formatNHWC*
_output_shapes
:	�
�
ending/sentence_rnn/stackPack%ending/sentence_rnn/fc/output/BiasAdd'ending/sentence_rnn/fc_1/output/BiasAdd*
T0*

axis*
N*#
_output_shapes
:�

eval_predictions/SqueezeSqueezeending/sentence_rnn/stack*
T0*
_output_shapes
:	�*
squeeze_dims

c
!eval_predictions/ArgMax/dimensionConst*
dtype0*
_output_shapes
: *
value	B :
�
eval_predictions/ArgMaxArgMaxeval_predictions/Squeeze!eval_predictions/ArgMax/dimension*
output_type0	*
_output_shapes	
:�*

Tidx0*
T0
~
eval_predictions/ToInt32Casteval_predictions/ArgMax*

SrcT0	*
Truncate( *

DstT0*
_output_shapes	
:�
v
%train_predictions/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
x
'train_predictions/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
x
'train_predictions/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
�
train_predictions/strided_sliceStridedSliceending/sentence_rnn/stack%train_predictions/strided_slice/stack'train_predictions/strided_slice/stack_1'train_predictions/strided_slice/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask*
_output_shapes
:	�*
Index0*
T0*
shrink_axis_mask
�
train_predictions/SqueezeSqueezetrain_predictions/strided_slice*
T0*
_output_shapes	
:�*
squeeze_dims

e
train_predictions/SigmoidSigmoidtrain_predictions/Squeeze*
T0*
_output_shapes	
:�
a
train_predictions/RoundRoundtrain_predictions/Sigmoid*
T0*
_output_shapes	
:�

train_predictions/ToInt32Casttrain_predictions/Round*

SrcT0*
Truncate( *

DstT0*
_output_shapes	
:�
t
)SparseSoftmaxCrossEntropyWithLogits/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
GSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits#SparseSoftmaxCrossEntropyWithLogitseval_predictions/SqueezeIteratorGetNext:1*
Tlabels0*&
_output_shapes
:�:	�*
T0
Q
Const_3Const*
valueB: *
dtype0*
_output_shapes
:
�
MeanMeanGSparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogitsConst_3*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
a
EqualEqualeval_predictions/ToInt32IteratorGetNext:1*
_output_shapes	
:�*
T0
X
CastCastEqual*

SrcT0
*
Truncate( *

DstT0*
_output_shapes	
:�
Q
Const_4Const*
dtype0*
_output_shapes
:*
valueB: 
[
Mean_1MeanCastConst_4*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
[
global_step/initial_valueConst*
value	B : *
dtype0*
_output_shapes
: 
o
global_step
VariableV2*
dtype0*
	container *
_output_shapes
: *
shape: *
shared_name 
�
global_step/AssignAssignglobal_stepglobal_step/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
T0*
_class
loc:@global_step*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  �?
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
_output_shapes
: *
T0*

index_type0
k
!gradients/Mean_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Fill!gradients/Mean_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
d
gradients/Mean_grad/ConstConst*
valueB:�*
dtype0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/Const*

Tmultiples0*
T0*
_output_shapes	
:�
`
gradients/Mean_grad/Const_1Const*
valueB
 *   C*
dtype0*
_output_shapes
: 
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Const_1*
T0*
_output_shapes	
:�
�
gradients/zeros_like	ZerosLikeISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
_output_shapes
:	�*
T0
�
fgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradientPreventGradientISparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits:1*
T0*
_output_shapes
:	�*�
message��Currently there is no way to take the second derivative of sparse_softmax_cross_entropy_with_logits due to the fused implementation's interaction with tf.gradients()
�
egradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dimConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
agradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims
ExpandDimsgradients/Mean_grad/truedivegradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDims/dim*
T0*
_output_shapes
:	�*

Tdim0
�
Zgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mulMulagradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/ExpandDimsfgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/PreventGradient*
T0*
_output_shapes
:	�
�
-gradients/eval_predictions/Squeeze_grad/ShapeConst*!
valueB"�         *
dtype0*
_output_shapes
:
�
/gradients/eval_predictions/Squeeze_grad/ReshapeReshapeZgradients/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits_grad/mul-gradients/eval_predictions/Squeeze_grad/Shape*#
_output_shapes
:�*
T0*
Tshape0
�
0gradients/ending/sentence_rnn/stack_grad/unstackUnpack/gradients/eval_predictions/Squeeze_grad/Reshape*
T0*	
num*

axis**
_output_shapes
:	�:	�
t
9gradients/ending/sentence_rnn/stack_grad/tuple/group_depsNoOp1^gradients/ending/sentence_rnn/stack_grad/unstack
�
Agradients/ending/sentence_rnn/stack_grad/tuple/control_dependencyIdentity0gradients/ending/sentence_rnn/stack_grad/unstack:^gradients/ending/sentence_rnn/stack_grad/tuple/group_deps*
_output_shapes
:	�*
T0*C
_class9
75loc:@gradients/ending/sentence_rnn/stack_grad/unstack
�
Cgradients/ending/sentence_rnn/stack_grad/tuple/control_dependency_1Identity2gradients/ending/sentence_rnn/stack_grad/unstack:1:^gradients/ending/sentence_rnn/stack_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/ending/sentence_rnn/stack_grad/unstack*
_output_shapes
:	�
�
@gradients/ending/sentence_rnn/fc/output/BiasAdd_grad/BiasAddGradBiasAddGradAgradients/ending/sentence_rnn/stack_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:
�
Egradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/fc/output/BiasAdd_grad/BiasAddGradB^gradients/ending/sentence_rnn/stack_grad/tuple/control_dependency
�
Mgradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/control_dependencyIdentityAgradients/ending/sentence_rnn/stack_grad/tuple/control_dependencyF^gradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/ending/sentence_rnn/stack_grad/unstack*
_output_shapes
:	�
�
Ogradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/fc/output/BiasAdd_grad/BiasAddGradF^gradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/fc/output/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
Bgradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/BiasAddGradBiasAddGradCgradients/ending/sentence_rnn/stack_grad/tuple/control_dependency_1*
data_formatNHWC*
_output_shapes
:*
T0
�
Ggradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/group_depsNoOpC^gradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/BiasAddGradD^gradients/ending/sentence_rnn/stack_grad/tuple/control_dependency_1
�
Ogradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/stack_grad/tuple/control_dependency_1H^gradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/ending/sentence_rnn/stack_grad/unstack*
_output_shapes
:	�
�
Qgradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/BiasAddGradH^gradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
:gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMulMatMulMgradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/control_dependencyending/output/kernel/read*
T0*
transpose_a( * 
_output_shapes
:
��*
transpose_b(
�
<gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMul_1MatMulending/sentence_rnn/dropout/mulMgradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes
:	�*
transpose_b( *
T0
�
Dgradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/group_depsNoOp;^gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMul=^gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMul_1
�
Lgradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/control_dependencyIdentity:gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMulE^gradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMul* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/control_dependency_1Identity<gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMul_1E^gradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/group_deps*
_output_shapes
:	�*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMul_1
�
<gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMulMatMulOgradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/control_dependencyending/output/kernel/read*
transpose_b(*
T0*
transpose_a( * 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMul_1MatMul!ending/sentence_rnn/dropout_1/mulOgradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
_output_shapes
:	�*
transpose_b( 
�
Fgradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/group_depsNoOp=^gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMul?^gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMul_1
�
Ngradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/control_dependencyIdentity<gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMulG^gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMul
�
Pgradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/control_dependency_1Identity>gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMul_1G^gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/fc_1/output/MatMul_grad/MatMul_1*
_output_shapes
:	�
�
gradients/AddNAddNOgradients/ending/sentence_rnn/fc/output/BiasAdd_grad/tuple/control_dependency_1Qgradients/ending/sentence_rnn/fc_1/output/BiasAdd_grad/tuple/control_dependency_1*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/fc/output/BiasAdd_grad/BiasAddGrad*
N*
_output_shapes
:
�
2gradients/ending/sentence_rnn/dropout/mul_grad/MulMulLgradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/control_dependency!ending/sentence_rnn/dropout/Floor*
T0* 
_output_shapes
:
��
�
4gradients/ending/sentence_rnn/dropout/mul_grad/Mul_1MulLgradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/control_dependency#ending/sentence_rnn/dropout/truediv*
T0* 
_output_shapes
:
��
�
?gradients/ending/sentence_rnn/dropout/mul_grad/tuple/group_depsNoOp3^gradients/ending/sentence_rnn/dropout/mul_grad/Mul5^gradients/ending/sentence_rnn/dropout/mul_grad/Mul_1
�
Ggradients/ending/sentence_rnn/dropout/mul_grad/tuple/control_dependencyIdentity2gradients/ending/sentence_rnn/dropout/mul_grad/Mul@^gradients/ending/sentence_rnn/dropout/mul_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/ending/sentence_rnn/dropout/mul_grad/Mul* 
_output_shapes
:
��
�
Igradients/ending/sentence_rnn/dropout/mul_grad/tuple/control_dependency_1Identity4gradients/ending/sentence_rnn/dropout/mul_grad/Mul_1@^gradients/ending/sentence_rnn/dropout/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/ending/sentence_rnn/dropout/mul_grad/Mul_1* 
_output_shapes
:
��
�
4gradients/ending/sentence_rnn/dropout_1/mul_grad/MulMulNgradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/control_dependency#ending/sentence_rnn/dropout_1/Floor* 
_output_shapes
:
��*
T0
�
6gradients/ending/sentence_rnn/dropout_1/mul_grad/Mul_1MulNgradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/control_dependency%ending/sentence_rnn/dropout_1/truediv*
T0* 
_output_shapes
:
��
�
Agradients/ending/sentence_rnn/dropout_1/mul_grad/tuple/group_depsNoOp5^gradients/ending/sentence_rnn/dropout_1/mul_grad/Mul7^gradients/ending/sentence_rnn/dropout_1/mul_grad/Mul_1
�
Igradients/ending/sentence_rnn/dropout_1/mul_grad/tuple/control_dependencyIdentity4gradients/ending/sentence_rnn/dropout_1/mul_grad/MulB^gradients/ending/sentence_rnn/dropout_1/mul_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients/ending/sentence_rnn/dropout_1/mul_grad/Mul* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/dropout_1/mul_grad/tuple/control_dependency_1Identity6gradients/ending/sentence_rnn/dropout_1/mul_grad/Mul_1B^gradients/ending/sentence_rnn/dropout_1/mul_grad/tuple/group_deps*
T0*I
_class?
=;loc:@gradients/ending/sentence_rnn/dropout_1/mul_grad/Mul_1* 
_output_shapes
:
��
�
gradients/AddN_1AddNNgradients/ending/sentence_rnn/fc/output/MatMul_grad/tuple/control_dependency_1Pgradients/ending/sentence_rnn/fc_1/output/MatMul_grad/tuple/control_dependency_1*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/fc/output/MatMul_grad/MatMul_1*
N*
_output_shapes
:	�
�
8gradients/ending/sentence_rnn/dropout/truediv_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�   �  
}
:gradients/ending/sentence_rnn/dropout/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Hgradients/ending/sentence_rnn/dropout/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs8gradients/ending/sentence_rnn/dropout/truediv_grad/Shape:gradients/ending/sentence_rnn/dropout/truediv_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
:gradients/ending/sentence_rnn/dropout/truediv_grad/RealDivRealDivGgradients/ending/sentence_rnn/dropout/mul_grad/tuple/control_dependencyending/sentence_rnn/dropout/sub*
T0* 
_output_shapes
:
��
�
6gradients/ending/sentence_rnn/dropout/truediv_grad/SumSum:gradients/ending/sentence_rnn/dropout/truediv_grad/RealDivHgradients/ending/sentence_rnn/dropout/truediv_grad/BroadcastGradientArgs*
T0* 
_output_shapes
:
��*

Tidx0*
	keep_dims( 
�
:gradients/ending/sentence_rnn/dropout/truediv_grad/ReshapeReshape6gradients/ending/sentence_rnn/dropout/truediv_grad/Sum8gradients/ending/sentence_rnn/dropout/truediv_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
6gradients/ending/sentence_rnn/dropout/truediv_grad/NegNegending/sentence_rnn/concat_2*
T0* 
_output_shapes
:
��
�
<gradients/ending/sentence_rnn/dropout/truediv_grad/RealDiv_1RealDiv6gradients/ending/sentence_rnn/dropout/truediv_grad/Negending/sentence_rnn/dropout/sub*
T0* 
_output_shapes
:
��
�
<gradients/ending/sentence_rnn/dropout/truediv_grad/RealDiv_2RealDiv<gradients/ending/sentence_rnn/dropout/truediv_grad/RealDiv_1ending/sentence_rnn/dropout/sub* 
_output_shapes
:
��*
T0
�
6gradients/ending/sentence_rnn/dropout/truediv_grad/mulMulGgradients/ending/sentence_rnn/dropout/mul_grad/tuple/control_dependency<gradients/ending/sentence_rnn/dropout/truediv_grad/RealDiv_2*
T0* 
_output_shapes
:
��
�
8gradients/ending/sentence_rnn/dropout/truediv_grad/Sum_1Sum6gradients/ending/sentence_rnn/dropout/truediv_grad/mulJgradients/ending/sentence_rnn/dropout/truediv_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
<gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape_1Reshape8gradients/ending/sentence_rnn/dropout/truediv_grad/Sum_1:gradients/ending/sentence_rnn/dropout/truediv_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
Cgradients/ending/sentence_rnn/dropout/truediv_grad/tuple/group_depsNoOp;^gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape=^gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape_1
�
Kgradients/ending/sentence_rnn/dropout/truediv_grad/tuple/control_dependencyIdentity:gradients/ending/sentence_rnn/dropout/truediv_grad/ReshapeD^gradients/ending/sentence_rnn/dropout/truediv_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/dropout/truediv_grad/tuple/control_dependency_1Identity<gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape_1D^gradients/ending/sentence_rnn/dropout/truediv_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape_1*
_output_shapes
: 
�
:gradients/ending/sentence_rnn/dropout_1/truediv_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:

<gradients/ending/sentence_rnn/dropout_1/truediv_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Jgradients/ending/sentence_rnn/dropout_1/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs:gradients/ending/sentence_rnn/dropout_1/truediv_grad/Shape<gradients/ending/sentence_rnn/dropout_1/truediv_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<gradients/ending/sentence_rnn/dropout_1/truediv_grad/RealDivRealDivIgradients/ending/sentence_rnn/dropout_1/mul_grad/tuple/control_dependency!ending/sentence_rnn/dropout_1/sub*
T0* 
_output_shapes
:
��
�
8gradients/ending/sentence_rnn/dropout_1/truediv_grad/SumSum<gradients/ending/sentence_rnn/dropout_1/truediv_grad/RealDivJgradients/ending/sentence_rnn/dropout_1/truediv_grad/BroadcastGradientArgs*
T0* 
_output_shapes
:
��*

Tidx0*
	keep_dims( 
�
<gradients/ending/sentence_rnn/dropout_1/truediv_grad/ReshapeReshape8gradients/ending/sentence_rnn/dropout_1/truediv_grad/Sum:gradients/ending/sentence_rnn/dropout_1/truediv_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
8gradients/ending/sentence_rnn/dropout_1/truediv_grad/NegNegending/sentence_rnn/concat_5*
T0* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/dropout_1/truediv_grad/RealDiv_1RealDiv8gradients/ending/sentence_rnn/dropout_1/truediv_grad/Neg!ending/sentence_rnn/dropout_1/sub* 
_output_shapes
:
��*
T0
�
>gradients/ending/sentence_rnn/dropout_1/truediv_grad/RealDiv_2RealDiv>gradients/ending/sentence_rnn/dropout_1/truediv_grad/RealDiv_1!ending/sentence_rnn/dropout_1/sub* 
_output_shapes
:
��*
T0
�
8gradients/ending/sentence_rnn/dropout_1/truediv_grad/mulMulIgradients/ending/sentence_rnn/dropout_1/mul_grad/tuple/control_dependency>gradients/ending/sentence_rnn/dropout_1/truediv_grad/RealDiv_2*
T0* 
_output_shapes
:
��
�
:gradients/ending/sentence_rnn/dropout_1/truediv_grad/Sum_1Sum8gradients/ending/sentence_rnn/dropout_1/truediv_grad/mulLgradients/ending/sentence_rnn/dropout_1/truediv_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
>gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape_1Reshape:gradients/ending/sentence_rnn/dropout_1/truediv_grad/Sum_1<gradients/ending/sentence_rnn/dropout_1/truediv_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Egradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/group_depsNoOp=^gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape?^gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape_1
�
Mgradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/control_dependencyIdentity<gradients/ending/sentence_rnn/dropout_1/truediv_grad/ReshapeF^gradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape* 
_output_shapes
:
��
�
Ogradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/control_dependency_1Identity>gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape_1F^gradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/group_deps*
_output_shapes
: *
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape_1
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/group_depsNoOpL^gradients/ending/sentence_rnn/dropout/truediv_grad/tuple/control_dependency
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/control_dependencyIdentityKgradients/ending/sentence_rnn/dropout/truediv_grad/tuple/control_dependencyL^gradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/control_dependency_1IdentityKgradients/ending/sentence_rnn/dropout/truediv_grad/tuple/control_dependencyL^gradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/group_deps*
T0*M
_classC
A?loc:@gradients/ending/sentence_rnn/dropout/truediv_grad/Reshape* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/group_depsNoOpN^gradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/control_dependency
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/control_dependencyIdentityMgradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/control_dependencyN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/control_dependency_1IdentityMgradients/ending/sentence_rnn/dropout_1/truediv_grad/tuple/control_dependencyN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/dropout_1/truediv_grad/Reshape* 
_output_shapes
:
��
�
?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/MulMulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/control_dependency+ending/sentence_rnn/rnn/sentence_cell/add_7* 
_output_shapes
:
��*
T0
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul_1MulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/control_dependency0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_12*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/tuple/group_depsNoOp@^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/MulB^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul_1
�
Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/tuple/control_dependencyIdentity?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/MulM^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul* 
_output_shapes
:
��
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/tuple/control_dependency_1IdentityAgradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul_1M^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul_1
�
?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/MulMulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_8*
T0* 
_output_shapes
:
��
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_9_grad/tuple/control_dependency_10ending/sentence_rnn/rnn/sentence_cell/Sigmoid_13*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/tuple/group_depsNoOp@^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/MulB^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/Mul_1
�
Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/tuple/control_dependencyIdentity?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/MulM^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*R
_classH
FDloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/Mul
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/tuple/control_dependency_1IdentityAgradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/Mul_1M^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/Mul_1
�
Agradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/MulMulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/control_dependency-ending/sentence_rnn/rnn_1/sentence_cell/add_7*
T0* 
_output_shapes
:
��
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/control_dependency2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_12* 
_output_shapes
:
��*
T0
�
Ngradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/tuple/group_depsNoOpB^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/MulD^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul_1
�
Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/tuple/control_dependencyIdentityAgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/MulO^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul* 
_output_shapes
:
��
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/tuple/control_dependency_1IdentityCgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul_1O^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul_1
�
Agradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/MulMulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_8*
T0* 
_output_shapes
:
��
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/Mul_1MulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_9_grad/tuple/control_dependency_12ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_13*
T0* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/tuple/group_depsNoOpB^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/MulD^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/Mul_1
�
Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/tuple/control_dependencyIdentityAgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/MulO^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/Mul
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/tuple/control_dependency_1IdentityCgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/Mul_1O^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/Mul_1* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_12_grad/SigmoidGradSigmoidGrad0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_12Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_13_grad/SigmoidGradSigmoidGrad0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_13Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_8_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_8Vgradients/ending/sentence_rnn/rnn/sentence_cell/mul_13_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_12_grad/SigmoidGradSigmoidGrad2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_12Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_13_grad/SigmoidGradSigmoidGrad2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_13Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_8_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_8Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_13_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Pgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/ShapeBgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/SumSumKgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_12_grad/SigmoidGradPgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/ReshapeReshape>gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Sum@gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Shape* 
_output_shapes
:
��*
T0*
Tshape0
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Sum_1SumKgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_12_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Reshape_1Reshape@gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Sum_1Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/tuple/group_depsNoOpC^gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/ReshapeE^gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Reshape_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/tuple/control_dependencyIdentityBgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/ReshapeL^gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Reshape
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/tuple/control_dependency_1IdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Reshape_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/Reshape_1*
_output_shapes
: 
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Rgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/ShapeDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/SumSumMgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_12_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/ReshapeReshape@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/SumBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Sum_1SumMgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_12_grad/SigmoidGradTgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Reshape_1ReshapeBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Sum_1Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/ReshapeG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Reshape_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/ReshapeN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Reshape* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Reshape_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/Reshape_1*
_output_shapes
: 

gradients/zeros_like_1	ZerosLike/ending/sentence_rnn/rnn/sentence_cell/split_4:3*
T0* 
_output_shapes
:
��
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/split_4_grad/concatConcatV2Kgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_13_grad/SigmoidGradDgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_8_grad/TanhGradSgradients/ending/sentence_rnn/rnn/sentence_cell/add_8_grad/tuple/control_dependencygradients/zeros_like_17ending/sentence_rnn/rnn/sentence_cell/split_4/split_dim*
T0*
N* 
_output_shapes
:
��*

Tidx0
�
gradients/zeros_like_2	ZerosLike1ending/sentence_rnn/rnn_1/sentence_cell/split_4:3* 
_output_shapes
:
��*
T0
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/split_4_grad/concatConcatV2Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_13_grad/SigmoidGradFgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_8_grad/TanhGradUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_8_grad/tuple/control_dependencygradients/zeros_like_29ending/sentence_rnn/rnn_1/sentence_cell/split_4/split_dim*
N* 
_output_shapes
:
��*

Tidx0*
T0
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/BiasAddGradBiasAddGradCgradients/ending/sentence_rnn/rnn/sentence_cell/split_4_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ogradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/group_depsNoOpK^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/BiasAddGradD^gradients/ending/sentence_rnn/rnn/sentence_cell/split_4_grad/concat
�
Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/split_4_grad/concatP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/split_4_grad/concat* 
_output_shapes
:
��
�
Ygradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/control_dependency_1IdentityJgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/BiasAddGradP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/group_deps*
_output_shapes	
:�*
T0*]
_classS
QOloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/BiasAddGrad
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/BiasAddGradBiasAddGradEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_4_grad/concat*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Qgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/group_depsNoOpM^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/BiasAddGradF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_4_grad/concat
�
Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_4_grad/concatR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_4_grad/concat* 
_output_shapes
:
��
�
[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/control_dependency_1IdentityLgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/BiasAddGradR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/group_deps*
_output_shapes	
:�*
T0*_
_classU
SQloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/BiasAddGrad
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMulMatMulWgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
transpose_b(*
T0*
transpose_a( * 
_output_shapes
:
��-
�
Fgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMul_1MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_4Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
�-�*
transpose_b( 
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMulG^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMul_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMulO^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMul* 
_output_shapes
:
��-
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMul_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMul_1* 
_output_shapes
:
�-�
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMulMatMulYgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
T0*
transpose_a( * 
_output_shapes
:
��-*
transpose_b(
�
Hgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMul_1MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_4Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
�-�*
transpose_b( 
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/group_depsNoOpG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMulI^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMul_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/control_dependencyIdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMulQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMul* 
_output_shapes
:
��-
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/control_dependency_1IdentityHgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMul_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/MatMul_1* 
_output_shapes
:
�-�
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/modFloorMod3ending/sentence_rnn/rnn/sentence_cell/concat_4/axisBgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Rank*
T0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/ConcatOffsetConcatOffsetAgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/modCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/ShapeEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Shape_1*
N* 
_output_shapes
::
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/SliceSliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/control_dependencyJgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/ConcatOffsetCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Shape*
T0*
Index0* 
_output_shapes
:
��%
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Slice_1SliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/ConcatOffset:1Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Shape_1*
T0*
Index0* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/tuple/group_depsNoOpD^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/SliceF^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Slice_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/SliceO^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/tuple/group_deps* 
_output_shapes
:
��%*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Slice
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/tuple/control_dependency_1IdentityEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Slice_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/Slice_1* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/modFloorMod5ending/sentence_rnn/rnn_1/sentence_cell/concat_4/axisDgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Rank*
T0*
_output_shapes
: 
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/ConcatOffsetConcatOffsetCgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/modEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/ShapeGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Shape_1*
N* 
_output_shapes
::
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/SliceSliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/ConcatOffsetEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Shape*
T0*
Index0* 
_output_shapes
:
��%
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Slice_1SliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/control_dependencyNgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/ConcatOffset:1Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Shape_1*
T0*
Index0* 
_output_shapes
:
��
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/tuple/group_depsNoOpF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/SliceH^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Slice_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/SliceQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Slice* 
_output_shapes
:
��%
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/tuple/control_dependency_1IdentityGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Slice_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*Z
_classP
NLloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/Slice_1
�
?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/MulMulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_7*
T0* 
_output_shapes
:
��
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/Mul_1MulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_4_grad/tuple/control_dependency_10ending/sentence_rnn/rnn/sentence_cell/Sigmoid_11* 
_output_shapes
:
��*
T0
�
Lgradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/tuple/group_depsNoOp@^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/MulB^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/Mul_1
�
Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/tuple/control_dependencyIdentity?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/MulM^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/Mul* 
_output_shapes
:
��
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/tuple/control_dependency_1IdentityAgradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/Mul_1M^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/Mul_1* 
_output_shapes
:
��
�
Agradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/MulMulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_7*
T0* 
_output_shapes
:
��
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/Mul_1MulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_4_grad/tuple/control_dependency_12ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_11*
T0* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/tuple/group_depsNoOpB^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/MulD^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/Mul_1
�
Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/tuple/control_dependencyIdentityAgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/MulO^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/Mul
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/tuple/control_dependency_1IdentityCgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/Mul_1O^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/Mul_1* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_11_grad/SigmoidGradSigmoidGrad0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_11Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_7_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_7Vgradients/ending/sentence_rnn/rnn/sentence_cell/mul_11_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_11_grad/SigmoidGradSigmoidGrad2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_11Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_7_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_7Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_11_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
T0
�
gradients/AddN_2AddNVgradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/tuple/control_dependency_1Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_7_grad/TanhGrad*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul_1*
N* 
_output_shapes
:
��
f
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/group_depsNoOp^gradients/AddN_2
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/control_dependencyIdentitygradients/AddN_2L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul_1* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/control_dependency_1Identitygradients/AddN_2L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_12_grad/Mul_1* 
_output_shapes
:
��
�
gradients/AddN_3AddNXgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/tuple/control_dependency_1Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_7_grad/TanhGrad*
N* 
_output_shapes
:
��*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul_1
h
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/group_depsNoOp^gradients/AddN_3
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/control_dependencyIdentitygradients/AddN_3N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul_1* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/control_dependency_1Identitygradients/AddN_3N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_12_grad/Mul_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/MulMulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/control_dependency+ending/sentence_rnn/rnn/sentence_cell/add_5*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul_1MulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/control_dependency/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_9*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul_1* 
_output_shapes
:
��
�
?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/MulMulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_6*
T0* 
_output_shapes
:
��
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_7_grad/tuple/control_dependency_10ending/sentence_rnn/rnn/sentence_cell/Sigmoid_10* 
_output_shapes
:
��*
T0
�
Lgradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/tuple/group_depsNoOp@^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/MulB^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/Mul_1
�
Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/tuple/control_dependencyIdentity?gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/MulM^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/tuple/group_deps*
T0*R
_classH
FDloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/Mul* 
_output_shapes
:
��
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/tuple/control_dependency_1IdentityAgradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/Mul_1M^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/Mul_1
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/MulMulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/control_dependency-ending/sentence_rnn/rnn_1/sentence_cell/add_5*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/control_dependency1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_9*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul_1* 
_output_shapes
:
��
�
Agradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/MulMulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_6*
T0* 
_output_shapes
:
��
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/Mul_1MulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_7_grad/tuple/control_dependency_12ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_10* 
_output_shapes
:
��*
T0
�
Ngradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/tuple/group_depsNoOpB^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/MulD^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/Mul_1
�
Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/tuple/control_dependencyIdentityAgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/MulO^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/Mul
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/tuple/control_dependency_1IdentityCgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/Mul_1O^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/Mul_1* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_9_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_9Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_10_grad/SigmoidGradSigmoidGrad0ending/sentence_rnn/rnn/sentence_cell/Sigmoid_10Tgradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_6_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_6Vgradients/ending/sentence_rnn/rnn/sentence_cell/mul_10_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_9_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_9Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_10_grad/SigmoidGradSigmoidGrad2ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_10Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_6_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_6Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_10_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Pgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/ShapeBgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/SumSumJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_9_grad/SigmoidGradPgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/BroadcastGradientArgs* 
_output_shapes
:
��*

Tidx0*
	keep_dims( *
T0
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/ReshapeReshape>gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Sum@gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Sum_1SumJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_9_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Reshape_1Reshape@gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Sum_1Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/tuple/group_depsNoOpC^gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/ReshapeE^gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Reshape_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/tuple/control_dependencyIdentityBgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/ReshapeL^gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Reshape
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/tuple/control_dependency_1IdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Reshape_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/Reshape_1*
_output_shapes
: 
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Rgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/ShapeDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/SumSumLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_9_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/BroadcastGradientArgs* 
_output_shapes
:
��*

Tidx0*
	keep_dims( *
T0
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/ReshapeReshape@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/SumBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Sum_1SumLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_9_grad/SigmoidGradTgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Reshape_1ReshapeBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Sum_1Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/ReshapeG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Reshape_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/ReshapeN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Reshape* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Reshape_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/Reshape_1*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/split_3_grad/concatConcatV2Kgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_10_grad/SigmoidGradDgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_6_grad/TanhGradSgradients/ending/sentence_rnn/rnn/sentence_cell/add_6_grad/tuple/control_dependencyKgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_11_grad/SigmoidGrad7ending/sentence_rnn/rnn/sentence_cell/split_3/split_dim*
N* 
_output_shapes
:
��*

Tidx0*
T0
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/split_3_grad/concatConcatV2Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_10_grad/SigmoidGradFgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_6_grad/TanhGradUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_6_grad/tuple/control_dependencyMgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_11_grad/SigmoidGrad9ending/sentence_rnn/rnn_1/sentence_cell/split_3/split_dim*
T0*
N* 
_output_shapes
:
��*

Tidx0
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/BiasAddGradBiasAddGradCgradients/ending/sentence_rnn/rnn/sentence_cell/split_3_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ogradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/group_depsNoOpK^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/BiasAddGradD^gradients/ending/sentence_rnn/rnn/sentence_cell/split_3_grad/concat
�
Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/split_3_grad/concatP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/split_3_grad/concat* 
_output_shapes
:
��
�
Ygradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/control_dependency_1IdentityJgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/BiasAddGradP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/BiasAddGrad*
_output_shapes	
:�
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/BiasAddGradBiasAddGradEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_3_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Qgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/group_depsNoOpM^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/BiasAddGradF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_3_grad/concat
�
Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_3_grad/concatR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_3_grad/concat* 
_output_shapes
:
��
�
[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/control_dependency_1IdentityLgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/BiasAddGradR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/BiasAddGrad*
_output_shapes	
:�
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMulMatMulWgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
transpose_a( * 
_output_shapes
:
��-*
transpose_b(*
T0
�
Fgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMul_1MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_3Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
�-�*
transpose_b( 
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMulG^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMul_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMulO^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMul* 
_output_shapes
:
��-
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMul_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/MatMul_1* 
_output_shapes
:
�-�
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMulMatMulYgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
T0*
transpose_a( * 
_output_shapes
:
��-*
transpose_b(
�
Hgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMul_1MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_3Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
�-�*
transpose_b( 
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/group_depsNoOpG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMulI^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMul_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/control_dependencyIdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMulQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMul* 
_output_shapes
:
��-
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/control_dependency_1IdentityHgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMul_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/MatMul_1* 
_output_shapes
:
�-�
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/modFloorMod3ending/sentence_rnn/rnn/sentence_cell/concat_3/axisBgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Rank*
_output_shapes
: *
T0
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Shape_1Const*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/ConcatOffsetConcatOffsetAgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/modCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/ShapeEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Shape_1*
N* 
_output_shapes
::
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/SliceSliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/control_dependencyJgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/ConcatOffsetCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Shape*
T0*
Index0* 
_output_shapes
:
��%
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Slice_1SliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/ConcatOffset:1Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Shape_1*
T0*
Index0* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/tuple/group_depsNoOpD^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/SliceF^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Slice_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/SliceO^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/tuple/group_deps* 
_output_shapes
:
��%*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Slice
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/tuple/control_dependency_1IdentityEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Slice_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/Slice_1
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/modFloorMod5ending/sentence_rnn/rnn_1/sentence_cell/concat_3/axisDgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Rank*
_output_shapes
: *
T0
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Shape_1Const*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/ConcatOffsetConcatOffsetCgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/modEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/ShapeGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Shape_1*
N* 
_output_shapes
::
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/SliceSliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/ConcatOffsetEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Shape*
T0*
Index0* 
_output_shapes
:
��%
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Slice_1SliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/control_dependencyNgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/ConcatOffset:1Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Shape_1*
T0*
Index0* 
_output_shapes
:
��
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/tuple/group_depsNoOpF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/SliceH^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Slice_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/SliceQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Slice* 
_output_shapes
:
��%
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/tuple/control_dependency_1IdentityGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Slice_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/Slice_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/MulMulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_5*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/Mul_1MulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_3_grad/tuple/control_dependency_1/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_8*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/Mul_1
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/MulMulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_5*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/Mul_1MulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_3_grad/tuple/control_dependency_11ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_8*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/Mul
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/Mul_1* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_8_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_8Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_5_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_5Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_8_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
T0
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_8_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_8Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_5_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_5Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_8_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
gradients/AddN_4AddNUgradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/tuple/control_dependency_1Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_5_grad/TanhGrad*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul_1*
N* 
_output_shapes
:
��
f
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/group_depsNoOp^gradients/AddN_4
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/control_dependencyIdentitygradients/AddN_4L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul_1* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/control_dependency_1Identitygradients/AddN_4L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_9_grad/Mul_1* 
_output_shapes
:
��
�
gradients/AddN_5AddNWgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/tuple/control_dependency_1Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_5_grad/TanhGrad*
N* 
_output_shapes
:
��*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul_1
h
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/group_depsNoOp^gradients/AddN_5
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/control_dependencyIdentitygradients/AddN_5N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul_1* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/control_dependency_1Identitygradients/AddN_5N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_9_grad/Mul_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/MulMulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/control_dependency+ending/sentence_rnn/rnn/sentence_cell/add_3* 
_output_shapes
:
��*
T0
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul_1MulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/control_dependency/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_6* 
_output_shapes
:
��*
T0
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/MulMulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_4* 
_output_shapes
:
��*
T0
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_5_grad/tuple/control_dependency_1/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_7* 
_output_shapes
:
��*
T0
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/Mul_1* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/MulMulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/control_dependency-ending/sentence_rnn/rnn_1/sentence_cell/add_3*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/control_dependency1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_6*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul_1* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/MulMulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_4*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/Mul_1MulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_5_grad/tuple/control_dependency_11ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_7*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/Mul
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/Mul_1* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_6_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_6Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_7_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_7Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_4_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_4Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_7_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_6_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_6Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_7_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_7Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_4_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_4Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_7_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Pgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/ShapeBgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/SumSumJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_6_grad/SigmoidGradPgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/ReshapeReshape>gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Sum@gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Sum_1SumJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_6_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Reshape_1Reshape@gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Sum_1Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/tuple/group_depsNoOpC^gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/ReshapeE^gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Reshape_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/tuple/control_dependencyIdentityBgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/ReshapeL^gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Reshape
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/tuple/control_dependency_1IdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Reshape_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/Reshape_1*
_output_shapes
: 
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB 
�
Rgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/ShapeDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/SumSumLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_6_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/BroadcastGradientArgs*
T0* 
_output_shapes
:
��*

Tidx0*
	keep_dims( 
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/ReshapeReshape@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/SumBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Sum_1SumLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_6_grad/SigmoidGradTgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Reshape_1ReshapeBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Sum_1Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/ReshapeG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Reshape_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/ReshapeN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Reshape* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Reshape_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/Reshape_1*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/split_2_grad/concatConcatV2Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_7_grad/SigmoidGradDgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_4_grad/TanhGradSgradients/ending/sentence_rnn/rnn/sentence_cell/add_4_grad/tuple/control_dependencyJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_8_grad/SigmoidGrad7ending/sentence_rnn/rnn/sentence_cell/split_2/split_dim*

Tidx0*
T0*
N* 
_output_shapes
:
��
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/split_2_grad/concatConcatV2Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_7_grad/SigmoidGradFgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_4_grad/TanhGradUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_4_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_8_grad/SigmoidGrad9ending/sentence_rnn/rnn_1/sentence_cell/split_2/split_dim*
T0*
N* 
_output_shapes
:
��*

Tidx0
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/BiasAddGradBiasAddGradCgradients/ending/sentence_rnn/rnn/sentence_cell/split_2_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ogradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/group_depsNoOpK^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/BiasAddGradD^gradients/ending/sentence_rnn/rnn/sentence_cell/split_2_grad/concat
�
Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/split_2_grad/concatP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/split_2_grad/concat* 
_output_shapes
:
��
�
Ygradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/control_dependency_1IdentityJgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/BiasAddGradP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/BiasAddGrad*
_output_shapes	
:�
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/BiasAddGradBiasAddGradEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_2_grad/concat*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Qgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/group_depsNoOpM^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/BiasAddGradF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_2_grad/concat
�
Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_2_grad/concatR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_2_grad/concat
�
[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/control_dependency_1IdentityLgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/BiasAddGradR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/group_deps*
_output_shapes	
:�*
T0*_
_classU
SQloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/BiasAddGrad
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMulMatMulWgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
T0*
transpose_a( * 
_output_shapes
:
��-*
transpose_b(
�
Fgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMul_1MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_2Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/control_dependency*
transpose_a(* 
_output_shapes
:
�-�*
transpose_b( *
T0
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMulG^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMul_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMulO^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMul* 
_output_shapes
:
��-
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMul_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/group_deps* 
_output_shapes
:
�-�*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/MatMul_1
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMulMatMulYgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
T0*
transpose_a( * 
_output_shapes
:
��-*
transpose_b(
�
Hgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMul_1MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_2Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
�-�*
transpose_b( 
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/group_depsNoOpG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMulI^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMul_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/control_dependencyIdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMulQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMul* 
_output_shapes
:
��-
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/control_dependency_1IdentityHgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMul_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/MatMul_1* 
_output_shapes
:
�-�
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/modFloorMod3ending/sentence_rnn/rnn/sentence_cell/concat_2/axisBgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Rank*
T0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/ConcatOffsetConcatOffsetAgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/modCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/ShapeEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Shape_1*
N* 
_output_shapes
::
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/SliceSliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/control_dependencyJgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/ConcatOffsetCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Shape*
T0*
Index0* 
_output_shapes
:
��%
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Slice_1SliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/ConcatOffset:1Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Shape_1*
T0*
Index0* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/tuple/group_depsNoOpD^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/SliceF^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Slice_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/SliceO^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Slice* 
_output_shapes
:
��%
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/tuple/control_dependency_1IdentityEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Slice_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/Slice_1* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/modFloorMod5ending/sentence_rnn/rnn_1/sentence_cell/concat_2/axisDgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Rank*
_output_shapes
: *
T0
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/ConcatOffsetConcatOffsetCgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/modEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/ShapeGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Shape_1*
N* 
_output_shapes
::
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/SliceSliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/ConcatOffsetEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Shape* 
_output_shapes
:
��%*
T0*
Index0
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Slice_1SliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/control_dependencyNgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/ConcatOffset:1Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Shape_1*
T0*
Index0* 
_output_shapes
:
��
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/tuple/group_depsNoOpF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/SliceH^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Slice_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/SliceQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Slice* 
_output_shapes
:
��%
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/tuple/control_dependency_1IdentityGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Slice_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/Slice_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/MulMulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_3*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/Mul_1MulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_2_grad/tuple/control_dependency_1/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_5*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/Mul_1
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/MulMulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_3*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/Mul_1MulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_2_grad/tuple/control_dependency_11ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_5*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/Mul* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/Mul_1
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_5_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_5Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_3_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_3Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_5_grad/tuple/control_dependency_1* 
_output_shapes
:
��*
T0
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_5_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_5Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_3_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_3Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_5_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
gradients/AddN_6AddNUgradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/tuple/control_dependency_1Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_3_grad/TanhGrad*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul_1*
N* 
_output_shapes
:
��
f
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/group_depsNoOp^gradients/AddN_6
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/control_dependencyIdentitygradients/AddN_6L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul_1* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/control_dependency_1Identitygradients/AddN_6L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_6_grad/Mul_1* 
_output_shapes
:
��
�
gradients/AddN_7AddNWgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/tuple/control_dependency_1Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_3_grad/TanhGrad*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul_1*
N* 
_output_shapes
:
��
h
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/group_depsNoOp^gradients/AddN_7
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/control_dependencyIdentitygradients/AddN_7N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul_1
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/control_dependency_1Identitygradients/AddN_7N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_6_grad/Mul_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/MulMulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/control_dependency+ending/sentence_rnn/rnn/sentence_cell/add_1*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul_1MulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/control_dependency/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_3*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/MulMulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_2*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_3_grad/tuple/control_dependency_1/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_4*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/Mul_1* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/MulMulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/control_dependency-ending/sentence_rnn/rnn_1/sentence_cell/add_1*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/control_dependency1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_3*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul_1* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/MulMulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_2*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/Mul_1MulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_3_grad/tuple/control_dependency_11ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_4*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/Mul* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/Mul_1* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_3_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_3Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_4_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_4Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_2_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_2Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_4_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_3_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_3Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_4_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_4Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_2_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_2Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_4_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Pgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/ShapeBgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/SumSumJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_3_grad/SigmoidGradPgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/BroadcastGradientArgs*
T0* 
_output_shapes
:
��*

Tidx0*
	keep_dims( 
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/ReshapeReshape>gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Sum@gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Sum_1SumJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_3_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Reshape_1Reshape@gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Sum_1Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/tuple/group_depsNoOpC^gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/ReshapeE^gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Reshape_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/tuple/control_dependencyIdentityBgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/ReshapeL^gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Reshape* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/tuple/control_dependency_1IdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Reshape_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/Reshape_1*
_output_shapes
: 
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Rgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/ShapeDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/SumSumLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_3_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/BroadcastGradientArgs*
T0* 
_output_shapes
:
��*

Tidx0*
	keep_dims( 
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/ReshapeReshape@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/SumBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Sum_1SumLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_3_grad/SigmoidGradTgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Reshape_1ReshapeBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Sum_1Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/ReshapeG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Reshape_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/ReshapeN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Reshape
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Reshape_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/tuple/group_deps*
_output_shapes
: *
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/Reshape_1
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/split_1_grad/concatConcatV2Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_4_grad/SigmoidGradDgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_2_grad/TanhGradSgradients/ending/sentence_rnn/rnn/sentence_cell/add_2_grad/tuple/control_dependencyJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_5_grad/SigmoidGrad7ending/sentence_rnn/rnn/sentence_cell/split_1/split_dim*
N* 
_output_shapes
:
��*

Tidx0*
T0
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/split_1_grad/concatConcatV2Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_4_grad/SigmoidGradFgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_2_grad/TanhGradUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_2_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_5_grad/SigmoidGrad9ending/sentence_rnn/rnn_1/sentence_cell/split_1/split_dim*

Tidx0*
T0*
N* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/BiasAddGradBiasAddGradCgradients/ending/sentence_rnn/rnn/sentence_cell/split_1_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Ogradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/group_depsNoOpK^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/BiasAddGradD^gradients/ending/sentence_rnn/rnn/sentence_cell/split_1_grad/concat
�
Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/split_1_grad/concatP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/split_1_grad/concat* 
_output_shapes
:
��
�
Ygradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/control_dependency_1IdentityJgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/BiasAddGradP^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/group_deps*
T0*]
_classS
QOloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/BiasAddGrad*
_output_shapes	
:�
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/BiasAddGradBiasAddGradEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_1_grad/concat*
T0*
data_formatNHWC*
_output_shapes	
:�
�
Qgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/group_depsNoOpM^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/BiasAddGradF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_1_grad/concat
�
Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_1_grad/concatR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_1_grad/concat* 
_output_shapes
:
��
�
[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/control_dependency_1IdentityLgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/BiasAddGradR^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/group_deps*
T0*_
_classU
SQloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/BiasAddGrad*
_output_shapes	
:�
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMulMatMulWgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
T0*
transpose_a( * 
_output_shapes
:
��-*
transpose_b(
�
Fgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMul_1MatMul.ending/sentence_rnn/rnn/sentence_cell/concat_1Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
�-�*
transpose_b( 
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMulG^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMul_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMulO^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMul* 
_output_shapes
:
��-
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMul_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/MatMul_1* 
_output_shapes
:
�-�
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMulMatMulYgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
transpose_a( * 
_output_shapes
:
��-*
transpose_b(*
T0
�
Hgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMul_1MatMul0ending/sentence_rnn/rnn_1/sentence_cell/concat_1Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
�-�*
transpose_b( 
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/group_depsNoOpG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMulI^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMul_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/control_dependencyIdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMulQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/group_deps* 
_output_shapes
:
��-*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMul
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/control_dependency_1IdentityHgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMul_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/MatMul_1* 
_output_shapes
:
�-�
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/modFloorMod3ending/sentence_rnn/rnn/sentence_cell/concat_1/axisBgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Rank*
T0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB"�   �  
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/ConcatOffsetConcatOffsetAgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/modCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/ShapeEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Shape_1*
N* 
_output_shapes
::
�
Cgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/SliceSliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/control_dependencyJgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/ConcatOffsetCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Shape*
T0*
Index0* 
_output_shapes
:
��%
�
Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Slice_1SliceVgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/ConcatOffset:1Egradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Shape_1*
T0*
Index0* 
_output_shapes
:
��
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/tuple/group_depsNoOpD^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/SliceF^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Slice_1
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/SliceO^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/tuple/group_deps*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Slice* 
_output_shapes
:
��%
�
Xgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/tuple/control_dependency_1IdentityEgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Slice_1O^gradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/Slice_1* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/modFloorMod5ending/sentence_rnn/rnn_1/sentence_cell/concat_1/axisDgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Rank*
_output_shapes
: *
T0
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Shape_1Const*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/ConcatOffsetConcatOffsetCgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/modEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/ShapeGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Shape_1*
N* 
_output_shapes
::
�
Egradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/SliceSliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/ConcatOffsetEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Shape*
T0*
Index0* 
_output_shapes
:
��%
�
Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Slice_1SliceXgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/control_dependencyNgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/ConcatOffset:1Ggradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Shape_1*
T0*
Index0* 
_output_shapes
:
��
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/tuple/group_depsNoOpF^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/SliceH^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Slice_1
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/tuple/control_dependencyIdentityEgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/SliceQ^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/tuple/group_deps*
T0*X
_classN
LJloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Slice* 
_output_shapes
:
��%
�
Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/tuple/control_dependency_1IdentityGgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Slice_1Q^gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/Slice_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/MulMulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn/sentence_cell/Tanh_1* 
_output_shapes
:
��*
T0
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/Mul_1MulXgradients/ending/sentence_rnn/rnn/sentence_cell/concat_1_grad/tuple/control_dependency_1/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_2* 
_output_shapes
:
��*
T0
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/Mul_1* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/MulMulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/tuple/control_dependency_1.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_1*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/Mul_1MulZgradients/ending/sentence_rnn/rnn_1/sentence_cell/concat_1_grad/tuple/control_dependency_11ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_2*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/Mul* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/Mul_1
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_2Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_1_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn/sentence_cell/Tanh_1Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_2_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_2_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_2Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_1_grad/TanhGradTanhGrad.ending/sentence_rnn/rnn_1/sentence_cell/Tanh_1Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_2_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
gradients/AddN_8AddNUgradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/tuple/control_dependency_1Dgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_1_grad/TanhGrad*
N* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul_1
f
Kgradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/group_depsNoOp^gradients/AddN_8
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/control_dependencyIdentitygradients/AddN_8L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul_1* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/control_dependency_1Identitygradients/AddN_8L^gradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_3_grad/Mul_1
�
gradients/AddN_9AddNWgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/tuple/control_dependency_1Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_1_grad/TanhGrad*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul_1*
N* 
_output_shapes
:
��
h
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/group_depsNoOp^gradients/AddN_9
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/control_dependencyIdentitygradients/AddN_9N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul_1* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/control_dependency_1Identitygradients/AddN_9N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_3_grad/Mul_1
�
<gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/MulMulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/control_dependency/ending/sentence_rnn/rnn/LSTMCellZeroState/zeros*
T0* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/Mul_1MulSgradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/control_dependency-ending/sentence_rnn/rnn/sentence_cell/Sigmoid*
T0* 
_output_shapes
:
��
�
Igradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/tuple/group_depsNoOp=^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/Mul?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/Mul_1
�
Qgradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/tuple/control_dependencyIdentity<gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/MulJ^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/tuple/group_deps*
T0*O
_classE
CAloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/Mul* 
_output_shapes
:
��
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/tuple/control_dependency_1Identity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/Mul_1J^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/Mul_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/MulMulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/control_dependency_1*ending/sentence_rnn/rnn/sentence_cell/Tanh*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn/sentence_cell/add_1_grad/tuple/control_dependency_1/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_1* 
_output_shapes
:
��*
T0
�
Kgradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/MulA^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/MulL^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/Mul_1L^gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/Mul_1* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/MulMulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/control_dependency1ending/sentence_rnn/rnn_1/LSTMCellZeroState/zeros*
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/Mul_1MulUgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/control_dependency/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid*
T0* 
_output_shapes
:
��
�
Kgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/tuple/group_depsNoOp?^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/MulA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/Mul_1
�
Sgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/tuple/control_dependencyIdentity>gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/MulL^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/tuple/group_deps*
T0*Q
_classG
ECloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/Mul* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/tuple/control_dependency_1Identity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/Mul_1L^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/Mul_1* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/MulMulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/control_dependency_1,ending/sentence_rnn/rnn_1/sentence_cell/Tanh*
T0* 
_output_shapes
:
��
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/Mul_1MulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_1_grad/tuple/control_dependency_11ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_1*
T0* 
_output_shapes
:
��
�
Mgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/MulC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/Mul_1
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/MulN^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/Mul* 
_output_shapes
:
��
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/Mul_1N^gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/Mul_1* 
_output_shapes
:
��
�
Hgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_grad/SigmoidGradSigmoidGrad-ending/sentence_rnn/rnn/sentence_cell/SigmoidQgradients/ending/sentence_rnn/rnn/sentence_cell/mul_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_1Sgradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/tuple/control_dependency* 
_output_shapes
:
��*
T0
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_grad/TanhGradTanhGrad*ending/sentence_rnn/rnn/sentence_cell/TanhUgradients/ending/sentence_rnn/rnn/sentence_cell/mul_1_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
Jgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_grad/SigmoidGradSigmoidGrad/ending/sentence_rnn/rnn_1/sentence_cell/SigmoidSgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_1_grad/SigmoidGradSigmoidGrad1ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_1Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/tuple/control_dependency*
T0* 
_output_shapes
:
��
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_grad/TanhGradTanhGrad,ending/sentence_rnn/rnn_1/sentence_cell/TanhWgradients/ending/sentence_rnn/rnn_1/sentence_cell/mul_1_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Ngradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Shape@gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
<gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/SumSumHgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_grad/SigmoidGradNgradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/ReshapeReshape<gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Sum>gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
>gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Sum_1SumHgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_grad/SigmoidGradPgradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/BroadcastGradientArgs:1*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Reshape_1Reshape>gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Sum_1@gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
�
Igradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/tuple/group_depsNoOpA^gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/ReshapeC^gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Reshape_1
�
Qgradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/tuple/control_dependencyIdentity@gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/ReshapeJ^gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/tuple/group_deps*
T0*S
_classI
GEloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Reshape* 
_output_shapes
:
��
�
Sgradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/tuple/control_dependency_1IdentityBgradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Reshape_1J^gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/Reshape_1*
_output_shapes
: 
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/ShapeConst*
valueB"�   �  *
dtype0*
_output_shapes
:
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
�
Pgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/ShapeBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
>gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/SumSumJgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_grad/SigmoidGradPgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/BroadcastGradientArgs* 
_output_shapes
:
��*

Tidx0*
	keep_dims( *
T0
�
Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/ReshapeReshape>gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Sum@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Shape*
T0*
Tshape0* 
_output_shapes
:
��
�
@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Sum_1SumJgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_grad/SigmoidGradRgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Reshape_1Reshape@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Sum_1Bgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
�
Kgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/tuple/group_depsNoOpC^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/ReshapeE^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Reshape_1
�
Sgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/tuple/control_dependencyIdentityBgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/ReshapeL^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Reshape* 
_output_shapes
:
��
�
Ugradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/tuple/control_dependency_1IdentityDgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Reshape_1L^gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/Reshape_1*
_output_shapes
: 
�
Agradients/ending/sentence_rnn/rnn/sentence_cell/split_grad/concatConcatV2Jgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_1_grad/SigmoidGradBgradients/ending/sentence_rnn/rnn/sentence_cell/Tanh_grad/TanhGradQgradients/ending/sentence_rnn/rnn/sentence_cell/add_grad/tuple/control_dependencyJgradients/ending/sentence_rnn/rnn/sentence_cell/Sigmoid_2_grad/SigmoidGrad5ending/sentence_rnn/rnn/sentence_cell/split/split_dim*
N* 
_output_shapes
:
��*

Tidx0*
T0
�
Cgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_grad/concatConcatV2Lgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_1_grad/SigmoidGradDgradients/ending/sentence_rnn/rnn_1/sentence_cell/Tanh_grad/TanhGradSgradients/ending/sentence_rnn/rnn_1/sentence_cell/add_grad/tuple/control_dependencyLgradients/ending/sentence_rnn/rnn_1/sentence_cell/Sigmoid_2_grad/SigmoidGrad7ending/sentence_rnn/rnn_1/sentence_cell/split/split_dim*
T0*
N* 
_output_shapes
:
��*

Tidx0
�
Hgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/BiasAddGradBiasAddGradAgradients/ending/sentence_rnn/rnn/sentence_cell/split_grad/concat*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Mgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/group_depsNoOpI^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/BiasAddGradB^gradients/ending/sentence_rnn/rnn/sentence_cell/split_grad/concat
�
Ugradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/control_dependencyIdentityAgradients/ending/sentence_rnn/rnn/sentence_cell/split_grad/concatN^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*T
_classJ
HFloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/split_grad/concat
�
Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/control_dependency_1IdentityHgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/BiasAddGradN^gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:�
�
Jgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/BiasAddGradBiasAddGradCgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_grad/concat*
data_formatNHWC*
_output_shapes	
:�*
T0
�
Ogradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/group_depsNoOpK^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/BiasAddGradD^gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_grad/concat
�
Wgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/control_dependencyIdentityCgradients/ending/sentence_rnn/rnn_1/sentence_cell/split_grad/concatP^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/group_deps* 
_output_shapes
:
��*
T0*V
_classL
JHloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/split_grad/concat
�
Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/control_dependency_1IdentityJgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/BiasAddGradP^gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:�*
T0*]
_classS
QOloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/BiasAddGrad
�
Bgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMulMatMulUgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
T0*
transpose_a( * 
_output_shapes
:
��-*
transpose_b(
�
Dgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMul_1MatMul,ending/sentence_rnn/rnn/sentence_cell/concatUgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
�-�*
transpose_b( 
�
Lgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/tuple/group_depsNoOpC^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMulE^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMul_1
�
Tgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/tuple/control_dependencyIdentityBgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMulM^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/tuple/group_deps*
T0*U
_classK
IGloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMul* 
_output_shapes
:
��-
�
Vgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/tuple/control_dependency_1IdentityDgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMul_1M^gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
�-�
�
Dgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMulMatMulWgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/control_dependency$ending/rnn/sentence_cell/kernel/read*
T0*
transpose_a( * 
_output_shapes
:
��-*
transpose_b(
�
Fgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMul_1MatMul.ending/sentence_rnn/rnn_1/sentence_cell/concatWgradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
�-�*
transpose_b( 
�
Ngradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/tuple/group_depsNoOpE^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMulG^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMul_1
�
Vgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/tuple/control_dependencyIdentityDgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMulO^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMul* 
_output_shapes
:
��-
�
Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/tuple/control_dependency_1IdentityFgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMul_1O^gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/MatMul_1* 
_output_shapes
:
�-�
�
gradients/AddN_10AddNYgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/tuple/control_dependency_1[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_4_grad/tuple/control_dependency_1Ygradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_3_grad/tuple/control_dependency_1[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_3_grad/tuple/control_dependency_1Ygradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_2_grad/tuple/control_dependency_1[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_2_grad/tuple/control_dependency_1Ygradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_1_grad/tuple/control_dependency_1[gradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_1_grad/tuple/control_dependency_1Wgradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_grad/tuple/control_dependency_1Ygradients/ending/sentence_rnn/rnn_1/sentence_cell/BiasAdd_grad/tuple/control_dependency_1*
N
*
_output_shapes	
:�*
T0*]
_classS
QOloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/BiasAdd_4_grad/BiasAddGrad
�
gradients/AddN_11AddNXgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/tuple/control_dependency_1Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_4_grad/tuple/control_dependency_1Xgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_3_grad/tuple/control_dependency_1Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_3_grad/tuple/control_dependency_1Xgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_2_grad/tuple/control_dependency_1Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_2_grad/tuple/control_dependency_1Xgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_1_grad/tuple/control_dependency_1Zgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_1_grad/tuple/control_dependency_1Vgradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_grad/tuple/control_dependency_1Xgradients/ending/sentence_rnn/rnn_1/sentence_cell/MatMul_grad/tuple/control_dependency_1*
N
* 
_output_shapes
:
�-�*
T0*Y
_classO
MKloc:@gradients/ending/sentence_rnn/rnn/sentence_cell/MatMul_4_grad/MatMul_1
h
clip_by_norm/mulMulgradients/AddN_11gradients/AddN_11*
T0* 
_output_shapes
:
�-�
c
clip_by_norm/ConstConst*
dtype0*
_output_shapes
:*
valueB"       
�
clip_by_norm/SumSumclip_by_norm/mulclip_by_norm/Const*
_output_shapes

:*

Tidx0*
	keep_dims(*
T0
[
clip_by_norm/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
r
clip_by_norm/GreaterGreaterclip_by_norm/Sumclip_by_norm/Greater/y*
_output_shapes

:*
T0
m
clip_by_norm/ones_like/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
a
clip_by_norm/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
clip_by_norm/ones_likeFillclip_by_norm/ones_like/Shapeclip_by_norm/ones_like/Const*
T0*

index_type0*
_output_shapes

:
�
clip_by_norm/SelectSelectclip_by_norm/Greaterclip_by_norm/Sumclip_by_norm/ones_like*
_output_shapes

:*
T0
W
clip_by_norm/SqrtSqrtclip_by_norm/Select*
T0*
_output_shapes

:
�
clip_by_norm/Select_1Selectclip_by_norm/Greaterclip_by_norm/Sqrtclip_by_norm/Sum*
T0*
_output_shapes

:
Y
clip_by_norm/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
m
clip_by_norm/mul_1Mulgradients/AddN_11clip_by_norm/mul_1/y*
T0* 
_output_shapes
:
�-�
[
clip_by_norm/Maximum/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
w
clip_by_norm/MaximumMaximumclip_by_norm/Select_1clip_by_norm/Maximum/y*
T0*
_output_shapes

:
t
clip_by_norm/truedivRealDivclip_by_norm/mul_1clip_by_norm/Maximum*
T0* 
_output_shapes
:
�-�
Y
clip_by_normIdentityclip_by_norm/truediv*
T0* 
_output_shapes
:
�-�
e
clip_by_norm_1/mulMulgradients/AddN_10gradients/AddN_10*
T0*
_output_shapes	
:�
^
clip_by_norm_1/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
clip_by_norm_1/SumSumclip_by_norm_1/mulclip_by_norm_1/Const*
T0*
_output_shapes
:*

Tidx0*
	keep_dims(
]
clip_by_norm_1/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
t
clip_by_norm_1/GreaterGreaterclip_by_norm_1/Sumclip_by_norm_1/Greater/y*
_output_shapes
:*
T0
h
clip_by_norm_1/ones_like/ShapeConst*
dtype0*
_output_shapes
:*
valueB:
c
clip_by_norm_1/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
clip_by_norm_1/ones_likeFillclip_by_norm_1/ones_like/Shapeclip_by_norm_1/ones_like/Const*
T0*

index_type0*
_output_shapes
:
�
clip_by_norm_1/SelectSelectclip_by_norm_1/Greaterclip_by_norm_1/Sumclip_by_norm_1/ones_like*
T0*
_output_shapes
:
W
clip_by_norm_1/SqrtSqrtclip_by_norm_1/Select*
T0*
_output_shapes
:
�
clip_by_norm_1/Select_1Selectclip_by_norm_1/Greaterclip_by_norm_1/Sqrtclip_by_norm_1/Sum*
T0*
_output_shapes
:
[
clip_by_norm_1/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
l
clip_by_norm_1/mul_1Mulgradients/AddN_10clip_by_norm_1/mul_1/y*
T0*
_output_shapes	
:�
]
clip_by_norm_1/Maximum/yConst*
dtype0*
_output_shapes
: *
valueB
 *   A
y
clip_by_norm_1/MaximumMaximumclip_by_norm_1/Select_1clip_by_norm_1/Maximum/y*
_output_shapes
:*
T0
u
clip_by_norm_1/truedivRealDivclip_by_norm_1/mul_1clip_by_norm_1/Maximum*
T0*
_output_shapes	
:�
X
clip_by_norm_1Identityclip_by_norm_1/truediv*
T0*
_output_shapes	
:�
g
clip_by_norm_2/mulMulgradients/AddN_1gradients/AddN_1*
T0*
_output_shapes
:	�
e
clip_by_norm_2/ConstConst*
valueB"       *
dtype0*
_output_shapes
:
�
clip_by_norm_2/SumSumclip_by_norm_2/mulclip_by_norm_2/Const*

Tidx0*
	keep_dims(*
T0*
_output_shapes

:
]
clip_by_norm_2/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
x
clip_by_norm_2/GreaterGreaterclip_by_norm_2/Sumclip_by_norm_2/Greater/y*
T0*
_output_shapes

:
o
clip_by_norm_2/ones_like/ShapeConst*
valueB"      *
dtype0*
_output_shapes
:
c
clip_by_norm_2/ones_like/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  �?
�
clip_by_norm_2/ones_likeFillclip_by_norm_2/ones_like/Shapeclip_by_norm_2/ones_like/Const*
T0*

index_type0*
_output_shapes

:
�
clip_by_norm_2/SelectSelectclip_by_norm_2/Greaterclip_by_norm_2/Sumclip_by_norm_2/ones_like*
T0*
_output_shapes

:
[
clip_by_norm_2/SqrtSqrtclip_by_norm_2/Select*
_output_shapes

:*
T0
�
clip_by_norm_2/Select_1Selectclip_by_norm_2/Greaterclip_by_norm_2/Sqrtclip_by_norm_2/Sum*
T0*
_output_shapes

:
[
clip_by_norm_2/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
o
clip_by_norm_2/mul_1Mulgradients/AddN_1clip_by_norm_2/mul_1/y*
T0*
_output_shapes
:	�
]
clip_by_norm_2/Maximum/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
}
clip_by_norm_2/MaximumMaximumclip_by_norm_2/Select_1clip_by_norm_2/Maximum/y*
T0*
_output_shapes

:
y
clip_by_norm_2/truedivRealDivclip_by_norm_2/mul_1clip_by_norm_2/Maximum*
T0*
_output_shapes
:	�
\
clip_by_norm_2Identityclip_by_norm_2/truediv*
_output_shapes
:	�*
T0
^
clip_by_norm_3/mulMulgradients/AddNgradients/AddN*
T0*
_output_shapes
:
^
clip_by_norm_3/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
clip_by_norm_3/SumSumclip_by_norm_3/mulclip_by_norm_3/Const*
_output_shapes
:*

Tidx0*
	keep_dims(*
T0
]
clip_by_norm_3/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
t
clip_by_norm_3/GreaterGreaterclip_by_norm_3/Sumclip_by_norm_3/Greater/y*
_output_shapes
:*
T0
h
clip_by_norm_3/ones_like/ShapeConst*
dtype0*
_output_shapes
:*
valueB:
c
clip_by_norm_3/ones_like/ConstConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
clip_by_norm_3/ones_likeFillclip_by_norm_3/ones_like/Shapeclip_by_norm_3/ones_like/Const*
T0*

index_type0*
_output_shapes
:
�
clip_by_norm_3/SelectSelectclip_by_norm_3/Greaterclip_by_norm_3/Sumclip_by_norm_3/ones_like*
_output_shapes
:*
T0
W
clip_by_norm_3/SqrtSqrtclip_by_norm_3/Select*
_output_shapes
:*
T0
�
clip_by_norm_3/Select_1Selectclip_by_norm_3/Greaterclip_by_norm_3/Sqrtclip_by_norm_3/Sum*
_output_shapes
:*
T0
[
clip_by_norm_3/mul_1/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
h
clip_by_norm_3/mul_1Mulgradients/AddNclip_by_norm_3/mul_1/y*
_output_shapes
:*
T0
]
clip_by_norm_3/Maximum/yConst*
valueB
 *   A*
dtype0*
_output_shapes
: 
y
clip_by_norm_3/MaximumMaximumclip_by_norm_3/Select_1clip_by_norm_3/Maximum/y*
T0*
_output_shapes
:
t
clip_by_norm_3/truedivRealDivclip_by_norm_3/mul_1clip_by_norm_3/Maximum*
T0*
_output_shapes
:
W
clip_by_norm_3Identityclip_by_norm_3/truediv*
T0*
_output_shapes
:
�
beta1_power/initial_valueConst*%
_class
loc:@ending/output/bias*
valueB
 *fff?*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *%
_class
loc:@ending/output/bias*
	container *
shape: 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
T0*%
_class
loc:@ending/output/bias*
validate_shape(*
_output_shapes
: *
use_locking(
q
beta1_power/readIdentitybeta1_power*
T0*%
_class
loc:@ending/output/bias*
_output_shapes
: 
�
beta2_power/initial_valueConst*%
_class
loc:@ending/output/bias*
valueB
 *w�?*
dtype0*
_output_shapes
: 
�
beta2_power
VariableV2*
shared_name *%
_class
loc:@ending/output/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*%
_class
loc:@ending/output/bias
q
beta2_power/readIdentitybeta2_power*
T0*%
_class
loc:@ending/output/bias*
_output_shapes
: 
�
Fending/rnn/sentence_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"�  �  *2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
dtype0*
_output_shapes
:
�
<ending/rnn/sentence_cell/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *2
_class(
&$loc:@ending/rnn/sentence_cell/kernel
�
6ending/rnn/sentence_cell/kernel/Adam/Initializer/zerosFillFending/rnn/sentence_cell/kernel/Adam/Initializer/zeros/shape_as_tensor<ending/rnn/sentence_cell/kernel/Adam/Initializer/zeros/Const* 
_output_shapes
:
�-�*
T0*

index_type0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel
�
$ending/rnn/sentence_cell/kernel/Adam
VariableV2*
shape:
�-�*
dtype0* 
_output_shapes
:
�-�*
shared_name *2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
	container 
�
+ending/rnn/sentence_cell/kernel/Adam/AssignAssign$ending/rnn/sentence_cell/kernel/Adam6ending/rnn/sentence_cell/kernel/Adam/Initializer/zeros*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
validate_shape(* 
_output_shapes
:
�-�*
use_locking(
�
)ending/rnn/sentence_cell/kernel/Adam/readIdentity$ending/rnn/sentence_cell/kernel/Adam*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel* 
_output_shapes
:
�-�
�
Hending/rnn/sentence_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"�  �  *2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
dtype0*
_output_shapes
:
�
>ending/rnn/sentence_cell/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
dtype0*
_output_shapes
: 
�
8ending/rnn/sentence_cell/kernel/Adam_1/Initializer/zerosFillHending/rnn/sentence_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensor>ending/rnn/sentence_cell/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
�-�*
T0*

index_type0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel
�
&ending/rnn/sentence_cell/kernel/Adam_1
VariableV2*
shared_name *2
_class(
&$loc:@ending/rnn/sentence_cell/kernel*
	container *
shape:
�-�*
dtype0* 
_output_shapes
:
�-�
�
-ending/rnn/sentence_cell/kernel/Adam_1/AssignAssign&ending/rnn/sentence_cell/kernel/Adam_18ending/rnn/sentence_cell/kernel/Adam_1/Initializer/zeros*
validate_shape(* 
_output_shapes
:
�-�*
use_locking(*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel
�
+ending/rnn/sentence_cell/kernel/Adam_1/readIdentity&ending/rnn/sentence_cell/kernel/Adam_1*
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel* 
_output_shapes
:
�-�
�
Dending/rnn/sentence_cell/bias/Adam/Initializer/zeros/shape_as_tensorConst*
valueB:�*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
dtype0*
_output_shapes
:
�
:ending/rnn/sentence_cell/bias/Adam/Initializer/zeros/ConstConst*
valueB
 *    *0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
dtype0*
_output_shapes
: 
�
4ending/rnn/sentence_cell/bias/Adam/Initializer/zerosFillDending/rnn/sentence_cell/bias/Adam/Initializer/zeros/shape_as_tensor:ending/rnn/sentence_cell/bias/Adam/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
_output_shapes	
:�
�
"ending/rnn/sentence_cell/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
	container *
shape:�
�
)ending/rnn/sentence_cell/bias/Adam/AssignAssign"ending/rnn/sentence_cell/bias/Adam4ending/rnn/sentence_cell/bias/Adam/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
validate_shape(*
_output_shapes	
:�
�
'ending/rnn/sentence_cell/bias/Adam/readIdentity"ending/rnn/sentence_cell/bias/Adam*
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
_output_shapes	
:�
�
Fending/rnn/sentence_cell/bias/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB:�*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
dtype0*
_output_shapes
:
�
<ending/rnn/sentence_cell/bias/Adam_1/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *0
_class&
$"loc:@ending/rnn/sentence_cell/bias
�
6ending/rnn/sentence_cell/bias/Adam_1/Initializer/zerosFillFending/rnn/sentence_cell/bias/Adam_1/Initializer/zeros/shape_as_tensor<ending/rnn/sentence_cell/bias/Adam_1/Initializer/zeros/Const*
T0*

index_type0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
_output_shapes	
:�
�
$ending/rnn/sentence_cell/bias/Adam_1
VariableV2*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
+ending/rnn/sentence_cell/bias/Adam_1/AssignAssign$ending/rnn/sentence_cell/bias/Adam_16ending/rnn/sentence_cell/bias/Adam_1/Initializer/zeros*
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
)ending/rnn/sentence_cell/bias/Adam_1/readIdentity$ending/rnn/sentence_cell/bias/Adam_1*
_output_shapes	
:�*
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias
�
;ending/output/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"�     *'
_class
loc:@ending/output/kernel*
dtype0*
_output_shapes
:
�
1ending/output/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *'
_class
loc:@ending/output/kernel*
dtype0*
_output_shapes
: 
�
+ending/output/kernel/Adam/Initializer/zerosFill;ending/output/kernel/Adam/Initializer/zeros/shape_as_tensor1ending/output/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@ending/output/kernel*
_output_shapes
:	�
�
ending/output/kernel/Adam
VariableV2*
shape:	�*
dtype0*
_output_shapes
:	�*
shared_name *'
_class
loc:@ending/output/kernel*
	container 
�
 ending/output/kernel/Adam/AssignAssignending/output/kernel/Adam+ending/output/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	�*
use_locking(*
T0*'
_class
loc:@ending/output/kernel
�
ending/output/kernel/Adam/readIdentityending/output/kernel/Adam*
_output_shapes
:	�*
T0*'
_class
loc:@ending/output/kernel
�
=ending/output/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"�     *'
_class
loc:@ending/output/kernel
�
3ending/output/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *'
_class
loc:@ending/output/kernel*
dtype0*
_output_shapes
: 
�
-ending/output/kernel/Adam_1/Initializer/zerosFill=ending/output/kernel/Adam_1/Initializer/zeros/shape_as_tensor3ending/output/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*'
_class
loc:@ending/output/kernel*
_output_shapes
:	�
�
ending/output/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes
:	�*
shared_name *'
_class
loc:@ending/output/kernel*
	container *
shape:	�
�
"ending/output/kernel/Adam_1/AssignAssignending/output/kernel/Adam_1-ending/output/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@ending/output/kernel*
validate_shape(*
_output_shapes
:	�
�
 ending/output/kernel/Adam_1/readIdentityending/output/kernel/Adam_1*
T0*'
_class
loc:@ending/output/kernel*
_output_shapes
:	�
�
)ending/output/bias/Adam/Initializer/zerosConst*
valueB*    *%
_class
loc:@ending/output/bias*
dtype0*
_output_shapes
:
�
ending/output/bias/Adam
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *%
_class
loc:@ending/output/bias
�
ending/output/bias/Adam/AssignAssignending/output/bias/Adam)ending/output/bias/Adam/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@ending/output/bias*
validate_shape(*
_output_shapes
:
�
ending/output/bias/Adam/readIdentityending/output/bias/Adam*
T0*%
_class
loc:@ending/output/bias*
_output_shapes
:
�
+ending/output/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *%
_class
loc:@ending/output/bias
�
ending/output/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:*
shared_name *%
_class
loc:@ending/output/bias*
	container *
shape:
�
 ending/output/bias/Adam_1/AssignAssignending/output/bias/Adam_1+ending/output/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@ending/output/bias*
validate_shape(*
_output_shapes
:
�
ending/output/bias/Adam_1/readIdentityending/output/bias/Adam_1*
T0*%
_class
loc:@ending/output/bias*
_output_shapes
:
W
Adam/learning_rateConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
5Adam/update_ending/rnn/sentence_cell/kernel/ApplyAdam	ApplyAdamending/rnn/sentence_cell/kernel$ending/rnn/sentence_cell/kernel/Adam&ending/rnn/sentence_cell/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonclip_by_norm*
use_nesterov( * 
_output_shapes
:
�-�*
use_locking( *
T0*2
_class(
&$loc:@ending/rnn/sentence_cell/kernel
�
3Adam/update_ending/rnn/sentence_cell/bias/ApplyAdam	ApplyAdamending/rnn/sentence_cell/bias"ending/rnn/sentence_cell/bias/Adam$ending/rnn/sentence_cell/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonclip_by_norm_1*
use_nesterov( *
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@ending/rnn/sentence_cell/bias
�
*Adam/update_ending/output/kernel/ApplyAdam	ApplyAdamending/output/kernelending/output/kernel/Adamending/output/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonclip_by_norm_2*
use_locking( *
T0*'
_class
loc:@ending/output/kernel*
use_nesterov( *
_output_shapes
:	�
�
(Adam/update_ending/output/bias/ApplyAdam	ApplyAdamending/output/biasending/output/bias/Adamending/output/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilonclip_by_norm_3*
use_locking( *
T0*%
_class
loc:@ending/output/bias*
use_nesterov( *
_output_shapes
:
�
Adam/mulMulbeta1_power/read
Adam/beta1)^Adam/update_ending/output/bias/ApplyAdam+^Adam/update_ending/output/kernel/ApplyAdam4^Adam/update_ending/rnn/sentence_cell/bias/ApplyAdam6^Adam/update_ending/rnn/sentence_cell/kernel/ApplyAdam*
_output_shapes
: *
T0*%
_class
loc:@ending/output/bias
�
Adam/AssignAssignbeta1_powerAdam/mul*
T0*%
_class
loc:@ending/output/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2)^Adam/update_ending/output/bias/ApplyAdam+^Adam/update_ending/output/kernel/ApplyAdam4^Adam/update_ending/rnn/sentence_cell/bias/ApplyAdam6^Adam/update_ending/rnn/sentence_cell/kernel/ApplyAdam*
T0*%
_class
loc:@ending/output/bias*
_output_shapes
: 
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
use_locking( *
T0*%
_class
loc:@ending/output/bias*
validate_shape(*
_output_shapes
: 
�
Adam/updateNoOp^Adam/Assign^Adam/Assign_1)^Adam/update_ending/output/bias/ApplyAdam+^Adam/update_ending/output/kernel/ApplyAdam4^Adam/update_ending/rnn/sentence_cell/bias/ApplyAdam6^Adam/update_ending/rnn/sentence_cell/kernel/ApplyAdam
z

Adam/valueConst^Adam/update*
_class
loc:@global_step*
value	B :*
dtype0*
_output_shapes
: 
~
Adam	AssignAddglobal_step
Adam/value*
use_locking( *
T0*
_class
loc:@global_step*
_output_shapes
: 
N
	loss/tagsConst*
valueB
 Bloss*
dtype0*
_output_shapes
: 
G
lossScalarSummary	loss/tagsMean*
T0*
_output_shapes
: 
V
accuracy/tagsConst*
valueB Baccuracy*
dtype0*
_output_shapes
: 
Q
accuracyScalarSummaryaccuracy/tagsMean_1*
T0*
_output_shapes
: 
S
Merge/MergeSummaryMergeSummarylossaccuracy*
N*
_output_shapes
: 
U
Merge_1/MergeSummaryMergeSummarylossaccuracy*
N*
_output_shapes
: �o
�*
�
*Dataset_map_<class 'functools.partial'>_64
arg0
arg1
concat_1
cast2DWrapper for passing nested structures to and from tf.data functions.�A
strided_slice/stackConst*
valueB: *
dtype0C
strided_slice/stack_1Const*
dtype0*
valueB:C
strided_slice/stack_2Const*
valueB:*
dtype0�
strided_sliceStridedSlicearg1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask *
T0*
Index0*
shrink_axis_maskB
random_uniform/shapeConst*
valueB:*
dtype0?
random_uniform/minConst*
dtype0*
valueB
 *    ?
random_uniform/maxConst*
valueB
 *  �?*
dtype0{
random_uniform/RandomUniformRandomUniformrandom_uniform/shape:output:0*
T0*
dtype0*
seed2 *

seed \
random_uniform/subSubrandom_uniform/max:output:0random_uniform/min:output:0*
T0a
random_uniform/mulMul%random_uniform/RandomUniform:output:0random_uniform/sub:z:0*
T0S
random_uniformAddrandom_uniform/mul:z:0random_uniform/min:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0�
strided_slice_1StridedSlicerandom_uniform:z:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask *
Index0*
T03
Less/yConst*
valueB
 *UUU?*
dtype0@
LessLessstrided_slice_1:output:0Less/y:output:0*
T02
cond/SwitchSwitchLess:z:0Less:z:0*
T0
=
cond/switch_tIdentitycond/Switch:output_true:0*
T0
>
cond/switch_fIdentitycond/Switch:output_false:0*
T0
+
cond/pred_idIdentityLess:z:0*
T0
�
cond/OneShotIteratorOneShotIterator^cond/switch_t*
	container *0
dataset_factoryR
_make_dataset_KVahewP2F78*
output_types	
2*
shared_name *6
output_shapes%
#:�%:�%:�%:�%:�%T
cond/IteratorToStringHandleIteratorToStringHandlecond/OneShotIterator:handle:0�
cond/IteratorGetNextIteratorGetNextcond/OneShotIterator:handle:0*
output_types	
2*6
output_shapes%
#:�%:�%:�%:�%:�%S

cond/stackPack!cond/IteratorGetNext:components:4*
N*
T0*

axis W
cond/random_uniform/shapeConst^cond/switch_f*
valueB:*
dtype0Q
cond/random_uniform/minConst^cond/switch_f*
value	B : *
dtype0Q
cond/random_uniform/maxConst^cond/switch_f*
dtype0*
value	B :�
cond/random_uniformRandomUniformInt"cond/random_uniform/shape:output:0 cond/random_uniform/min:output:0 cond/random_uniform/max:output:0*
seed2 *

Tout0*

seed *
T0L
cond/GatherV2/axisConst^cond/switch_f*
value	B : *
dtype0�
cond/GatherV2GatherV2#cond/GatherV2/Switch:output_false:0cond/random_uniform:output:0cond/GatherV2/axis:output:0*
Tindices0*
Tparams0*
Taxis0]
cond/GatherV2/SwitchSwitcharg0cond/pred_id:output:0*
T0*
_class
	loc:@arg0R

cond/MergeMergecond/GatherV2:output:0cond/stack:output:0*
T0*
N8
ExpandDims/dimConst*
value	B : *
dtype0^

ExpandDims
ExpandDimsstrided_slice:output:0ExpandDims/dim:output:0*

Tdim0*
T05
concat/axisConst*
value	B : *
dtype0p
concatConcatV2ExpandDims:output:0cond/Merge:output:0concat/axis:output:0*
T0*
N*

Tidx0:
one_hot/on_valueConst*
value	B :*
dtype0;
one_hot/off_valueConst*
value	B : *
dtype09
one_hot/indicesConst*
dtype0	*
value	B	 R 7
one_hot/depthConst*
value	B :*
dtype0�
one_hotOneHotone_hot/indices:output:0one_hot/depth:output:0one_hot/on_value:output:0one_hot/off_value:output:0*
T0*
axis���������*
TI0	5
range/startConst*
dtype0*
value	B : 5
range/limitConst*
value	B :*
dtype05
range/deltaConst*
value	B :*
dtype0\
rangeRangerange/start:output:0range/limit:output:0range/delta:output:0*

Tidx0P
RandomShuffleRandomShufflerange:output:0*

seed *
T0*
seed2 7
GatherV2/axisConst*
value	B : *
dtype0�
GatherV2GatherV2concat:output:0RandomShuffle:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams09
GatherV2_1/axisConst*
value	B : *
dtype0�

GatherV2_1GatherV2one_hot:output:0RandomShuffle:output:0GatherV2_1/axis:output:0*
Tparams0*
Taxis0*
Tindices0:
ArgMax/dimensionConst*
value	B : *
dtype0h
ArgMaxArgMaxGatherV2_1:output:0ArgMax/dimension:output:0*
output_type0	*

Tidx0*
T0E
CastCastArgMax:output:0*

SrcT0	*
Truncate( *

DstT07
concat_1/axisConst*
dtype0*
value	B : e

concat_1_0ConcatV2arg0GatherV2:output:0concat_1/axis:output:0*
T0*
N*

Tidx0"
castCast:y:0"
concat_1concat_1_0:output:0
�
�
"Dataset_flat_map_read_one_file_151
arg0
tfrecorddataset2DWrapper for passing nested structures to and from tf.data functions.�9
compression_typeConst*
valueB B *
dtype07
buffer_sizeConst*
valueB		 R��*
dtype0	Y
TFRecordDatasetTFRecordDatasetarg0compression_type:output:0buffer_size:output:0"+
tfrecorddatasetTFRecordDataset:handle:0
�
�
Dataset_map_tensorize_dict_172
arg0
arg1
arg2
arg3
arg4
arg5	
stack2DWrapper for passing nested structures to and from tf.data functions.Q
stack_0Packarg0arg1arg2arg3arg4arg5*
T0*

axis *
N"
stackstack_0:output:0
�
�
Dataset_map_extract_fn_35
arg0)
%parsesingleexample_parsesingleexample+
'parsesingleexample_parsesingleexample_0+
'parsesingleexample_parsesingleexample_1+
'parsesingleexample_parsesingleexample_2+
'parsesingleexample_parsesingleexample_32DWrapper for passing nested structures to and from tf.data functions.A
ParseSingleExample/ConstConst*
valueB *
dtype0C
ParseSingleExample/Const_1Const*
valueB *
dtype0C
ParseSingleExample/Const_2Const*
dtype0*
valueB C
ParseSingleExample/Const_3Const*
valueB *
dtype0C
ParseSingleExample/Const_4Const*
dtype0*
valueB �
%ParseSingleExample/ParseSingleExampleParseSingleExamplearg0!ParseSingleExample/Const:output:0#ParseSingleExample/Const_1:output:0#ParseSingleExample/Const_2:output:0#ParseSingleExample/Const_3:output:0#ParseSingleExample/Const_4:output:0*
Tdense	
2*

num_sparse *G

dense_keys9
7	sentence1	sentence2	sentence3	sentence4	sentence5*5
dense_shapes%
#:�%:�%:�%:�%:�%*
sparse_types
 *
sparse_keys
 "_
'parsesingleexample_parsesingleexample_24ParseSingleExample/ParseSingleExample:dense_values:3"_
'parsesingleexample_parsesingleexample_34ParseSingleExample/ParseSingleExample:dense_values:4"]
%parsesingleexample_parsesingleexample4ParseSingleExample/ParseSingleExample:dense_values:0"_
'parsesingleexample_parsesingleexample_04ParseSingleExample/ParseSingleExample:dense_values:1"_
'parsesingleexample_parsesingleexample_14ParseSingleExample/ParseSingleExample:dense_values:2
�
Q
_make_dataset_KVahewP2F78
modeldataset2Factory function for a dataset.�{
optimizationsConst*V
valueMBKBmap_and_batch_fusionBnoop_eliminationBshuffle_and_repeat_fusion*
dtype0�
TensorSliceDataset/ConstConst*�
value�B~ Bx/Users/arthur/Documents/natural_lanugage_understanding/nlu_project2/data/processed/train_stories_skip_thoughts.tfrecords*
dtype0^
'TensorSliceDataset/flat_filenames/shapeConst*
dtype0*
valueB:
����������
!TensorSliceDataset/flat_filenamesReshape!TensorSliceDataset/Const:output:00TensorSliceDataset/flat_filenames/shape:output:0*
T0*
Tshape0�
TensorSliceDatasetTensorSliceDataset*TensorSliceDataset/flat_filenames:output:0*
output_shapes
: *
Toutput_types
2�
FlatMapDatasetFlatMapDatasetTensorSliceDataset:handle:0*
output_types
2*

Targuments
 *
output_shapes
: *)
f$R"
 Dataset_flat_map_read_one_file_4�

MapDataset
MapDatasetFlatMapDataset:handle:0*
output_types	
2*
use_inter_op_parallelism(*

Targuments
 *
preserve_cardinality( *6
output_shapes%
#:�%:�%:�%:�%:�%*"
fR
Dataset_map_extract_fn_10�
OptimizeDatasetOptimizeDatasetMapDataset:handle:0optimizations:output:0*6
output_shapes%
#:�%:�%:�%:�%:�%*
output_types	
2�
ModelDatasetModelDatasetOptimizeDataset:handle:0*
output_types	
2*6
output_shapes%
#:�%:�%:�%:�%:�%"%
modeldatasetModelDataset:handle:0
�
�
 Dataset_flat_map_read_one_file_4
arg0
tfrecorddataset2DWrapper for passing nested structures to and from tf.data functions.�9
compression_typeConst*
dtype0*
valueB B 7
buffer_sizeConst*
dtype0	*
valueB		 R��Y
TFRecordDatasetTFRecordDatasetarg0compression_type:output:0buffer_size:output:0"+
tfrecorddatasetTFRecordDataset:handle:0
�
�
!Dataset_flat_map_read_one_file_29
arg0
tfrecorddataset2DWrapper for passing nested structures to and from tf.data functions.�9
compression_typeConst*
valueB B *
dtype07
buffer_sizeConst*
valueB		 R��*
dtype0	Y
TFRecordDatasetTFRecordDatasetarg0compression_type:output:0buffer_size:output:0"+
tfrecorddatasetTFRecordDataset:handle:0
�
�
Dataset_map_extract_fn_157
arg0)
%parsesingleexample_parsesingleexample+
'parsesingleexample_parsesingleexample_0+
'parsesingleexample_parsesingleexample_1+
'parsesingleexample_parsesingleexample_2+
'parsesingleexample_parsesingleexample_3+
'parsesingleexample_parsesingleexample_42DWrapper for passing nested structures to and from tf.data functions.A
ParseSingleExample/ConstConst*
valueB *
dtype0C
ParseSingleExample/Const_1Const*
valueB *
dtype0C
ParseSingleExample/Const_2Const*
valueB *
dtype0C
ParseSingleExample/Const_3Const*
valueB *
dtype0C
ParseSingleExample/Const_4Const*
valueB *
dtype0C
ParseSingleExample/Const_5Const*
valueB *
dtype0�
%ParseSingleExample/ParseSingleExampleParseSingleExamplearg0!ParseSingleExample/Const:output:0#ParseSingleExample/Const_1:output:0#ParseSingleExample/Const_2:output:0#ParseSingleExample/Const_3:output:0#ParseSingleExample/Const_4:output:0#ParseSingleExample/Const_5:output:0*

num_sparse *N

dense_keys@
>ending1ending2	sentence1	sentence2	sentence3	sentence4*
sparse_types
 *<
dense_shapes,
*:�%:�%:�%:�%:�%:�%*
sparse_keys
 *
Tdense

2"_
'parsesingleexample_parsesingleexample_04ParseSingleExample/ParseSingleExample:dense_values:1"_
'parsesingleexample_parsesingleexample_14ParseSingleExample/ParseSingleExample:dense_values:2"_
'parsesingleexample_parsesingleexample_24ParseSingleExample/ParseSingleExample:dense_values:3"_
'parsesingleexample_parsesingleexample_34ParseSingleExample/ParseSingleExample:dense_values:4"_
'parsesingleexample_parsesingleexample_44ParseSingleExample/ParseSingleExample:dense_values:5"]
%parsesingleexample_parsesingleexample4ParseSingleExample/ParseSingleExample:dense_values:0
�
�
Dataset_map_extract_fn_10
arg0)
%parsesingleexample_parsesingleexample+
'parsesingleexample_parsesingleexample_0+
'parsesingleexample_parsesingleexample_1+
'parsesingleexample_parsesingleexample_2+
'parsesingleexample_parsesingleexample_32DWrapper for passing nested structures to and from tf.data functions.A
ParseSingleExample/ConstConst*
valueB *
dtype0C
ParseSingleExample/Const_1Const*
valueB *
dtype0C
ParseSingleExample/Const_2Const*
valueB *
dtype0C
ParseSingleExample/Const_3Const*
dtype0*
valueB C
ParseSingleExample/Const_4Const*
valueB *
dtype0�
%ParseSingleExample/ParseSingleExampleParseSingleExamplearg0!ParseSingleExample/Const:output:0#ParseSingleExample/Const_1:output:0#ParseSingleExample/Const_2:output:0#ParseSingleExample/Const_3:output:0#ParseSingleExample/Const_4:output:0*

num_sparse *G

dense_keys9
7	sentence1	sentence2	sentence3	sentence4	sentence5*5
dense_shapes%
#:�%:�%:�%:�%:�%*
sparse_types
 *
sparse_keys
 *
Tdense	
2"]
%parsesingleexample_parsesingleexample4ParseSingleExample/ParseSingleExample:dense_values:0"_
'parsesingleexample_parsesingleexample_04ParseSingleExample/ParseSingleExample:dense_values:1"_
'parsesingleexample_parsesingleexample_14ParseSingleExample/ParseSingleExample:dense_values:2"_
'parsesingleexample_parsesingleexample_24ParseSingleExample/ParseSingleExample:dense_values:3"_
'parsesingleexample_parsesingleexample_34ParseSingleExample/ParseSingleExample:dense_values:4
�
�
,Dataset_map_split_skip_thoughts_sentences_48
arg0
arg1
arg2
arg3
arg4
packed_6
packed_72DWrapper for passing nested structures to and from tf.data functions.D
packedPackarg0arg1arg2arg3*
N*
T0*

axis 4
packed_1Packarg4*
T0*

axis *
NF
packed_2Packarg0arg1arg2arg3*
T0*

axis *
NF
packed_3Packarg0arg1arg2arg3*
T0*

axis *
N4
packed_4Packarg4*
T0*

axis *
N4
packed_5Packarg4*
N*
T0*

axis H

packed_6_0Packarg0arg1arg2arg3*
T0*

axis *
N6

packed_7_0Packarg4*
T0*

axis *
N"
packed_7packed_7_0:output:0"
packed_6packed_6_0:output:0""#
	summaries

loss:0

accuracy:0"�
trainable_variables��
�
!ending/rnn/sentence_cell/kernel:0&ending/rnn/sentence_cell/kernel/Assign&ending/rnn/sentence_cell/kernel/read:02<ending/rnn/sentence_cell/kernel/Initializer/random_uniform:08
�
ending/rnn/sentence_cell/bias:0$ending/rnn/sentence_cell/bias/Assign$ending/rnn/sentence_cell/bias/read:021ending/rnn/sentence_cell/bias/Initializer/zeros:08
�
ending/output/kernel:0ending/output/kernel/Assignending/output/kernel/read:021ending/output/kernel/Initializer/random_uniform:08
v
ending/output/bias:0ending/output/bias/Assignending/output/bias/read:02&ending/output/bias/Initializer/zeros:08"
train_op

Adam"�
	variables��
�
!ending/rnn/sentence_cell/kernel:0&ending/rnn/sentence_cell/kernel/Assign&ending/rnn/sentence_cell/kernel/read:02<endin