scripts="adam_op_test.py    expanddims_squeeze_op_test.py  relu_op_test.py \
blobs_queue_db_test.py      fc_op_test.py                  reshape_op_test.py \
channel_shuffle_op_test.py  shape_op_test.py \
concat_split_op_test.py     leaky_relu_op_test.py          sigmoid_op_test.py \
convfusion_op_test.py       LRN_op_test.py                 softmax_op_test.py \
conv_op_test.py             moment_sgd_op_test.py          spatial_bn_op_test.py \
conv_transpose_test.py      operator_fallback_op_test.py \
copy_op_test.py             order_switch_op_test.py  \
dropout_op_test.py          pool_op_test.py                transpose_op_test.py \
elementwise_sum_op_test.py  pre_convert_test.py            weightedsum_op_test.py"

base="caffe2/python/ideep/"

for s in $scripts; do
    echo -e "Testing $s\c"
    if python $base$s 2>&1 | grep -q OK; then
        echo -e " - \e[32mPASSED\e[39m"
    else
        echo -e " - \e[31mFAILED\e[39m"
    fi
done