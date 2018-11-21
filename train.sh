MAX_EPOCHS=25
SIZE_EPOCH=80000

while [[ $EPOCH -le $MAX_EPOCHS ]]
do
    echo "$EPOCH"
    echo 'start training'
    if [[ $EPOCH -eq 1 ]]
    then
        let STEP=EPOCH*SIZE_EPOCH
        python trainAttentionMask.py 0 attentionMask-8-128 --init_weights ResNet-50-model.caffemodel --step $SIZE_EPOCH
    else
        let STEP_OLD=(EPOCH-1)*SIZE_EPOCH
        let STEP=EPOCH*SIZE_EPOCH
        python trainAttentionMask.py 0 attentionMask-8-128 --restore attentionMask-8-128_iter_$STEP_OLD.solverstate --step $SIZE_EPOCH
    fi
    echo 'training done'
    echo 'start validation'
    python testAttentionMask.py 0 attentionMask-8-128 --init_weights attentionMask-8-128_iter_$STEP.caffemodel --dataset train2014
    echo 'validation done'
    echo 'start evaluation'
    echo "$EPOCH" >> trainEval.txt
    python evalCOCO.py attentionMask-8-128 --dataset train2014 --useSegm True >> trainEval.txt
    echo 'evaluation done'
    let EPOCH=EPOCH+1
done
