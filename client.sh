#! /bin/bash -e

IMAGE='deepclean-prod:client-20.07'
CHANNELS=( $(cat channels.txt) )
DATA_DIR='/dev/shm/llhoft/H1'
FILE_PATTERN='H-H1_llhoft-{}-1.gwf'
CMD="
    python client.py \
        --url 34.82.145.3 \
            --model-name gwe2e \
            --model-version 1 \
            --sequence-id 1001 \
            --kernel-stride 0.002 \
            --witness-h-data-dir $DATA_DIR \
            --witness-l-data-dir $DATA_DIR \
            --strain-data-dir $DATA_DIR \
            --witness-h-file-pattern $FILE_PATTERN \
            --witness-l-file-pattern $FILE_PATTERN \
            --strain-file-pattern $FILE_PATTERN \
            --witness-h-channels ${CHANNELS[@]:1} \
            --witness-l-channels ${CHANNELS[@]:1} \
            --strain-channels ${CHANNELS[@]:0:2} \
"
echo "Running command:"
echo $CMD

singularity exec \
    --home $PWD:/srv \
    --pwd /srv \
    --bind $PWD/stillwater:/opt/stillwater \
    --bind /cvmfs \
    --scratch /var/tmp \
    --scratch /tmp \
    --pid \
    /cvmfs/singularity.opensciencegrid.org/alec.gunny/$IMAGE \
        /bin/bash -c "source activate deepclean && $CMD"
