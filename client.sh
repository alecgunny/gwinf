#! /bin/bash -e


IMAGE='deepclean-prod:client-20.07'
singularity exec \
    --home $PWD:/srv \
    --pwd /srv \
    --bind $PWD/stillwater:/opt/stillwater \
    --bind /cvmfs \
    --scratch /var/tmp \
    --scratch /tmp \
    --pid \
    /cvmfs/singularity.opensciencegrid.org/alec.gunny/$IMAGE \
        python client.py \
            --url 34.82.145.3 \
            --model-name gwe2e \
            --model-version 1 \
            --sequence-id 1001 \
            --kernel-stride 0.002 \
            --witness-h-data-dir "/dev/shm/llhoft/H1" \
            --witness-l-data-dir "/dev/shm/llhoft/H1" \
            --strain-data-dir "/dev/shm/llhoft/H1" \
            --witness-h-file-pattern "H-H1_llhoft-{}-1.gwf" \
            --witness-l-file-pattern "H-H1_llhoft-{}-1.gwf" \
            --strain-file-pattern "H-H1_llhoft-{}-1.gwf" \
            --witness-h-channels \
                H1:PEM-CS_MAINSMON_EBAY_1_DQ \
                H1:ASC-INP1_P_INMON \
                H1:ASC-INP1_Y_INMON \
                H1:ASC-MICH_P_INMON \
                H1:ASC-MICH_Y_INMON \
                H1:ASC-PRC1_P_INMON \
                H1:ASC-PRC1_Y_INMON \
                H1:ASC-PRC2_P_INMON \
                H1:ASC-PRC2_Y_INMON \
                H1:ASC-SRC1_P_INMON \
                H1:ASC-SRC1_Y_INMON \
                H1:ASC-SRC2_P_INMON \
                H1:ASC-SRC2_Y_INMON \
                H1:ASC-DHARD_P_INMON \
                H1:ASC-DHARD_Y_INMON \
                H1:ASC-CHARD_P_INMON \
                H1:ASC-CHARD_Y_INMON \
                H1:ASC-DSOFT_P_INMON \
                H1:ASC-DSOFT_Y_INMON \
                H1:ASC-CSOFT_P_INMON \
                H1:ASC-CSOFT_Y_INMON \
            --witness-l-channels \
                H1:PEM-CS_MAINSMON_EBAY_1_DQ \
                H1:ASC-INP1_P_INMON \
                H1:ASC-INP1_Y_INMON \
                H1:ASC-MICH_P_INMON \
                H1:ASC-MICH_Y_INMON \
                H1:ASC-PRC1_P_INMON \
                H1:ASC-PRC1_Y_INMON \
                H1:ASC-PRC2_P_INMON \
                H1:ASC-PRC2_Y_INMON \
                H1:ASC-SRC1_P_INMON \
                H1:ASC-SRC1_Y_INMON \
                H1:ASC-SRC2_P_INMON \
                H1:ASC-SRC2_Y_INMON \
                H1:ASC-DHARD_P_INMON \
                H1:ASC-DHARD_Y_INMON \
                H1:ASC-CHARD_P_INMON \
                H1:ASC-CHARD_Y_INMON \
                H1:ASC-DSOFT_P_INMON \
                H1:ASC-DSOFT_Y_INMON \
                H1:ASC-CSOFT_P_INMON \
                H1:ASC-CSOFT_Y_INMON \
            --strain-channels \
                H1:GDS-CALIB_STRAIN \
                H1:PEM-CS_MAINSMON_EBAY_1_DQ \