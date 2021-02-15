#! /bin/bash -e

while getopts ":u:m:s:k:dh" opt; do
    case ${opt} in
        u )
            url=${OPTARG}
            ;;
        m )
            model=${OPTARG}
            ;;
        s )
            seqid=${OPTARG}
            ;;
        k )
            kstride=${OPTARG}
            ;;
        d )
            dummy="--use-dummy"
            ;;
        h )
            echo "Run a streaming inference client"
            echo "Options:"
            echo "--------"
            echo "    -u:    Server URL. Required."
            echo "    -m:    Model name. Defaults to 'gwe2e'"
            echo "    -s:    Sequence ID used to identify this stream. Defaults to 1001"
            echo "    -k:    Kernel stride in seconds. Defaults to 0.002"
            echo "    -d:    Use dummy data"
            echo "    -h:    Display this help"
            exit 0
            ;;
        \? )
            echo "Unrecognized argument ${opt}"
            exit 1
    esac
done
shift $((OPTIND -1))

model=${model:-gwe2e}
seqid=${seqid:-1001}
kstride=${kstride:-0.002}
if [[ -z ${url} ]]; then
    echo "Must specify server url"
    exit 1
fi

CHANNELS=( $(cat channels.txt) )
DATA_DIR='/dev/shm/llhoft/H1'
FILE_PATTERN='H-H1_llhoft-{}-1.gwf'

python client.py \
    --url ${url} \
    --model-name ${model} \
    --model-version 1 \
    --sequence-id ${seqid} \
    --kernel-stride ${kstride} \
    --witness-h-data-dir $DATA_DIR \
    --witness-l-data-dir $DATA_DIR \
    --strain-data-dir $DATA_DIR \
    --witness-h-file-pattern $FILE_PATTERN \
    --witness-l-file-pattern $FILE_PATTERN \
    --strain-file-pattern $FILE_PATTERN \
    --witness-h-channels ${CHANNELS[@]:1} \
    --witness-l-channels ${CHANNELS[@]:1} \
    --strain-channels ${CHANNELS[@]:0:2} \
    --num-iterations 10000 \
    ${dummy}
