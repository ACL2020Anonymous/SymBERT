DATA128_DIR="data/wiki-zh-bert-pretrain-3clssmooth"
DATA512_DIR="data/wiki-zh-bert-pretrain-3clssmooth-512"
EXPNAME="wiki-zh-bert-pretrain-3clssmooth"
MODEL_DIR="models/${EXPNAME}"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LD_LIBRARY_PATH=/usr/local/cuda-10.0/extras/CUPTI/lib64:/usr/local/cuda-10.0/lib64:/usr/local/nvidia/lib64/:/usr/local/nccl_2.4.7/lib mpirun --allow-run-as-root --machinefile host --map-by ppr:8:node -x CUDA_VISIBLE_DEVICES -x HOROVOD_MPI_THREADS_DISABLE=1 -x HOROVOD_STALL_CHECK_DISABLE=1 --mca coll_tuned_use_dynamic_rules 1 --mca coll_tuned_allgatherv_algorithm 3 --mca coll_tuned_allgather_algorithm 4 --mca btl_openib_cuda_async_recv false --mca btl_openib_rroce_enable 1  --mca btl_openib_want_cuda_gdr 0 --mca btl_openib_cpc_include rdmacm --mca mpi_warn_on_fork false -mca plm_rsh_args "-p 12345" -x LD_LIBRARY_PATH -x NCCL_DEBUG=INFO -x NCCL_CHECKS_DISABLE=1 -x NCCL_SHM_DISABLE=0 -x NCCL_SOCKET_IFNAME=^lo,docker0 -mca btl_tcp_if_exclude lo,docker0 -x NCCL_IB_DISABLE=0 -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_CUDA_SUPPORT=1 --mca orte_base_help_aggregate 0 -mca pml ob1 python run_pretraining.py \
  --input_file="${DATA128_DIR}/*.tfrecord" \
  --eval_file="${DATA128_DIR}/0.tfrecord" \
  --output_dir=${MODEL_DIR} \
  --do_train=True \
  --do_eval=False \
  --train_batch_size=128 \
  --eval_batch_size=8 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=170000 \
  --num_warmup_steps=10000 \
  --learning_rate=5e-5 \
  --save_checkpoints_steps=10000 \
  --report_loss \
  --bert_config_file="models/chinese_L-12_H-768_A-12/bert_config.json" \
  --random_seed=0 \
  --use_horovod=true \
  >> ${EXPNAME}-128.log 2>&1 &

wait

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 LD_LIBRARY_PATH=/usr/local/cuda-10.0/extras/CUPTI/lib64:/usr/local/cuda-10.0/lib64:/usr/local/nvidia/lib64/:/usr/local/nccl_2.4.7/lib mpirun --allow-run-as-root --machinefile host --map-by ppr:8:node -x CUDA_VISIBLE_DEVICES -x HOROVOD_MPI_THREADS_DISABLE=1 -x HOROVOD_STALL_CHECK_DISABLE=1 --mca coll_tuned_use_dynamic_rules 1 --mca coll_tuned_allgatherv_algorithm 3 --mca coll_tuned_allgather_algorithm 4 --mca btl_openib_cuda_async_recv false --mca btl_openib_rroce_enable 1  --mca btl_openib_want_cuda_gdr 0 --mca btl_openib_cpc_include rdmacm --mca mpi_warn_on_fork false -mca plm_rsh_args "-p 12345" -x LD_LIBRARY_PATH -x NCCL_DEBUG=INFO -x NCCL_CHECKS_DISABLE=1 -x NCCL_SHM_DISABLE=0 -x NCCL_SOCKET_IFNAME=^lo,docker0 -mca btl_tcp_if_exclude lo,docker0 -x NCCL_IB_DISABLE=0 -x NCCL_IB_GID_INDEX=3 -x NCCL_IB_CUDA_SUPPORT=1 --mca orte_base_help_aggregate 0 -mca pml ob1 python run_pretraining.py \
  --input_file="${DATA512_DIR}/*" \
  --eval_file="${DATA512_DIR}/0.tfrecord" \
  --output_dir=${MODEL_DIR} \
  --do_train=True \
  --do_eval=False \
  --train_batch_size=32 \
  --eval_batch_size=8 \
  --max_seq_length=512 \
  --max_predictions_per_seq=80 \
  --num_train_steps=245000 \
  --num_warmup_steps=0 \
  --learning_rate=3e-5 \
  --save_checkpoints_steps=10000 \
  --report_loss \
  --bert_config_file="models/chinese_L-12_H-768_A-12/bert_config.json" \
  --random_seed=0 \
  --use_horovod=true \
  >> ${EXPNAME}-512.log 2>&1 &
