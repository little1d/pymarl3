# linux
CUDA_VISIBLE_DEVICES="0" python src/main.py --config=hpn_qmix --env-config=codertask obs_last_action=False runner=parallel batch_size_run=8 buffer_size=5000 t_max=10050000 batch_size=128

# windows
python src\main.py --config=hpn_qmix --env-config=codertask  obs_last_action=False runner=parallel batch_size_run=8 buffer_size=5000 t_max=10050000 batch_size=128