# linux
nohup CUDA_VISIBLE_DEVICES="0" python -u experiments.py --env codertask --config hpn_qmix  config qmix --params1 mixer=qmix  > ./pymarl_log716.log 2>&1 &
# windows
 python -u experiments.py --env codertask --config hpn_qmix  config qmix --params1 mixer=qmix  > ./pymarl_log716.log 2>&1 &