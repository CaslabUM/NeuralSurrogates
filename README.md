Codes to accompany CiSE submission Predictivity and Utility of Neural Surrogates of Multiscale PDEs by Karthik Duraisamy, University of Michigan

#Typical run script for KS FNO
#Download ks1d data from pdearena

python ./ks_fno_train_fast5min.py --data_dir data_ks1d_phlippe --device mps --timeout_sec 1e9 --epochs 5 --batch_size 128 --lr 5e-5 --modes 16 --width 64 --depth 4 --max_train_traj 1024 --resume outputs_mps/fno1d.pt --outdir outputs_mps

python ./ks_fno_make_plots.py --ckpt outputs_mps/fno1d.pt --data_dir data_ks1d_phlippe --variant fixed --split test --traj_idx 0 --rollout_steps 80 --plot_times 0,1,2,5,10,20,40,80 --outdir outputs_mps

python3 qoi_plots.py

# Typical run script for Spectral bias. Pure numpy. No pytorch

python3 spectral_bias_demo.py


------
#Typical installation of dependencies for KS FNO
conda create -n mymps python=3.11 -y
conda activate mymps
conda install -y -c pytorch -c conda-forge pytorch torchvision torchaudio numpy h5py
conda install -y -c conda-forge matplotlib
