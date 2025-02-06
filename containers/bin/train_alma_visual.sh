sbatch \
    -n 16 \
    --mem-per-cpu=1000 \
    --gpus=rtx_4090:1 \
    --time=1440 \
    --wrap="\
    bash /cluster/home/anqiali/MasterThesis/vq_bet_official/./examples/train_alma_visual.py \
    save_path=/cluster/home/anqiali/MasterThesis/checkpoints/oven_transformer/2024-09-20/09-44-30 \
    batch_size=300 \
    epochs=200 \
    action_window_size=20 \
    window_size=50 \
    noise_enhance_coef=0.05 \
    model.vqvae_model.vqvae_groups=2 \
    model.vqvae_model.vqvae_n_embed=16 \
    model.vqvae_model.n_latent_dims=512 \
    data.data_directory=/cluster/scratch/anqiali/data_pt/data_image.h5 \
    model.res_iter=true \
    model.uniformly_downsample=10 \
    vqvae_load_dir=/cluster/home/anqiali/MasterThesis/checkpoints/oven_vq_vae/2024-09-18/11-40-24/pretrain_oven/resstate/trained_vqvae.pt \
    wandb.run_name=transformer_visual/resstate_a20_normalize_image_noise"
