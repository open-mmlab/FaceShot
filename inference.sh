python inference_opendomain.py \
--video_path="demo/videos/01-01-08-02-01-01-19.mp4" \
--img_path="demo/images/000001.jpg" \
--ckpt_dir="ckpts/mofa/ldmk_controlnet" \
--save_root="results" \
--max_frame_len=125 \
--ldmk_render="retargeting"