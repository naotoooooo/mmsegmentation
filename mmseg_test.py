# from mmseg.apis import inference_model, init_model, show_result_pyplot, MMSegInferencer
# import mmcv

# #モデルの選択、および学習済みモデルの呼び出し
# config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
# checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# # build the model from a config file and a checkpoint file
# model = init_model(config_file, checkpoint_file, device='cuda:0')

# # test a single image and show the results
# imgs = 'data/dkan_outdoor_2024-11-08-06-03-31/rgb'  # or img = mmcv.imread(img), which will only load it once
# result = inference_model(model, imgs)
# # visualize the results in a new window
# show_result_pyplot(model, imgs, result, show=True,wait_time=0.5)
# # or save the visualization results to image files
# # you can change the opacity of the painted segmentation map in (0, 1].
# show_result_pyplot(model, imgs, result, show=True, out_file='result.jpg', opacity=1.0)
# # test a video and show the results
# video = mmcv.VideoReader('video.mp4')
# for frame in video:
#    result = inference_model(model, frame)
#    show_result_pyplot(model, frame, result, wait_time=1)


#決められた枚数ごとにセマセグを行い結果を保存するプログラム
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv
import os
import shutil

# Model configuration and checkpoint file paths
config_file = 'configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py'
checkpoint_file = 'mask2former_r50_8xb2-90k_cityscapes-512x1024_20221202_140802-ffd9d750.pth'

# Initialize the model
model = init_model(config_file, checkpoint_file, device='cuda:0')

# Path to the directory of images to process
img_dir = 'data/dkan_outdoor_2024-11-08-06-03-31/rgb/'

# # Directory to save results
output_dir = 'output_results'
# Clear output_dir if it exists, then recreate it
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Remove all contents of the directory
os.makedirs(output_dir, exist_ok=True)


# Counter for processing images
counter = 0

# Loop through each image in the directory
for img_name in sorted(os.listdir(img_dir)):
    img_path = os.path.join(img_dir, img_name)
    
    # Perform inference on the image
    result = inference_model(model, img_path)

    # Increment counter
    counter += 1

    # Display every 50th image
    if counter % 50 == 0:
        output_path_05 = os.path.join(output_dir, f'result_{img_name.split(".")[0]}_opacity0.5.jpg')
        show_result_pyplot(model, img_path, result, show=False, opacity=0.5,out_file=output_path_05)
        output_path_09 = os.path.join(output_dir, f'result_{img_name.split(".")[0]}_opacity0.9.jpg')
        show_result_pyplot(model, img_path, result, show=False, opacity=0.9,out_file=output_path_09)

    
    # # Save the visualization result
    # output_path = os.path.join(output_dir, f'result_{img_name}')
    # show_result_pyplot(model, img_path, result, out_file=output_path, opacity=0.7,wait_time=0.1)

print(f"Processing complete. Results saved to {output_dir}.")
