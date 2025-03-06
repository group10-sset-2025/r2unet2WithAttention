from flask import Flask, request, render_template, send_file, jsonify
import os 
from io import BytesIO
import nibabel as nib
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from attendion_net import R2UNet3DWithAttention
from nilearn import plotting, image
import matplotlib.pyplot as plt
from zipfile import ZipFile


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = R2UNet3DWithAttention(in_channels=3, num_classes=4, recur_num=2).to(device)
model.load_state_dict(torch.load('./models/r2_unet3d_model_epoch_50_attention.pth', map_location=device))
model.eval()

label_dict = {
    0: "Not Tumor",
    1: "Necrotic/Non-enhancing tumor core",
    2: "Peritumoral edema",
    3: "Enhancing tumor"
}
# Helper functions
def load_nii(file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    return data, img.affine

def normalize_image(image):
    scaler = MinMaxScaler()
    orig_shape = image.shape
    norm = scaler.fit_transform(image.reshape(-1, 1)).reshape(orig_shape)
    return norm

def save_prediction_as_nii(prediction, affine, output_path):
    nii = nib.Nifti1Image(prediction.astype(np.uint8), affine)
    nib.save(nii, output_path)

def generate_segmentation_overlay(brain_img_path, segmentation_path, output_png_path):
    brain_img = image.load_img(brain_img_path)
    seg_img = image.load_img(segmentation_path)

    if not (brain_img.affine == seg_img.affine).all():
        seg_img = image.resample_to_img(seg_img, brain_img)

    display = plotting.plot_roi(seg_img, bg_img=brain_img, title="Predicted Segmentation", display_mode='ortho')
    display.savefig(output_png_path)
    display.close()

def save_aligned_image(brain_img, segmentation, output_path):
    slice_idx = brain_img.shape[2] // 2
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(brain_img[:, :, slice_idx], cmap='gray')
    seg_overlay = ax.imshow(segmentation[:, :, slice_idx], cmap='jet', alpha=0.4)
    cbar = plt.colorbar(seg_overlay, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels([label_dict[i] for i in range(4)])
    ax.set_title("Brain Image with Segmentation")
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def send_files_as_zip(file_paths, zip_name):
    buffer = BytesIO()
    with ZipFile(buffer, 'w') as zip_file:
        for file_path in file_paths:
            zip_file.write(file_path, os.path.basename(file_path))
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=zip_name, mimetype='application/zip')

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')



from flask import send_from_directory

@app.route('/upload', methods=['POST'])
def upload():
    if 'flair' not in request.files or 't1ce' not in request.files or 't2' not in request.files:
        return "Please upload all three NIfTI files (flair, t1ce, t2)", 400

    flair_file = request.files['flair']
    t1ce_file = request.files['t1ce']
    t2_file = request.files['t2']

    flair_path = os.path.join(app.config['UPLOAD_FOLDER'], flair_file.filename)
    t1ce_path = os.path.join(app.config['UPLOAD_FOLDER'], t1ce_file.filename)
    t2_path = os.path.join(app.config['UPLOAD_FOLDER'], t2_file.filename)

    flair_file.save(flair_path)
    t1ce_file.save(t1ce_path)
    t2_file.save(t2_path)

    # Processing
    flair_img, affine = load_nii(flair_path)
    t1ce_img, _ = load_nii(t1ce_path)
    t2_img, _ = load_nii(t2_path)

    flair_norm = normalize_image(flair_img)
    t1ce_norm = normalize_image(t1ce_img)
    t2_norm = normalize_image(t2_img)

    # Crop to required size
    flair_cropped = flair_norm[56:184, 56:184, 13:141]
    t1ce_cropped = t1ce_norm[56:184, 56:184, 13:141]
    t2_cropped = t2_norm[56:184, 56:184, 13:141]

    input_image = np.stack([flair_cropped, t1ce_cropped, t2_cropped], axis=-1)
    input_image = np.transpose(input_image, (3, 0, 1, 2))
    input_tensor = torch.tensor(input_image, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1)
        pred_np = pred.cpu().numpy().squeeze()

    full_segmentation = np.zeros(flair_img.shape, dtype=pred_np.dtype)
    full_segmentation[56:184, 56:184, 13:141] = pred_np

    result_nii_path = os.path.join(app.config['RESULT_FOLDER'], 'predicted_segmentation.nii')
    result_png_path = os.path.join(app.config['RESULT_FOLDER'], 'predicted_segmentation.png')

    save_prediction_as_nii(full_segmentation, affine, result_nii_path)
    generate_segmentation_overlay(t1ce_path, result_nii_path, result_png_path)

    aligned_image_path = os.path.join(RESULT_FOLDER, 'aligned_segmentation.png')
    save_aligned_image(flair_img, full_segmentation, aligned_image_path)

    # Return URLs for the generated images
    return jsonify({
        'segmentation_image': f'/results/{os.path.basename(result_png_path)}',
        'aligned_image': f'/results/{os.path.basename(aligned_image_path)}'
    })

# Serve static files from the results folder
@app.route('/results/<filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
