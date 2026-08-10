"""
app_unet_landmarks.py
---------------------
Streamlit app for the from-scratch 9-landmark bee-wing U-Net.

Workflow:
1. User uploads image
2. User crops the wing region
3. User clicks "Save crop and start prediction"
4. Model predicts landmarks on the cropped image
5. Coordinates are mapped back to the original image
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from streamlit_cropper import st_cropper

    HAS_STREAMLIT_CROPPER = True
except Exception:
    HAS_STREAMLIT_CROPPER = False
    st_cropper = None

from inference_unet_landmarks import (
    create_landmark_overlay,
    detect_landmarks,
    load_landmark_model,
)


st.set_page_config(
    page_title="Bee Wing 9-Landmark U-Net",
    page_icon="🪽",
    layout="wide",
)


# -----------------------------
# Image helpers
# -----------------------------

def uploaded_file_to_bgr(uploaded_file) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.getvalue()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise ValueError(f"Could not read uploaded image: {uploaded_file.name}")

    return image_bgr


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def bgr_to_pil_rgb(image_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(bgr_to_rgb(image_bgr))


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_val = float(np.nanmin(image))
    max_val = float(np.nanmax(image))

    if max_val - min_val < 1e-8:
        return np.zeros_like(image, dtype=np.uint8)

    return ((image - min_val) / (max_val - min_val) * 255).clip(0, 255).astype(np.uint8)


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# -----------------------------
# Crop helpers
# -----------------------------

def sanitize_crop_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:

    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(x1 + 1, min(x2, image_width))
    y2 = max(y1 + 1, min(y2, image_height))

    return x1, y1, x2, y2


def crop_image_bgr(
    image_bgr: np.ndarray,
    crop_box: Tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = crop_box
    return image_bgr[y1:y2, x1:x2].copy()


def draw_crop_rectangle(
    image_bgr: np.ndarray,
    crop_box: Tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = crop_box

    vis = image_bgr.copy()
    cv2.rectangle(
        vis,
        (x1, y1),
        (x2 - 1, y2 - 1),
        color=(0, 255, 255),
        thickness=3,
    )

    return vis


def offset_points_to_original(
    points_xy: np.ndarray,
    crop_x1: int,
    crop_y1: int,
) -> np.ndarray:
    points = np.asarray(points_xy, dtype=np.float32).copy()

    valid = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])

    points[valid, 0] += crop_x1
    points[valid, 1] += crop_y1

    return points


def crop_selector_ui(
    image_bgr: np.ndarray,
    key_prefix: str,
) -> Tuple[int, int, int, int]:
    image_height, image_width = image_bgr.shape[:2]

    st.markdown("#### Step 1 — Crop the wing region")
    st.caption(
        "Adjust the crop box so only the clear wing area is selected. "
        "When finished, click **Save crop and start prediction**."
    )

    latest_crop_key = f"{key_prefix}_latest_crop_box"

    if HAS_STREAMLIT_CROPPER:
        pil_image = bgr_to_pil_rgb(image_bgr)

        crop_result = st_cropper(
            pil_image,
            realtime_update=True,   # IMPORTANT: keep latest user-selected crop
            box_color="#FFD400",
            aspect_ratio=None,
            return_type="box",
            key=f"{key_prefix}_cropper",
        )

        if crop_result is None:
            crop_box = st.session_state.get(
                latest_crop_key,
                (0, 0, image_width, image_height),
            )
        else:
            left = crop_result.get("left", 0)
            top = crop_result.get("top", 0)
            width = crop_result.get("width", image_width)
            height = crop_result.get("height", image_height)

            crop_box = sanitize_crop_box(
                x1=left,
                y1=top,
                x2=left + width,
                y2=top + height,
                image_width=image_width,
                image_height=image_height,
            )

            # Save the actual crop selected by the user
            st.session_state[latest_crop_key] = crop_box

    else:
        st.warning(
            "Optional package `streamlit-cropper` is not installed. "
            "Using slider-based cropping instead. "
            "Install it with: `pip install streamlit-cropper pillow`"
        )

        x_range = st.slider(
            "X crop range",
            min_value=0,
            max_value=image_width,
            value=(0, image_width),
            step=1,
            key=f"{key_prefix}_x_range",
        )

        y_range = st.slider(
            "Y crop range",
            min_value=0,
            max_value=image_height,
            value=(0, image_height),
            step=1,
            key=f"{key_prefix}_y_range",
        )

        crop_box = sanitize_crop_box(
            x1=x_range[0],
            y1=y_range[0],
            x2=x_range[1],
            y2=y_range[1],
            image_width=image_width,
            image_height=image_height,
        )

        st.session_state[latest_crop_key] = crop_box

    crop_bgr = crop_image_bgr(image_bgr, crop_box)
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    crop_h, crop_w = crop_bgr.shape[:2]

    st.caption(
        f"Current selected crop: x={crop_x1}:{crop_x2}, y={crop_y1}:{crop_y2} "
        f"| crop size = {crop_w} × {crop_h}px"
    )

    col1, col2 = st.columns(2)

    with col1:
        original_with_box = draw_crop_rectangle(image_bgr, crop_box)
        st.image(
            bgr_to_rgb(original_with_box),
            caption="Original image with user's selected crop",
            use_container_width=True,
        )

    with col2:
        st.image(
            bgr_to_rgb(crop_bgr),
            caption="Exact crop that will be passed to the model",
            use_container_width=True,
        )

    return crop_box

# -----------------------------
# Export helper
# -----------------------------

def create_zip_bytes(outputs: Dict[str, Dict]) -> bytes:
    buffer = io.BytesIO()
    combined_rows = []

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for output_key, item in outputs.items():
            stem = Path(output_key).stem

            df = item["df"]
            crop_bgr = item["crop_bgr"]
            overlay_crop_bgr = item["overlay_crop_bgr"]
            overlay_original_bgr = item["overlay_original_bgr"]

            zf.writestr(
                f"{stem}_unet_landmarks.csv",
                df.to_csv(index=False),
            )

            ok, encoded_crop = cv2.imencode(".png", crop_bgr)
            if ok:
                zf.writestr(
                    f"{stem}_crop_passed_to_model.png",
                    encoded_crop.tobytes(),
                )

            ok, encoded_crop_overlay = cv2.imencode(".png", overlay_crop_bgr)
            if ok:
                zf.writestr(
                    f"{stem}_prediction_on_crop.png",
                    encoded_crop_overlay.tobytes(),
                )

            ok, encoded_original_overlay = cv2.imencode(".png", overlay_original_bgr)
            if ok:
                zf.writestr(
                    f"{stem}_prediction_mapped_to_original.png",
                    encoded_original_overlay.tobytes(),
                )

            combined_rows.extend(df.to_dict("records"))

        combined_df = pd.DataFrame(combined_rows)
        zf.writestr(
            "combined_unet_landmarks.csv",
            combined_df.to_csv(index=False),
        )

    buffer.seek(0)
    return buffer.getvalue()


# -----------------------------
# Model loading
# -----------------------------

@st.cache_resource
def cached_load_model(weights_path_or_url: str):
    """
    Load the model from a local path or a URL.
    If it's a URL, download it first.
    """
    is_url = weights_path_or_url.startswith("http")

    if is_url:
        # For URLs, create a local path to save the downloaded file
        model_filename = Path(weights_path_or_url).name
        local_weights_path = Path(model_filename)
    else:
        # It's a local path already
        local_weights_path = Path(weights_path_or_url)

    if not local_weights_path.exists():
        if not is_url:
            raise FileNotFoundError(f"Local model weights not found: {local_weights_path}")

        try:
            import requests
            st.info(f"Downloading model from {weights_path_or_url}...")
            r = requests.get(weights_path_or_url, allow_redirects=True)
            r.raise_for_status()
            with open(local_weights_path, "wb") as f:
                f.write(r.content)
            st.success("Model downloaded successfully.")
        except Exception as e:
            st.error(f"Failed to download model from URL: {weights_path_or_url}")
            raise e

    return load_landmark_model(str(local_weights_path))



# -----------------------------
# Main Streamlit app
# -----------------------------

st.title("🪽 Bee Wing 9-Landmark U-Net")

st.caption(
    "Upload a wing image, crop the wing region, then click "
    "**Save crop and start prediction** to run the model."
)

with st.sidebar:
    st.header("Model")

    model_path = st.text_input(
        "Model weights path",
        value="https://huggingface.co/ayushajmera/bee-wing-unet-landmarks/resolve/main/best_unet_landmarks.pth",
        help=(
            "Local path or URL to the model weights (e.g., from Hugging Face Hub). "
            "Use the checkpoint from train_unet_landmarks.py."
        ),
    )

    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Keep 0.0 to always output all 9 landmarks.",
    )

    st.header("Display")

    show_heatmap = st.checkbox(
        "Show heatmap overlay",
        value=True,
    )

    show_labels = st.checkbox(
        "Show landmark numbers",
        value=True,
    )

    export_bottom_left_y = st.checkbox(
        "Also export y_bottom_left",
        value=True,
        help=(
            "CSV annotations used bottom-left origin. "
            "This adds y_bottom_left = image_height - y_image."
        ),
    )


uploaded_files = st.file_uploader(
    "Upload wing images",
    type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
    accept_multiple_files=True,
)


model = None
meta = None

try:
    model, meta = cached_load_model(model_path)

    st.success(
        f"Loaded model: {model_path} | "
        f"landmarks={meta['num_landmarks']} | "
        f"image_size={meta['image_size']} | "
        f"device={meta['device']}"
    )

    if meta.get("best_pck") is not None or meta.get("best_mean_error") is not None:
        st.caption(
            f"Checkpoint metrics: "
            f"best_pck={meta.get('best_pck')}, "
            f"best_mean_error={meta.get('best_mean_error')}"
        )

except Exception as e:
    st.error(f"Could not load model from `{model_path}`.\n\n{e}")


all_rows = []
outputs = {}


if not uploaded_files:
    st.info("Upload one or more wing images to begin.")

elif model is None:
    st.warning("Fix the model path/load error before processing images.")

else:
    for file_idx, uploaded_file in enumerate(uploaded_files):
        st.divider()
        st.subheader(uploaded_file.name)

        try:
            image_bgr = uploaded_file_to_bgr(uploaded_file)
            original_h, original_w = image_bgr.shape[:2]

            st.caption(f"Original image size: {original_w} × {original_h}px")

            key_prefix = f"{file_idx}_{Path(uploaded_file.name).stem}"

            saved_crop_key = f"{key_prefix}_saved_crop_box"
            prediction_ready_key = f"{key_prefix}_prediction_ready"

            current_crop_box = crop_selector_ui(
                image_bgr=image_bgr,
                key_prefix=key_prefix,
            )

            crop_x1, crop_y1, crop_x2, crop_y2 = current_crop_box
            current_crop_bgr = crop_image_bgr(image_bgr, current_crop_box)
            current_crop_h, current_crop_w = current_crop_bgr.shape[:2]

            if current_crop_w < 32 or current_crop_h < 32:
                st.warning("The selected crop is very small. Please select a larger wing region.")
                continue

            col_save, col_reset = st.columns([1, 1])

            with col_save:
                save_and_predict = st.button(
                    "✅ Save crop and start prediction",
                    key=f"{key_prefix}_save_predict_button",
                    type="primary",
                    use_container_width=True,
                )

            with col_reset:
                reset_prediction = st.button(
                    "🔄 Reset prediction",
                    key=f"{key_prefix}_reset_button",
                    use_container_width=True,
                )

            if reset_prediction:
                st.session_state.pop(saved_crop_key, None)
                st.session_state[prediction_ready_key] = False
                st.info("Crop and prediction were reset. Adjust the crop and click the prediction button again.")
                st.stop()

            if save_and_predict:
                latest_crop_key = f"{key_prefix}_latest_crop_box"

                user_selected_crop_box = st.session_state.get(
                    latest_crop_key,
                    current_crop_box,
                )

                st.session_state[saved_crop_key] = user_selected_crop_box
                st.session_state[prediction_ready_key] = True

                st.success("User-selected crop saved. Starting prediction...")
                st.session_state[saved_crop_key] = current_crop_box
                st.session_state[prediction_ready_key] = True
                st.success("Crop saved. Starting prediction...")

            if not st.session_state.get(prediction_ready_key, False):
                st.info("After cropping, click **Save crop and start prediction** to run the model.")
                continue

            saved_crop_box = st.session_state.get(saved_crop_key, current_crop_box)

            crop_x1, crop_y1, crop_x2, crop_y2 = saved_crop_box
            crop_bgr = crop_image_bgr(image_bgr, saved_crop_box)
            crop_h, crop_w = crop_bgr.shape[:2]

            st.markdown("#### Step 2 — Prediction started")

            st.caption(
                f"Using saved crop: x={crop_x1}:{crop_x2}, y={crop_y1}:{crop_y2} "
                f"| crop size = {crop_w} × {crop_h}px"
            )

            with st.spinner("Predicting landmarks on cropped image..."):
                points_crop, confs, heatmap = detect_landmarks(
                    image_bgr=crop_bgr,
                    model=model,
                    meta=meta,
                    conf_threshold=conf_threshold,
                )

            points_original = offset_points_to_original(
                points_xy=points_crop,
                crop_x1=crop_x1,
                crop_y1=crop_y1,
            )

            overlay_crop = create_landmark_overlay(
                image_bgr=crop_bgr,
                points_xy=points_crop,
                confs=confs,
                heatmap=heatmap if show_heatmap else None,
                show_labels=show_labels,
            )

            overlay_original = create_landmark_overlay(
                image_bgr=image_bgr,
                points_xy=points_original,
                confs=confs,
                heatmap=None,
                show_labels=show_labels,
            )

            overlay_original = draw_crop_rectangle(
                overlay_original,
                saved_crop_box,
            )

            st.success("Prediction complete.")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    bgr_to_rgb(overlay_crop),
                    caption="Prediction on saved crop passed to model",
                    use_container_width=True,
                )

            with col2:
                st.image(
                    bgr_to_rgb(overlay_original),
                    caption="Prediction mapped back to original image",
                    use_container_width=True,
                )

            if show_heatmap:
                st.image(
                    normalize_to_uint8(heatmap),
                    caption="Max predicted heatmap on cropped image",
                    use_container_width=True,
                )

            rows = []

            for idx, ((x_crop, y_crop), (x_orig, y_orig), conf) in enumerate(
                zip(points_crop, points_original, confs),
                start=1,
            ):
                row = {
                    "image": uploaded_file.name,
                    "landmark": idx,

                    "x": float(x_orig) if np.isfinite(x_orig) else np.nan,
                    "y_image_top_left": float(y_orig) if np.isfinite(y_orig) else np.nan,

                    "x_crop": float(x_crop) if np.isfinite(x_crop) else np.nan,
                    "y_crop_top_left": float(y_crop) if np.isfinite(y_crop) else np.nan,

                    "confidence": float(conf),

                    "crop_x1": crop_x1,
                    "crop_y1": crop_y1,
                    "crop_x2": crop_x2,
                    "crop_y2": crop_y2,
                    "crop_width": crop_w,
                    "crop_height": crop_h,

                    "original_width": original_w,
                    "original_height": original_h,
                }

                if export_bottom_left_y:
                    row["y_bottom_left"] = (
                        float(original_h - y_orig)
                        if np.isfinite(y_orig)
                        else np.nan
                    )

                    row["y_crop_bottom_left"] = (
                        float(crop_h - y_crop)
                        if np.isfinite(y_crop)
                        else np.nan
                    )

                rows.append(row)

            df = pd.DataFrame(rows)

            st.markdown("#### Landmark coordinates")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                label=f"Download CSV for {uploaded_file.name}",
                data=dataframe_to_csv_bytes(df),
                file_name=f"{Path(uploaded_file.name).stem}_unet_landmarks.csv",
                mime="text/csv",
            )

            all_rows.extend(rows)

            output_key = f"{file_idx + 1:03d}_{uploaded_file.name}"

            outputs[output_key] = {
                "df": df,
                "crop_bgr": crop_bgr,
                "overlay_crop_bgr": overlay_crop,
                "overlay_original_bgr": overlay_original,
            }

        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")


if all_rows:
    st.divider()
    st.header("Combined outputs")

    combined_df = pd.DataFrame(all_rows)

    st.dataframe(combined_df, use_container_width=True)

    st.download_button(
        label="Download combined CSV",
        data=dataframe_to_csv_bytes(combined_df),
        file_name="combined_unet_landmarks.csv",
        mime="text/csv",
    )

    st.download_button(
        label="Download ZIP of all outputs",
        data=create_zip_bytes(outputs),
        file_name="unet_landmark_outputs.zip",
        mime="application/zip",
    )