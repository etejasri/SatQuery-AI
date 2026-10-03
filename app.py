import os
import base64
import io
import numpy as np

import gradio as gr
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from ultralytics import YOLO
from faster_whisper import WhisperModel


# ============================================================
# SATQUERY AI
# ============================================================
# Interactive Vision-Language Assistant
# for Remote Sensing Image Analysis
# ============================================================


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN
)


# ============================================================
# VOICE MODEL
# ============================================================

speech_model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)


# ============================================================
# BUILDING DETECTION MODEL
# ============================================================

building_model = YOLO(
    r"C:\Users\Teja\.cache\huggingface\hub\models--keremberke--yolov8n-building-segmentation\snapshots\eb61328c39fbc531470acba556858ec4ce71e6bf\best.pt"
)


# ============================================================
# IMAGE TO DATA URL
# ============================================================

def image_to_data_url(image_path):

    img = Image.open(image_path).convert("RGB")

    buffer = io.BytesIO()

    img.save(
        buffer,
        format="JPEG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/jpeg;base64,"
        + encoded
    )


# ============================================================
# LAND USE ESTIMATION
# ============================================================

def estimate_land_use(image_path):

    img = Image.open(
        image_path
    ).convert("RGB")

    img = img.resize(
        (300, 300)
    )

    pixels = np.array(
        img
    ).reshape(-1, 3)

    r = pixels[:, 0].astype(float)
    g = pixels[:, 1].astype(float)
    b = pixels[:, 2].astype(float)

    vegetation = (
        (g > r * 1.15)
        &
        (g > b * 1.10)
    )

    water = (
        (b > r * 1.15)
        &
        (b > g * 1.02)
    )

    built_up = (
        (abs(r - g) < 35)
        &
        (abs(g - b) < 35)
        &
        (abs(r - b) < 35)
    )

    total = len(pixels)

    vegetation_pct = (
        vegetation.sum()
        /
        total
        *
        100
    )

    water_pct = (
        water.sum()
        /
        total
        *
        100
    )

    built_up_pct = (
        built_up.sum()
        /
        total
        *
        100
    )

    other_pct = max(
        0,
        100
        -
        vegetation_pct
        -
        water_pct
        -
        built_up_pct
    )

    return [
        round(vegetation_pct, 1),
        round(water_pct, 1),
        round(built_up_pct, 1),
        round(other_pct, 1)
    ]


# ============================================================
# SINGLE IMAGE EVIDENCE GRAPH
# ============================================================

def create_single_evidence(image_path):

    values = estimate_land_use(
        image_path
    )

    labels = [
        "Vegetation",
        "Water",
        "Built-up",
        "Other"
    ]

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    bars = ax.bar(
        labels,
        values
    )

    ax.set_title(
        "Land-use Evidence"
    )

    ax.set_ylabel(
        "Estimated Percentage (%)"
    )

    ax.set_ylim(
        0,
        100
    )

    for bar, value in zip(
        bars,
        values
    ):
        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center"
        )

    plt.tight_layout()

    return fig


# ============================================================
# BEFORE VS AFTER EVIDENCE
# ============================================================

def create_comparison_evidence(
    before_image,
    after_image
):

    before_values = estimate_land_use(
        before_image
    )

    after_values = estimate_land_use(
        after_image
    )

    labels = [
        "Vegetation",
        "Water",
        "Built-up",
        "Other"
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    before_bars = ax.bar(
        x - width / 2,
        before_values,
        width,
        label="Before"
    )

    after_bars = ax.bar(
        x + width / 2,
        after_values,
        width,
        label="After"
    )

    ax.set_title(
        "Before vs After Evidence"
    )

    ax.set_ylabel(
        "Estimated Percentage (%)"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels
    )

    ax.set_ylim(
        0,
        100
    )

    ax.legend()

    for bar, value in zip(
        before_bars,
        before_values
    ):
        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center"
        )

    for bar, value in zip(
        after_bars,
        after_values
    ):
        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center"
        )

    plt.tight_layout()

    return fig


# ============================================================
# BUILDING DETECTION
# ============================================================

def count_buildings(
    image_path
):

    img = Image.open(
        image_path
    ).convert("RGB")

    width, height = img.size

    tiles = [
        (
            0,
            0,
            width // 2 + 100,
            height // 2 + 100
        ),
        (
            width // 2 - 100,
            0,
            width,
            height // 2 + 100
        ),
        (
            0,
            height // 2 - 100,
            width // 2 + 100,
            height
        ),
        (
            width // 2 - 100,
            height // 2 - 100,
            width,
            height
        )
    ]

    all_boxes = []
    all_confidences = []

    for tile in tiles:

        crop = img.crop(
            tile
        )

        results = building_model(
            crop,
            conf=0.20,
            imgsz=1280,
            verbose=False
        )

        result = results[0]

        if result.boxes is None:
            continue

        for box, confidence in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy()
        ):

            x1, y1, x2, y2 = box

            all_confidences.append(
                float(confidence)
            )

            x1 += tile[0]
            x2 += tile[0]

            y1 += tile[1]
            y2 += tile[1]

            box_width = x2 - x1
            box_height = y2 - y1

            if (
                box_width < width * 0.15
                and
                box_height < height * 0.15
            ):
                all_boxes.append(
                    [
                        x1,
                        y1,
                        x2,
                        y2
                    ]
                )

    # --------------------------------------------------------
    # Remove duplicate detections
    # --------------------------------------------------------

    kept_boxes = []

    for box in all_boxes:

        x1, y1, x2, y2 = box

        duplicate = False

        for old_box in kept_boxes:

            ox1, oy1, ox2, oy2 = old_box

            ix1 = max(
                x1,
                ox1
            )

            iy1 = max(
                y1,
                oy1
            )

            ix2 = min(
                x2,
                ox2
            )

            iy2 = min(
                y2,
                oy2
            )

            if (
                ix2 > ix1
                and
                iy2 > iy1
            ):

                intersection = (
                    (ix2 - ix1)
                    *
                    (iy2 - iy1)
                )

                area1 = (
                    (x2 - x1)
                    *
                    (y2 - y1)
                )

                area2 = (
                    (ox2 - ox1)
                    *
                    (oy2 - oy1)
                )

                union = (
                    area1
                    +
                    area2
                    -
                    intersection
                )

                if union > 0:

                    iou = (
                        intersection
                        /
                        union
                    )

                    if iou > 0.5:

                        duplicate = True

                        break

        if not duplicate:

            kept_boxes.append(
                box
            )

    # --------------------------------------------------------
    # Average confidence
    # --------------------------------------------------------

    if all_confidences:

        average_confidence = (
            sum(all_confidences)
            /
            len(all_confidences)
        )

    else:

        average_confidence = 0.0

    # --------------------------------------------------------
    # Draw boxes
    # --------------------------------------------------------

    output = np.array(
        img
    ).copy()

    for box in kept_boxes:

        x1, y1, x2, y2 = [
            int(v)
            for v in box
        ]

        output[
            max(0, y1):
            min(height, y1 + 3),
            max(0, x1):
            min(width, x2)
        ] = [
            255,
            0,
            0
        ]

        output[
            max(0, y2 - 3):
            min(height, y2),
            max(0, x1):
            min(width, x2)
        ] = [
            255,
            0,
            0
        ]

        output[
            max(0, y1):
            min(height, y2),
            max(0, x1):
            min(width, x1 + 3)
        ] = [
            255,
            0,
            0
        ]

        output[
            max(0, y1):
            min(height, y2),
            max(0, x2 - 3):
            min(width, x2)
        ] = [
            255,
            0,
            0
        ]

    output_path = (
        "building_detection.jpg"
    )

    Image.fromarray(
        output
    ).save(
        output_path
    )

    return (
        len(kept_boxes),
        output_path,
        average_confidence
    )


# ============================================================
# VOICE TO TEXT
# ============================================================

def voice_to_text(
    audio_path
):

    if audio_path is None:

        return ""

    segments, info = (
        speech_model.transcribe(
            audio_path
        )
    )

    text = ""

    for segment in segments:

        text += segment.text

    return text.strip()


# ============================================================
# AI SINGLE IMAGE ANALYSIS
# ============================================================

def analyze_single_image(
    image,
    question
):

    if image is None:

        return (
            "Please upload a satellite image.",
            None,
            None
        )

    if not question:

        question = (
            "Describe the important features "
            "visible in this satellite image."
        )

    try:

        # ----------------------------------------------------
        # Building question
        # ----------------------------------------------------

        if "building" in question.lower():

            count, detected_path, confidence = (
                count_buildings(
                    image
                )
            )

            fig, ax = plt.subplots(
                figsize=(7, 5)
            )

            bars = ax.bar(
                ["Buildings"],
                [count]
            )

            ax.set_title(
                "Building Detection Evidence"
            )

            ax.set_ylabel(
                "Detected Count"
            )

            for bar in bars:

                ax.text(
                    bar.get_x()
                    +
                    bar.get_width() / 2,
                    bar.get_height() + 1,
                    str(count),
                    ha="center"
                )

            plt.tight_layout()

            answer = (
                f"Detected building regions: "
                f"{count}\n\n"
                f"Average detection confidence: "
                f"{confidence:.0%}"
            )

            return (
                answer,
                detected_path,
                fig
            )

        # ----------------------------------------------------
        # General AI question
        # ----------------------------------------------------

        image_data = (
            image_to_data_url(
                image
            )
        )

        result = client.chat.completions.create(

            model="Qwen/Qwen3.8-27B",

            messages=[
                {
                    "role": "user",

                    "content": [

                        {
                            "type": "text",

                            "text": (
                                f"{question}\n\n"
                                "Analyze the satellite image "
                                "and answer the question "
                                "directly using simple sentences. "
                                "Describe visible evidence only."
                            )
                        },

                        {
                            "type": "image_url",

                            "image_url": {
                                "url": image_data
                            }
                        }
                    ]
                }
            ],

            max_tokens=500
        )

        answer = (
            result
            .choices[0]
            .message
            .content
        )

        if not answer:

            answer = (
                "The image was analyzed, "
                "but no final answer was returned."
            )

        evidence_graph = (
            create_single_evidence(
                image
            )
        )

        return (
            answer,
            None,
            evidence_graph
        )

    except Exception as e:

        print(
            "SINGLE IMAGE ERROR:",
            repr(e)
        )

        return (
            f"Error: {str(e)}",
            None,
            None
        )


# ============================================================
# TWO IMAGE COMPARISON
# ============================================================

def compare_two_images(
    image1,
    image2,
    question
):

    # --------------------------------------------------------
    # No images
    # --------------------------------------------------------

    if (
        image1 is None
        and
        image2 is None
    ):

        return (
            "Please upload at least one satellite image.",
            None,
            None
        )

    # --------------------------------------------------------
    # Only first image
    # --------------------------------------------------------

    if (
        image1 is not None
        and
        image2 is None
    ):

        return analyze_single_image(
            image1,
            question
        )

    # --------------------------------------------------------
    # Only second image
    # --------------------------------------------------------

    if (
        image1 is None
        and
        image2 is not None
    ):

        return analyze_single_image(
            image2,
            question
        )

    # --------------------------------------------------------
    # Two images
    # --------------------------------------------------------

    if not question:

        question = (
            "What has changed between "
            "these two satellite images?"
        )

    try:

        before_data = (
            image_to_data_url(
                image1
            )
        )

        after_data = (
            image_to_data_url(
                image2
            )
        )

        result = client.chat.completions.create(

            model="Qwen/Qwen3.8-27B",

            messages=[
                {
                    "role": "user",

                    "content": [

                        {
                            "type": "text",

                            "text": (
                                "Compare these two satellite images.\n\n"
                                "IMAGE 1 = BEFORE\n"
                                "IMAGE 2 = AFTER\n\n"
                                f"User question: {question}\n\n"
                                "Give the answer using these sections:\n\n"
                                "BEFORE:\n"
                                "Describe visible features in Image 1.\n\n"
                                "AFTER:\n"
                                "Describe visible features in Image 2.\n\n"
                                "CHANGES:\n"
                                "Describe the visible changes.\n\n"
                                "NEW OR MISSING:\n"
                                "Mention features that appear or disappear.\n\n"
                                "Use simple sentences and visible evidence only."
                            )
                        },

                        {
                            "type": "image_url",

                            "image_url": {
                                "url": before_data
                            }
                        },

                        {
                            "type": "image_url",

                            "image_url": {
                                "url": after_data
                            }
                        }
                    ]
                }
            ],

            max_tokens=700
        )

        answer = (
            result
            .choices[0]
            .message
            .content
        )

        if not answer:

            answer = (
                "The images were received, "
                "but no comparison answer was returned."
            )

        graph = (
            create_comparison_evidence(
                image1,
                image2
            )
        )

        return (
            answer,
            None,
            graph
        )

    except Exception as e:

        print(
            "COMPARISON ERROR:",
            repr(e)
        )

        return (
            f"Comparison error: {str(e)}",
            None,
            None
        )


# ============================================================
# UPLOAD PAGE
# ============================================================

def upload_page():

    return gr.update(
        visible=True
    )


# ============================================================
# MAIN CSS
# ============================================================

CUSTOM_CSS = """

html, body {
    margin: 0;
    padding: 0;
    width: 100%;
    height: 100%;
}

body {
    background: #f5f7fb !important;
}

.gradio-container {
    max-width: 100% !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}


/* Main application */

#app {
    min-height: 100vh;
    display: flex;
    flex-direction: row;
    background: #f5f7fb;
}


/* Fixed left sidebar */

#sidebar {
    width: 270px !important;
    min-width: 270px !important;
    max-width: 270px !important;

    height: 100vh !important;

    position: fixed !important;
    left: 0;
    top: 0;

    background: #111827 !important;

    padding: 25px 18px !important;

    box-sizing: border-box;

    overflow-y: auto;

    z-index: 1000;
}


/* Right content */

#content {
    margin-left: 270px !important;

    width: calc(100% - 270px) !important;

    min-height: 100vh;

    padding: 35px 45px !important;

    box-sizing: border-box;

    background: #f5f7fb !important;
}


/* Logo */

.logo {
    color: white;
    font-size: 25px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 8px;
}

.logo-subtitle {
    color: #9ca3af;
    font-size: 13px;
    text-align: center;
    margin-bottom: 35px;
}


/* Sidebar buttons */

.sidebar-btn {
    width: 100% !important;

    text-align: left !important;

    background: transparent !important;

    color: #d1d5db !important;

    border: none !important;

    margin-bottom: 8px !important;

    border-radius: 10px !important;

    font-size: 15px !important;
}

.sidebar-btn:hover {
    background: #1f2937 !important;
    color: white !important;
}


/* Page title */

.page-title {
    font-size: 32px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 8px;
}

.page-description {
    color: #6b7280;
    font-size: 16px;
    margin-bottom: 25px;
}


/* Cards */

.card {
    background: white;

    border-radius: 16px;

    padding: 25px;

    border: 1px solid #e5e7eb;

    box-shadow: 0 5px 20px rgba(0,0,0,0.05);

    margin-bottom: 20px;
}


/* Upload box */

.upload-box {
    background: white;

    border: 2px dashed #cbd5e1;

    border-radius: 16px;

    padding: 20px;

    margin-bottom: 20px;
}


/* Headings */

.section-heading {
    font-size: 20px;
    font-weight: 600;
    color: #111827;
    margin-bottom: 15px;
}


/* Responsive */

@media screen and (max-width: 900px) {

    #sidebar {
        width: 220px !important;
        min-width: 220px !important;
        max-width: 220px !important;
    }

    #content {
        margin-left: 220px !important;
        width: calc(100% - 220px) !important;
        padding: 25px !important;
    }
}

"""


# ============================================================
# USER INTERFACE
# ============================================================

with gr.Blocks(
    title="SatQuery AI",
    css=CUSTOM_CSS
) as demo:

    with gr.Row(
        elem_id="app"
    ):

        # ====================================================
        # LEFT SIDEBAR
        # ====================================================

        with gr.Column(
            elem_id="sidebar"
        ):

            gr.HTML(
                """
                <div class="logo">
                    🛰️ SatQuery AI
                </div>

                <div class="logo-subtitle">
                    Remote Sensing Analysis
                </div>
                """
            )

            home_btn = gr.Button(
                "🏠  Home",
                elem_classes="sidebar-btn"
            )

            upload_btn = gr.Button(
                "📤  Upload Image",
                elem_classes="sidebar-btn"
            )

            answer_btn = gr.Button(
                "🤖  AI Answer",
                elem_classes="sidebar-btn"
            )

            evidence_btn = gr.Button(
                "📊  Evidence",
                elem_classes="sidebar-btn"
            )

            building_btn = gr.Button(
                "🏢  Building Detection",
                elem_classes="sidebar-btn"
            )

            tools_btn = gr.Button(
                "🛠️  Tools Used",
                elem_classes="sidebar-btn"
            )


        # ====================================================
        # RIGHT CONTENT
        # ====================================================

        with gr.Column(
            elem_id="content"
        ):

            # =================================================
            # HOME
            # =================================================

            with gr.Column(
                visible=True
            ) as home_page:

                gr.HTML(
                    """
                    <div class="page-title">
                        🛰️ SatQuery AI
                    </div>

                    <div class="page-description">
                        Interactive Vision-Language Assistant
                        for Remote Sensing Image Analysis
                    </div>

                    <div class="card">

                        <div class="section-heading">
                            Welcome to SatQuery AI
                        </div>

                        <p>
                            Upload a satellite image and ask
                            questions using text or voice.
                        </p>

                        <p>
                            SatQuery AI analyzes the image and
                            provides an answer with visual evidence.
                        </p>

                        <p>
                            You can also compare a BEFORE image
                            with an AFTER image.
                        </p>

                    </div>
                    """
                )


            # =================================================
            # UPLOAD PAGE
            # =================================================

            with gr.Column(
                visible=False
            ) as upload_page_group:

                gr.HTML(
                    """
                    <div class="page-title">
                        📤 Upload Image
                    </div>

                    <div class="page-description">
                        Upload one image for analysis or two
                        images for BEFORE vs AFTER comparison.
                    </div>
                    """
                )

                with gr.Row():

                    with gr.Column(
                        elem_classes="card"
                    ):

                        image1 = gr.File(
                            file_types=["image"],
                            type="filepath",
                            label="🛰️ Image 1 / Before"
                        )

                    with gr.Column(
                        elem_classes="card"
                    ):

                        image2 = gr.File(
                            file_types=["image"],
                            type="filepath",
                            label="🛰️ Image 2 / After"
                        )

                with gr.Column(
                    elem_classes="card"
                ):

                    question = gr.Textbox(
                        label="⌨️ Ask a Question",
                        placeholder=(
                            "Example: How many buildings "
                            "are visible in this image?"
                        ),
                        lines=3
                    )

                    voice = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="🎤 Ask by Voice"
                    )

                    voice.change(
                        fn=voice_to_text,
                        inputs=voice,
                        outputs=question
                    )

                    analyze_button = gr.Button(
                        "🔍 Analyze",
                        variant="primary"
                    )


            # =================================================
            # AI ANSWER PAGE
            # =================================================

            with gr.Column(
                visible=False
            ) as answer_page:

                gr.HTML(
                    """
                    <div class="page-title">
                        🤖 AI Answer
                    </div>

                    <div class="page-description">
                        Answer generated from the uploaded
                        satellite image.
                    </div>
                    """
                )

                answer_output = gr.Textbox(
                    label="AI Answer",
                    lines=12,
                    interactive=False
                )


            # =================================================
            # EVIDENCE PAGE
            # =================================================

            with gr.Column(
                visible=False
            ) as evidence_page:

                gr.HTML(
                    """
                    <div class="page-title">
                        📊 Evidence
                    </div>

                    <div class="page-description">
                        Visual evidence generated from the
                        satellite image analysis.
                    </div>
                    """
                )

                evidence_output = gr.Plot(
                    label="Evidence Graph"
                )


            # =================================================
            # BUILDING PAGE
            # =================================================

            with gr.Column(
                visible=False
            ) as building_page:

                gr.HTML(
                    """
                    <div class="page-title">
                        🏢 Building Detection
                    </div>

                    <div class="page-description">
                        Detected building regions are shown
                        with bounding boxes.
                    </div>
                    """
                )

                building_output = gr.Image(
                    label="Detected Buildings"
                )

                building_info = gr.Textbox(
                    label="Detection Information",
                    lines=5,
                    interactive=False
                )


            # =================================================
            # TOOLS PAGE
            # =================================================

            with gr.Column(
                visible=False
            ) as tools_page:

                gr.HTML(
                    """
                    <div class="page-title">
                        🛠️ Tools Used
                    </div>

                    <div class="card">

                        <div class="section-heading">
                            Technologies used in SatQuery AI
                        </div>

                        <p>
                            <b>Gradio</b><br>
                            Used to create the web interface.
                        </p>

                        <p>
                            <b>Hugging Face</b><br>
                            Used for AI model access.
                        </p>

                        <p>
                            <b>Qwen Vision-Language Model</b><br>
                            Used for satellite image question answering
                            and image comparison.
                        </p>

                        <p>
                            <b>YOLO</b><br>
                            Used for building detection.
                        </p>

                        <p>
                            <b>Faster-Whisper</b><br>
                            Used for converting voice questions
                            into text.
                        </p>

                        <p>
                            <b>Matplotlib</b><br>
                            Used to create evidence graphs.
                        </p>

                        <p>
                            <b>Python</b><br>
                            Used for the complete application logic.
                        </p>

                    </div>
                    """
                )


    # ========================================================
    # ANALYSIS OUTPUTS
    # ========================================================

    def run_analysis(
        image1_path,
        image2_path,
        question_text
    ):

        answer_text, detected_path, graph = (
            compare_two_images(
                image1_path,
                image2_path,
                question_text
            )
        )

        building_info_text = ""

        if (
            image1_path
            and
            not image2_path
            and
            question_text
            and
            "building"
            in question_text.lower()
        ):

            try:

                count, path, confidence = (
                    count_buildings(
                        image1_path
                    )
                )

                building_info_text = (
                    f"Detected building regions: "
                    f"{count}\n\n"
                    f"Average detection confidence: "
                    f"{confidence:.0%}"
                )

            except Exception as e:

                building_info_text = (
                    f"Building detection error: {e}"
                )

        return (
            answer_text,
            detected_path,
            graph,
            building_info_text
        )


    analyze_button.click(
        fn=run_analysis,
        inputs=[
            image1,
            image2,
            question
        ],
        outputs=[
            answer_output,
            building_output,
            evidence_output,
            building_info
        ]
    )


    # ========================================================
    # SIDEBAR NAVIGATION
    # ========================================================

    def show_home():

        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )


    def show_upload():

        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )


    def show_answer():

        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )


    def show_evidence():

        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False)
        )


    def show_building():

        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False)
        )


    def show_tools():

        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True)
        )


    navigation_outputs = [
        home_page,
        upload_page_group,
        answer_page,
        evidence_page,
        building_page,
        tools_page
    ]


    home_btn.click(
        fn=show_home,
        outputs=navigation_outputs
    )

    upload_btn.click(
        fn=show_upload,
        outputs=navigation_outputs
    )

    answer_btn.click(
        fn=show_answer,
        outputs=navigation_outputs
    )

    evidence_btn.click(
        fn=show_evidence,
        outputs=navigation_outputs
    )

    building_btn.click(
        fn=show_building,
        outputs=navigation_outputs
    )

    tools_btn.click(
        fn=show_tools,
        outputs=navigation_outputs
    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch(
        inbrowser=True,
        share=False
    )