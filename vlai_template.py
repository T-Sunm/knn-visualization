import os, base64
import gradio as gr
from pathlib import Path

PROJECT_NAME = "KNN Demo"
AIO_YEAR = "2025"
AIO_MODULE = "03"
# END


def image_to_base64(image_path: str):
    # Construct the absolute path to the image
    full_image_path = Path(__file__).parent / image_path
    with open(full_image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def create_header():
    """Tạo header cho ứng dụng Gradio."""
    logo_base64_vku = image_to_base64("static/VKU.png")
    logo_base64_knn = image_to_base64("static/knn_logo.jpg")
    with gr.Row(elem_id="header", equal_height=False):
        with gr.Column(scale=1):
            gr.HTML(
                f"""
            <div class="header-logo-left">
                <a href="https" target="_blank">
                    <img src="data:image/png;base64,{logo_base64_vku}" alt="VKU Logo">
                </a>
            </div>
            """
            )
        with gr.Column(scale=2):
            gr.HTML(
                """
            <div class="header-title">
                <h1>KNN Algorithm Visualization</h1>
                <h2>Visualize how K-Nearest Neighbors works with your data</h2>
            </div>
            """
            )
        # with gr.Column(scale=1):
        #     gr.HTML(
        #         f"""
        #     <div class="header-logo-right">
        #          <a href="https" target="_blank">
        #             <img src="data:image/jpeg;base64,{logo_base64_knn}" alt="KNN Logo">
        #         </a>
        #     </div>
        #     """
        #     )


def create_footer():
    logo_base64_vlai = image_to_base64("static/vlai_logo.png")
    footer_html = (
        """
<style>
  .sticky-footer{position:fixed;bottom:0px;left:0;width:100%;background:#F4EBD3;
                 padding:10px;box-shadow:0 -2px 10px rgba(0,0,0,0.1);z-index:1000;}
  .content-wrap{padding-bottom:60px;}
</style>"""
        + f"""
<div class="sticky-footer">
  <div style="text-align:center;font-size:18px; color: #888">
    Created by 
    <a href="https://vlai.work" target="_blank" style="color:#465C88;text-decoration:none;font-weight:bold; display:inline-flex; align-items:center;"> VLAI 
    <img src="data:image/png;base64,{logo_base64_vlai}" alt="Logo" style="height:20px; width:auto;">
    </a> from <a href="https://aivietnam.edu.vn/" target="_blank" style="color:#355724;text-decoration:none;font-weight:bold">AI VIET NAM</a>
  </div>
</div>
"""
    )
    return gr.HTML(footer_html)


custom_css = """

.gradio-container {
    min-height: 100vh !important; 
    width: 100vw !important;
    margin: 0 !important;
    padding: 0px !important;
    background: linear-gradient(135deg, #F5EFE6 0%, #E8DFCA 50%, #AEBDCA 100%);
    background-size: 600% 600%;
    animation: gradientBG 7s ease infinite;
}

@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Minimize spacing and padding */
.content-wrap {
    padding: 2px !important;
    margin: 0 !important;
}

/* Reduce component spacing */
.gr-row {
    gap: 5px !important;
    margin: 2px 0 !important;
}

.gr-column {
    gap: 4px !important;
    padding: 4px !important;
}

/* Accordion optimization */
.gr-accordion {
    margin: 4px 0 !important;
}

.gr-accordion .gr-accordion-content {
    padding: 2px !important;
}

/* Form elements spacing */
.gr-form {
    gap: 2px !important;
}

/* Button styling */
.gr-button {
    margin: 2px 0 !important;
}

/* DataFrame optimization */
.gr-dataframe {
    margin: 4px 0 !important;
}

/* Remove horizontal scroll from data preview */
.gr-dataframe .wrap {
    overflow-x: auto !important;
    max-width: 100% !important;
}

/* Plot optimization */
.gr-plot {
    margin: 4px 0 !important;
}

/* Reduce markdown margins */
.gr-markdown {
    margin: 2px 0 !important;
}

/* Footer positioning */
.sticky-footer {
    position: fixed;
    bottom: 0px;
    left: 0;
    width: 100%;
    background: #F4EBD3;
    padding: 6px !important;
    box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
    z-index: 1000;
}
"""
