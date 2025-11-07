import gradio as gr
import pandas as pd
from src import knn_core
import vlai_template

# Global state
current_dataframe = None

# Dataset configurations
SAMPLE_DATA_CONFIG = {
    "Iris": {"target_column": "target", "problem_type": "classification"},
    "Wine": {"target_column": "target", "problem_type": "classification"},
    "Breast Cancer": {"target_column": "target", "problem_type": "classification"},
    "Diabetes": {"target_column": "target", "problem_type": "regression"},
}

force_light_theme_js = """
() => {
  const params = new URLSearchParams(window.location.search);
  if (!params.has('__theme')) {
    params.set('__theme', 'light');
    window.location.search = params.toString();
  }
}
"""


def validate_config(df, target_col, problem_type):
    """Validate target column and problem type compatibility"""
    if not target_col or target_col not in df.columns:
        return False, "❌ Please select a valid target column from the dropdown."

    if not problem_type:
        return (
            False,
            "❌ Please select either Classification or Regression as problem type.",
        )

    target_series = df[target_col]
    unique_vals = target_series.nunique()

    if problem_type == "classification":
        if unique_vals > 50:
            return (
                False,
                f"⚠️ Too many classes ({unique_vals}). Consider using Regression instead.",
            )
        if target_series.isnull().any():
            return (
                False,
                "⚠️ Target column contains missing values. Please clean your data.",
            )
    elif problem_type == "regression":
        if target_series.dtype == "object":
            return (
                False,
                "⚠️ Text values detected in target. Use Classification for categories.",
            )
        if unique_vals < 5:
            return (
                False,
                f"⚠️ Too few unique values ({unique_vals}). Consider using Classification.",
            )

    return (
        True,
        f"\n✅ Configuration is valid! Ready for {unique_vals} {'classes' if problem_type == 'classification' else 'values'}.",
    )


def get_status_message(
    is_sample, dataset_choice, target_col, problem_type, is_valid, validation_msg
):
    """Generate status message"""
    if is_sample:
        return f"✅ **Sample Dataset**: {dataset_choice} | **Target**: {target_col} | **Type**: {problem_type.title()}"
    elif target_col and problem_type:
        status_icon = "✅" if is_valid else "⚠️"
        return f"{status_icon} **Custom Data** | **Target**: {target_col} | **Type**: {problem_type.title()} | {validation_msg}"
    else:
        return "📁 **Custom data uploaded!** 👆 Please select target column and problem type above to continue."


def load_and_configure_data(file_obj=None, dataset_choice="Iris"):
    """Load data and configure target/problem type"""
    global current_dataframe

    try:
        df = knn_core.load_data(file_obj, dataset_choice)
        current_dataframe = df

        target_options = df.columns.tolist()
        is_sample = file_obj is None

        if is_sample:
            config = SAMPLE_DATA_CONFIG.get(dataset_choice, {})
            target_col = config.get("target_column")
            problem_type = config.get("problem_type")
        else:
            target_col = None
            problem_type = None

        # Validate and generate status
        if target_col and problem_type:
            is_valid, validation_msg = validate_config(df, target_col, problem_type)
            status_msg = get_status_message(
                is_sample,
                dataset_choice,
                target_col,
                problem_type,
                is_valid,
                validation_msg,
            )
        else:
            status_msg = get_status_message(
                is_sample, dataset_choice, target_col, problem_type, False, ""
            )

        # Generate input components
        input_updates = [gr.update(visible=False)] * 16
        inputs_visible = gr.update(visible=False)
        input_status = (
            "⚙️ Configure target and problem type above to enable feature inputs."
        )

        if target_col and problem_type and (not is_sample or is_valid):
            try:
                components_info = knn_core.create_input_components(df, target_col)
                for i in range(min(16, len(components_info))):
                    comp_info = components_info[i]
                    if comp_info["type"] == "number":
                        update_params = {
                            "visible": True,
                            "label": comp_info["name"],
                            "value": comp_info["value"],
                        }
                        if comp_info["minimum"] is not None:
                            update_params["minimum"] = comp_info["minimum"]
                        if comp_info["maximum"] is not None:
                            update_params["maximum"] = comp_info["maximum"]
                        input_updates[i] = gr.update(**update_params)
                    else:
                        input_updates[i] = gr.update(
                            visible=True,
                            label=comp_info["name"],
                            choices=comp_info["choices"],
                            value=comp_info["value"],
                        )
                inputs_visible = gr.update(visible=True)
                input_status = f"📝 **Ready!** Enter values for {len(components_info)} features below, then click Run Prediction! | {validation_msg}"
            except Exception as e:
                input_status = f"❌ Error generating inputs: {str(e)}"

        return (
            [
                df.head(5).round(2),
                gr.Dropdown(choices=target_options, value=target_col),
                gr.Dropdown(value=problem_type),
                status_msg,
            ]
            + input_updates
            + [inputs_visible, input_status]
        )

    except Exception as e:
        current_dataframe = None
        empty_updates = [
            pd.DataFrame(),
            gr.Dropdown(choices=[], value=None),
            gr.Dropdown(value=None),
            f"❌ **Error loading data**: {str(e)} | Please try a different file or dataset.",
        ]
        return (
            empty_updates
            + [gr.update(visible=False)] * 16
            + [gr.update(visible=False), "No data loaded."]
        )


def update_configuration(df_preview, target_col, problem_type):
    """Update configuration when target or problem type changes"""
    global current_dataframe
    df = current_dataframe

    if df is None or df.empty:
        return [gr.update(visible=False)] * 16 + [
            gr.update(visible=False),
            "No data available.",
        ]

    if not target_col or not problem_type:
        return [gr.update(visible=False)] * 16 + [
            gr.update(visible=False),
            "Select target column and problem type.",
        ]

    try:
        is_valid, validation_msg = validate_config(df, target_col, problem_type)

        if not is_valid:
            return [gr.update(visible=False)] * 16 + [
                gr.update(visible=False),
                f"⚠️ {validation_msg}",
            ]

        # Generate input components
        components_info = knn_core.create_input_components(df, target_col)
        input_updates = [gr.update(visible=False)] * 16

        for i in range(min(16, len(components_info))):
            comp_info = components_info[i]
            if comp_info["type"] == "number":
                # Không giới hạn min/max để cho phép user nhập giá trị ngoài phạm vi training data
                update_params = {
                    "visible": True,
                    "label": comp_info["name"],
                    "value": comp_info["value"],
                }
                if comp_info["minimum"] is not None:
                    update_params["minimum"] = comp_info["minimum"]
                if comp_info["maximum"] is not None:
                    update_params["maximum"] = comp_info["maximum"]
                input_updates[i] = gr.update(**update_params)
            else:
                input_updates[i] = gr.update(
                    visible=True,
                    label=comp_info["name"],
                    choices=comp_info["choices"],
                    value=comp_info["value"],
                )

        input_status = (
            f"📝 Enter values for {len(components_info)} features | {validation_msg}"
        )
        return input_updates + [gr.update(visible=True), input_status]

    except Exception as e:
        return [gr.update(visible=False)] * 16 + [
            gr.update(visible=False),
            f"❌ Error: {str(e)}",
        ]


def execute_prediction(
    df_preview,
    target_col,
    problem_type,
    k_value,
    distance_metric,
    weighting_method,
    *input_values,
):
    """Execute KNN prediction"""
    global current_dataframe
    df = current_dataframe

    # Validation checks
    if df is None or df.empty:
        return (
            None,
            "❌ **No data loaded!** 📊 Please select a sample dataset or upload a file first.",
            None,
            "Load data to get started.",
        )

    if not target_col or not problem_type:
        return (
            None,
            "❌ **Configuration incomplete!** 🎯 Please select target column and problem type above.",
            None,
            "Complete configuration to proceed.",
        )

    is_valid, validation_msg = validate_config(df, target_col, problem_type)
    if not is_valid:
        return (
            None,
            f"❌ **Configuration issue**: {validation_msg}",
            None,
            "Fix the configuration and try again.",
        )

    try:
        components_info = knn_core.create_input_components(df, target_col)
        new_point_dict = {}

        for i, comp_info in enumerate(components_info):
            if i < len(input_values) and input_values[i] is not None:
                new_point_dict[comp_info["name"]] = input_values[i]
            else:
                new_point_dict[comp_info["name"]] = comp_info["value"]

        fig, prediction, neighbor_df, summary, error = knn_core.run_knn_and_visualize(
            df,
            target_col,
            new_point_dict,
            int(k_value),
            distance_metric,
            weighting_method,
            problem_type,
        )

        if error:
            return (
                None,
                f"❌ **Prediction failed**: {error} | Please check your input values and try again.",
                None,
                "Adjust inputs and retry.",
            )

        if problem_type == "classification":
            result_header = f"## 🎯 **Classification Result**: {prediction}\n*Based on {int(k_value)} nearest neighbors using {distance_metric} distance*"
        else:
            result_header = f"## 🎯 **Regression Result**: {prediction:.3f}\n*Based on {int(k_value)} nearest neighbors using {distance_metric} distance*"

        return fig, result_header, neighbor_df, summary

    except Exception as e:
        return (
            None,
            f"❌ **Execution error**: {str(e)} | Please verify your input values are correct.",
            None,
            "Check inputs and try again.",
        )


# Main Application
with gr.Blocks(
    theme="gstaff/sketch",
    css=vlai_template.custom_css,
    fill_width=True,
    js=force_light_theme_js,
) as demo:
    vlai_template.create_header()

    # Main guidance text
    gr.Markdown(
        "### 🎯 **How to Use**: Select data → Configure target → Set parameters → Enter new point → Run prediction!"
    )

    with gr.Row(equal_height=False, variant="panel"):
        with gr.Column(scale=45):
            with gr.Accordion("📊 Data & Configuration", open=True):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "Start with sample datasets or upload your own CSV/Excel files."
                        )
                        file_upload = gr.File(
                            label="📁 Upload Your Data",
                            file_types=[".csv", ".xlsx", ".xls"],
                        )
                    with gr.Column(scale=3):
                        sample_dataset = gr.Dropdown(
                            choices=list(SAMPLE_DATA_CONFIG.keys()),
                            value="Iris",
                            label="🗂️ Sample Datasets",
                        )
                        problem_type_selector = gr.Dropdown(
                            choices=["classification", "regression"],
                            label="🎲 Problem Type",
                            interactive=True,
                        )
                        target_column = gr.Dropdown(
                            choices=[],
                            label="🎯 Target Column",
                            interactive=True,
                        )

                status_message = gr.Markdown("🔄 Loading sample data...")
                data_preview = gr.DataFrame(
                    label="📋 Data Preview (First 5 Rows)",
                    row_count=5,
                    interactive=False,
                    max_height=250,
                )

            with gr.Accordion("⚙️ Parameters & Input", open=True):
                gr.Markdown("**📐 Algorithm Parameters**")
                with gr.Row():
                    k_value = gr.Number(
                        label="K Value",
                        value=3,
                        minimum=1,
                        maximum=50,
                        precision=0,
                    )
                    distance_metric = gr.Dropdown(
                        choices=["euclidean", "manhattan", "cosine", "minkowski"],
                        value="euclidean",
                        label="📏 Distance Metric",
                    )
                    weighting_method = gr.Dropdown(
                        choices=["uniform", "distance"],
                        value="uniform",
                        label="⚖️ Weighting Method",
                    )

                inputs_group = gr.Group(visible=False)
                with inputs_group:
                    input_status = gr.Markdown("Configure inputs above.")
                    gr.Markdown(
                        "**📝 New Data Point** - Enter feature values for prediction:"
                    )

                    input_components = []
                    for row in range(4):
                        with gr.Row():
                            for col in range(4):
                                idx = row * 4 + col
                                if idx < 16:
                                    input_components.append(
                                        gr.Number(
                                            label=f"Feature {idx+1}", visible=False
                                        )
                                    )

                run_prediction_btn = gr.Button(
                    "🚀 Run Prediction",
                    variant="primary",
                    size="lg",
                )

        with gr.Column(scale=55):
            gr.Markdown("### 📊 **Results & Visualization**")

            visualization_plot = gr.Plot(
                label="Interactive KNN Visualization",
                visible=True,
            )

            prediction_result = gr.Markdown(
                "## 🎯 Prediction Result\n**Run prediction to see the result.**",
                label="📈 Final Prediction",
            )

            neighbor_details = gr.DataFrame(
                label="👥 Nearest Neighbors Details",
                row_count=5,
                wrap=True,
                max_height=250,
            )

            algorithm_summary = gr.Markdown(
                "**📋 Algorithm Summary**\n\nAlgorithm details will appear here after prediction.",
                label="🔍 Technical Details",
            )

    # Bottom guidance
    gr.Markdown(
        """💡 **Tips**: 
    - **High-dimensional data** automatically uses t-SNE for 2D visualization.    
    - **Orange circles** show visually closest points, **black diamonds** show algorithm-accurate neighbors.    
    - Try different **K values** and **distance metrics** to see how results change!
    """
    )

    # vlai_template.create_footer()

    # Event Bindings
    demo.load(
        fn=lambda: load_and_configure_data(None, "Iris"),
        outputs=[data_preview, target_column, problem_type_selector, status_message]
        + input_components
        + [inputs_group, input_status],
    )

    file_upload.upload(
        fn=lambda file: load_and_configure_data(file, "Iris"),
        inputs=[file_upload],
        outputs=[data_preview, target_column, problem_type_selector, status_message]
        + input_components
        + [inputs_group, input_status],
    )

    sample_dataset.change(
        fn=lambda choice: load_and_configure_data(None, choice),
        inputs=[sample_dataset],
        outputs=[data_preview, target_column, problem_type_selector, status_message]
        + input_components
        + [inputs_group, input_status],
    )

    target_column.change(
        fn=update_configuration,
        inputs=[data_preview, target_column, problem_type_selector],
        outputs=input_components + [inputs_group, input_status],
    )

    problem_type_selector.change(
        fn=update_configuration,
        inputs=[data_preview, target_column, problem_type_selector],
        outputs=input_components + [inputs_group, input_status],
    )

    run_prediction_btn.click(
        fn=execute_prediction,
        inputs=[
            data_preview,
            target_column,
            problem_type_selector,
            k_value,
            distance_metric,
            weighting_method,
        ]
        + input_components,
        outputs=[
            visualization_plot,
            prediction_result,
            neighbor_details,
            algorithm_summary,
        ],
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=["static"])
