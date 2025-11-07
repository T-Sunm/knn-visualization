import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.datasets import (
    load_iris, load_wine, load_diabetes, load_breast_cancer
)
import plotly.express as px
import plotly.graph_objects as go

def load_data(file_obj=None, dataset_choice="Iris"):
    """Load data from file or sample dataset"""
    if file_obj is not None:
        if file_obj.name.endswith('.csv'):
            return pd.read_csv(file_obj.name)
        elif file_obj.name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_obj.name)
        else:
            raise ValueError("Unsupported format. Upload CSV or Excel files.")
    
    # Sample datasets
    datasets = {
        "Iris": lambda: _sklearn_to_df(load_iris()),
        "Wine": lambda: _sklearn_to_df(load_wine()),
        "Breast Cancer": lambda: _sklearn_to_df(load_breast_cancer()),
        "Diabetes": lambda: _sklearn_to_df(load_diabetes()),
    }
    
    if dataset_choice not in datasets:
        raise ValueError(f"Unknown dataset: {dataset_choice}")
    
    return datasets[dataset_choice]()

def _sklearn_to_df(data):
    """Convert sklearn dataset to DataFrame"""
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

def analyze_dataframe(df):
    """Analyze DataFrame and return target options"""
    return df.columns.tolist(), df.columns[-1]

def determine_problem_type(df, target_col):
    """Auto-detect classification or regression"""
    if target_col not in df.columns:
        return "classification"
    
    target = df[target_col]
    unique_vals = target.nunique()
    
    if target.dtype == 'object' or unique_vals <= min(20, len(target) * 0.1):
        return "classification"
    return "regression"

def create_input_components(df, target_col):
    """Generate UI component specifications for features"""
    feature_cols = [col for col in df.columns if col != target_col]
    components = []
    
    for col in feature_cols:
        data = df[col]
        if data.dtype == 'object':
            unique_vals = sorted(data.unique())
            components.append({
                'name': col, 'type': 'dropdown',
                'choices': unique_vals, 'value': unique_vals[0]
            })
        else:
            components.append({
                'name': col, 'type': 'number', 
                'value': round(float(data.mean()), 2),
                'minimum': None, 
                'maximum': None
            })
    
    return components

def preprocess_and_reduce(df, target_col, new_point_dict):
    """Preprocess data and apply dimensionality reduction"""
    feature_cols = [col for col in df.columns if col != target_col]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Encode categorical variables
    encoders = {}
    for col in feature_cols:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le
    
    # Process new point
    new_point = []
    for col in feature_cols:
        if col in encoders:
            try:
                val = encoders[col].transform([str(new_point_dict[col])])[0]
            except ValueError:
                # Đưa ra error rõ ràng thay vì fallback không chính xác
                available_categories = list(encoders[col].classes_)
                raise ValueError(f"Unknown category '{new_point_dict[col]}' for column '{col}'. Available options: {available_categories}")
            new_point.append(val)
        else:
            new_point.append(float(new_point_dict[col]))
    
    new_point = np.array(new_point).reshape(1, -1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    new_point_scaled = scaler.transform(new_point)
    
    # Apply t-SNE if more than 2 features
    if X_scaled.shape[1] > 2:
        X_combined = np.vstack([X_scaled, new_point_scaled])
        tsne = TSNE(n_components=2, random_state=42, 
                   perplexity=min(30, len(X_combined)-1))
        X_reduced = tsne.fit_transform(X_combined)
        X_vis, new_point_vis = X_reduced[:-1], X_reduced[-1:]
        return X_scaled, y, new_point_scaled, X_vis, new_point_vis, feature_cols, encoders, scaler
    else:
        return X_scaled, y, new_point_scaled, X_scaled, new_point_scaled, feature_cols, encoders, scaler

def validate_distance_metric_compatibility(distance_metric, X_scaled):
    """Validate if distance metric is compatible with the data"""
    if distance_metric == "cosine":
        # Cosine distance requires non-zero vectors
        zero_rows = np.all(X_scaled == 0, axis=1)
        if np.any(zero_rows):
            return False, "Cosine distance cannot be used with zero vectors. Try 'euclidean' or 'manhattan' instead."
    return True, ""

def run_knn_and_visualize(df, target_col, new_point_dict, k, distance_metric, weighting_method, problem_type=None):
    """Execute KNN algorithm and generate visualization"""
    X_scaled, y, new_point_scaled, X_vis, new_point_vis, feature_cols, encoders, scaler = preprocess_and_reduce(
        df, target_col, new_point_dict
    )
    
    # Validate distance metric compatibility
    is_valid, error_msg = validate_distance_metric_compatibility(distance_metric, X_scaled)
    if not is_valid:
        return None, None, None, None, error_msg
    
    if problem_type is None:
        problem_type = determine_problem_type(df, target_col)
    
    has_tsne = X_scaled.shape[1] > 2
    
    # Validate K value
    max_k = len(X_scaled)
    if k > max_k:
        return None, None, None, None, f"K value ({k}) cannot be larger than number of data points ({max_k}). Please choose K ≤ {max_k}."
    
    # Original KNN for accurate prediction
    ModelClass = KNeighborsClassifier if problem_type == "classification" else KNeighborsRegressor
    model = ModelClass(n_neighbors=k, metric=distance_metric, weights=weighting_method)
    
    model.fit(X_scaled, y)
    prediction = model.predict(new_point_scaled)[0]
    distances, neighbor_indices = model.kneighbors(new_point_scaled)
    
    # Visual KNN for consistent visualization
    if has_tsne:
        visual_model = ModelClass(n_neighbors=k, metric=distance_metric, weights=weighting_method)
        visual_model.fit(X_vis, y)
        visual_distances, visual_neighbor_indices = visual_model.kneighbors(new_point_vis)
    else:
        visual_neighbor_indices = neighbor_indices
    
    # Create neighbor details table
    neighbor_details = []
    for i, (dist, idx) in enumerate(zip(distances[0], neighbor_indices[0])):
        neighbor_data = {'Rank': i+1, 'Distance': f"{dist:.3f}"}
        for col in feature_cols:
            neighbor_data[col] = round(df.iloc[idx][col], 2) if isinstance(df.iloc[idx][col], (int, float)) else df.iloc[idx][col]
        neighbor_data[target_col] = df.iloc[idx][target_col]
        neighbor_details.append(neighbor_data)
    
    neighbor_df = pd.DataFrame(neighbor_details)
    
    # Generate outputs
    fig = create_visualization(X_vis, y, new_point_vis, neighbor_indices[0], 
                             visual_neighbor_indices[0], target_col, prediction, has_tsne, feature_cols)
    summary = create_algorithm_summary(problem_type, neighbor_df, k, distance_metric, weighting_method, has_tsne)
    
    return fig, prediction, neighbor_df, summary, None

def create_visualization(X_vis, y, new_point_vis, neighbor_indices, visual_neighbor_indices, target_col, prediction, has_tsne, feature_cols):
    """Create interactive visualization"""
    fig = go.Figure()
    
    is_classification = y.nunique() <= 20
    visual_neighbor_x = X_vis[visual_neighbor_indices, 0]
    visual_neighbor_y = X_vis[visual_neighbor_indices, 1]
    
    if is_classification:
        # Classification visualization
        unique_targets = sorted(y.unique())
        colors = px.colors.qualitative.Set1[:len(unique_targets)]
        color_map = dict(zip(unique_targets, colors))
        
        for target_val in unique_targets:
            mask = y == target_val
            fig.add_trace(go.Scatter(
                x=X_vis[mask, 0], y=X_vis[mask, 1], mode='markers',
                name=f'{target_col}={target_val}',
                marker=dict(color=color_map[target_val], size=8, opacity=0.7),
                hovertemplate=f'{target_col}=%{{text}}<extra></extra>',
                text=[target_val] * np.sum(mask)
            ))
        
        # Visual neighbors
        visual_neighbor_targets = y.iloc[visual_neighbor_indices]
        fig.add_trace(go.Scatter(
            x=visual_neighbor_x, y=visual_neighbor_y, mode='markers', 
            name='Visual Neighbors',
            marker=dict(color='orange', size=12, symbol='circle-open', line=dict(width=3)),
            hovertemplate='Visual Neighbor<br>%{text}<extra></extra>',
            text=[f'{target_col}={val}' for val in visual_neighbor_targets]
        ))
        
        # Original neighbors if different
        if has_tsne and not np.array_equal(neighbor_indices, visual_neighbor_indices):
            original_neighbor_x = X_vis[neighbor_indices, 0]
            original_neighbor_y = X_vis[neighbor_indices, 1]
            original_neighbor_targets = y.iloc[neighbor_indices]
            
            fig.add_trace(go.Scatter(
                x=original_neighbor_x, y=original_neighbor_y, mode='markers',
                name='Algorithm Neighbors',
                marker=dict(color='black', size=10, symbol='diamond-open', line=dict(width=2)),
                hovertemplate='Algorithm Neighbor<br>%{text}<extra></extra>',
                text=[f'{target_col}={val}' for val in original_neighbor_targets]
            ))
        
        # New point
        fig.add_trace(go.Scatter(
            x=new_point_vis[:, 0], y=new_point_vis[:, 1], mode='markers',
            name=f'New Point (Predicted: {prediction})',
            marker=dict(color='red', size=15, symbol='x', line=dict(width=3)),
            hovertemplate=f'New Point<br>Predicted: {prediction}<extra></extra>'
        ))
        
    else:
        # Regression visualization with color gradient
        fig.add_trace(go.Scatter(
            x=X_vis[:, 0], y=X_vis[:, 1], mode='markers',
            name='Data Points',
            marker=dict(size=8, opacity=0.7, color=y, colorscale='Viridis',
                       colorbar=dict(title=target_col), showscale=True),
            hovertemplate=f'{target_col}=%{{marker.color:.3f}}<extra></extra>'
        ))
        
        # Visual neighbors
        visual_neighbor_values = y.iloc[visual_neighbor_indices]
        fig.add_trace(go.Scatter(
            x=visual_neighbor_x, y=visual_neighbor_y, mode='markers', 
            name='Visual Neighbors',
            marker=dict(color='orange', size=12, symbol='circle-open', line=dict(width=3)),
            hovertemplate='Visual Neighbor<br>%{text}<extra></extra>',
            text=[f'Value: {val:.3f}' for val in visual_neighbor_values]
        ))
        
        # Original neighbors if different
        if has_tsne and not np.array_equal(neighbor_indices, visual_neighbor_indices):
            original_neighbor_x = X_vis[neighbor_indices, 0]
            original_neighbor_y = X_vis[neighbor_indices, 1]
            original_neighbor_values = y.iloc[neighbor_indices]
            
            fig.add_trace(go.Scatter(
                x=original_neighbor_x, y=original_neighbor_y, mode='markers',
                name='Algorithm Neighbors',
                marker=dict(color='black', size=10, symbol='diamond-open', line=dict(width=2)),
                hovertemplate='Algorithm Neighbor<br>%{text}<extra></extra>',
                text=[f'Value: {val:.3f}' for val in original_neighbor_values]
            ))
        
        # New point
        fig.add_trace(go.Scatter(
            x=new_point_vis[:, 0], y=new_point_vis[:, 1], mode='markers',
            name=f'New Point (Predicted: {prediction:.3f})',
            marker=dict(color='red', size=15, symbol='x', line=dict(width=3)),
            hovertemplate=f'New Point<br>Predicted: {prediction:.3f}<extra></extra>'
        ))
    
    # Connection lines to visual neighbors
    for i in range(len(visual_neighbor_indices)):
        fig.add_trace(go.Scatter(
            x=[new_point_vis[0, 0], visual_neighbor_x[i]], 
            y=[new_point_vis[0, 1], visual_neighbor_y[i]],
            mode='lines', line=dict(color='orange', width=1, dash='dot'),
            showlegend=False, hoverinfo='skip'
        ))
    
    title_text = "KNN Visualization"
    if has_tsne:
        title_text += " (t-SNE 2D Projection)"
    
    # Sử dụng tên cột thật khi có đúng 2 features (không có t-SNE)
    if not has_tsne and len(feature_cols) == 2:
        xaxis_title = feature_cols[0]
        yaxis_title = feature_cols[1]
    else:
        xaxis_title = "Dimension 1"
        yaxis_title = "Dimension 2"
    
    fig.update_layout(
        title=title_text,
        xaxis_title=xaxis_title, yaxis_title=yaxis_title,
        hovermode='closest', width=780, height=520
    )
    
    return fig

def create_algorithm_summary(problem_type, neighbor_df, k, distance_metric, weighting_method, has_tsne):
    """Generate algorithm summary"""
    distances = [float(d) for d in neighbor_df['Distance']]
    
    tsne_note = ""
    if has_tsne:
        tsne_note = "\n**Note:** t-SNE applied for visualization. Orange circles show visual neighbors, black diamonds show algorithm neighbors."
    
    if problem_type == "classification":
        class_counts = neighbor_df.iloc[:, -1].value_counts()
        class_dist = ", ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
        
        summary = f"""## Algorithm Summary
**K:** {k} | **Distance:** {distance_metric} | **Weighting:** {weighting_method}
**Neighbors Distribution:** {class_dist}\n
**Performance:** {len(neighbor_df)} neighbors | Distance range: {min(distances):.3f} - {max(distances):.3f}\n{tsne_note}"""
    else:
        neighbor_values = [float(val) for val in neighbor_df.iloc[:, -1].values]
        avg_value = np.mean(neighbor_values)
        
        summary = f"""## Algorithm Summary
**K:** {k} | **Distance:** {distance_metric} | **Weighting:** {weighting_method}
**Neighbor Values:** {[f"{val:.3f}" for val in neighbor_values[:5]]}{"..." if len(neighbor_values) > 5 else ""}
**Average:** {avg_value:.3f}
**Performance:** {len(neighbor_df)} neighbors | Distance range: {min(distances):.3f} - {max(distances):.3f}{tsne_note}"""
    
    return summary
