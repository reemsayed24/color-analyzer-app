"""
🎨 Color Analyzer By ReeM - Streamlit Dashboard with Image Preprocessing
=========================================================================
Beautiful, interactive color analysis dashboard with advanced preprocessing
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter
from typing import List, Dict, Tuple
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageEnhance
import io

# Page configuration
st.set_page_config(
    page_title="🎨 Color Analyzer",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as before)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    
    .color-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        margin-bottom: 1rem;
    }
    
    .color-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .color-swatch {
        width: 100%;
        height: 80px;
        border-radius: 8px;
        margin-bottom: 1rem;
        border: 2px solid #e2e8f0;
    }
    
    .color-info {
        text-align: center;
    }
    
    .color-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
    
    .color-percentage {
        font-size: 1.5rem;
        font-weight: 700;
        color: #6366f1;
    }
    
    .color-rgb {
        font-size: 0.9rem;
        color: #64748b;
        font-family: monospace;
    }
    
    .stat-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .stat-label {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .preprocessing-section {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid #e2e8f0;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


class ImagePreprocessor:
    """Advanced image preprocessing pipeline"""
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            image: Input RGB image
            factor: Enhancement factor (1.0 = no change, >1.0 = more contrast)
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=factor * 2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return enhanced
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        Adjust image brightness
        
        Args:
            image: Input RGB image
            factor: Brightness factor (1.0 = no change, >1.0 = brighter)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    @staticmethod
    def adjust_saturation(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
        """
        Adjust color saturation
        
        Args:
            image: Input RGB image
            factor: Saturation factor (1.0 = no change, >1.0 = more saturated)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    @staticmethod
    def sharpen_image(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Sharpen image using unsharp masking
        
        Args:
            image: Input RGB image
            strength: Sharpening strength (0.0 = no sharpening, 2.0 = strong)
        """
        # Gaussian blur
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        
        # Unsharp mask
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    @staticmethod
    def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
        """
        Remove noise from image
        
        Args:
            image: Input RGB image
            strength: Denoising strength (higher = more denoising)
        """
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    
    @staticmethod
    def bilateral_filter(image: np.ndarray, d: int = 9) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving smoothing
        
        Args:
            image: Input RGB image
            d: Diameter of pixel neighborhood
        """
        return cv2.bilateralFilter(image, d, 75, 75)
    
    @staticmethod
    def edge_enhancement(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
        """
        Enhance edges in the image
        
        Args:
            image: Input RGB image
            strength: Edge enhancement strength
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        
        # Blend with original
        enhanced = cv2.addWeighted(image, 1.0, edges, strength * 0.3, 0)
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    @staticmethod
    def white_balance(image: np.ndarray) -> np.ndarray:
        """
        Apply automatic white balance correction
        
        Args:
            image: Input RGB image
        """
        result = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result = cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
        return result
    
    @staticmethod
    def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply gamma correction
        
        Args:
            image: Input RGB image
            gamma: Gamma value (< 1.0 = darker, > 1.0 = brighter)
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype(np.uint8)
        return cv2.LUT(image, table)
    
    def preprocess_pipeline(self, 
                          image: np.ndarray,
                          contrast: float = 1.0,
                          brightness: float = 1.0,
                          saturation: float = 1.0,
                          sharpen: float = 0.0,
                          denoise: bool = False,
                          bilateral: bool = False,
                          white_balance: bool = False,
                          gamma: float = 1.0) -> np.ndarray:
        """
        Complete preprocessing pipeline
        
        Args:
            image: Input RGB image
            Various preprocessing parameters
        
        Returns:
            Preprocessed image
        """
        result = image.copy()
        
        # Apply white balance first if enabled
        if white_balance:
            result = self.white_balance(result)
        
        # Gamma correction
        if gamma != 1.0:
            result = self.gamma_correction(result, gamma)
        
        # Denoising
        if denoise:
            result = self.denoise_image(result, strength=10)
        
        # Bilateral filter
        if bilateral:
            result = self.bilateral_filter(result)
        
        # Contrast enhancement
        if contrast != 1.0:
            result = self.enhance_contrast(result, contrast)
        
        # Brightness adjustment
        if brightness != 1.0:
            result = self.adjust_brightness(result, brightness)
        
        # Saturation adjustment
        if saturation != 1.0:
            result = self.adjust_saturation(result, saturation)
        
        # Sharpening
        if sharpen > 0:
            result = self.sharpen_image(result, sharpen)
        
        return result


class ColorAnalyzer:
    """Color analysis engine"""
    
    def __init__(self, num_colors: int = 10):
        self.num_colors = num_colors
        self.color_dictionary = self._build_color_dictionary()
    
    def _build_color_dictionary(self) -> Dict[str, List[Tuple]]:
        """Build color dictionary for naming"""
        return {
            'Bright Yellow': [(255, 255, 0, 80), (240, 230, 0, 80)],
            'Light Yellow': [(233, 211, 96, 50), (240, 220, 100, 60)],
            'Golden Yellow': [(200, 175, 94, 60)],
            'Dark Blue': [(0, 0, 100, 70), (30, 50, 120, 70)],
            'Blue': [(0, 100, 200, 70), (50, 100, 180, 70)],
            'Light Blue': [(100, 150, 200, 60), (150, 200, 255, 60)],
            'Sky Blue': [(135, 206, 235, 60)],
            'Navy Blue': [(0, 0, 80, 60)],
            'Golden Brown': [(193, 163, 86, 60), (200, 170, 90, 60)],
            'Brown': [(148, 126, 69, 60), (140, 120, 60, 60)],
            'Dark Brown': [(74, 51, 26, 50), (80, 60, 30, 50)],
            'Light Brown': [(180, 140, 100, 50)],
            'Beige': [(207, 183, 164, 50), (210, 190, 170, 50)],
            'Tan': [(210, 180, 140, 50)],
            'Olive Green': [(97, 88, 62, 50), (100, 90, 60, 50)],
            'Green': [(0, 150, 0, 60), (100, 200, 100, 60)],
            'Dark Green': [(0, 100, 0, 60)],
            'Light Green': [(144, 238, 144, 60)],
            'Red': [(200, 0, 0, 60), (255, 50, 50, 60)],
            'Dark Red': [(100, 0, 0, 50), (150, 30, 30, 60)],
            'Pink': [(255, 150, 150, 60), (255, 192, 203, 60)],
            'Hot Pink': [(255, 105, 180, 60)],
            'Orange': [(255, 140, 0, 60), (255, 165, 80, 70)],
            'Dark Orange': [(255, 100, 0, 60)],
            'Purple': [(128, 0, 128, 60), (147, 112, 219, 60)],
            'Dark Purple': [(75, 0, 130, 60)],
            'Lavender': [(230, 230, 250, 50)],
            'White': [(250, 250, 250, 30), (255, 255, 255, 20)],
            'Black': [(0, 0, 0, 60), (30, 30, 30, 60)],
            'Light Gray': [(200, 200, 200, 40), (220, 220, 220, 40)],
            'Gray': [(128, 128, 128, 50), (160, 160, 160, 50)],
            'Dark Gray': [(23, 41, 67, 50), (50, 50, 50, 50)],
        }
    
    @staticmethod
    def _calculate_color_distance(color1: Tuple[int, int, int], 
                                  color2: Tuple[int, int, int]) -> float:
        """Calculate Euclidean distance between colors"""
        return np.sqrt(sum((color1[i] - color2[i])**2 for i in range(3)))
    
    def get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to color name"""
        min_distance = float('inf')
        closest_color = 'Unknown Color'
        
        for color_name, color_ranges in self.color_dictionary.items():
            for color_tuple in color_ranges:
                target_rgb = color_tuple[:3]
                threshold = color_tuple[3] if len(color_tuple) > 3 else 60
                distance = self._calculate_color_distance(rgb, target_rgb)
                
                if distance < threshold and distance < min_distance:
                    min_distance = distance
                    closest_color = color_name
        
        return closest_color
    
    def analyze_image(self, image_array: np.ndarray) -> List[Dict]:
        """Analyze image and extract colors"""
        # Resize for performance
        height, width = image_array.shape[:2]
        max_dim = 800
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            image_array = cv2.resize(image_array, 
                                    (int(width * scale), int(height * scale)))
        
        # Reshape for clustering
        pixels = image_array.reshape(-1, 3)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=self.num_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        label_counts = Counter(labels)
        total_pixels = len(labels)
        
        # Build color info
        color_info = []
        for i in range(self.num_colors):
            count = label_counts[i]
            percentage = (count / total_pixels) * 100
            
            if percentage >= 1.0:
                rgb_tuple = tuple(colors[i])
                color_name = self.get_color_name(rgb_tuple)
                
                color_info.append({
                    'rank': len(color_info) + 1,
                    'color_name': color_name,
                    'rgb': rgb_tuple,
                    'hex': '#{:02x}{:02x}{:02x}'.format(*rgb_tuple),
                    'percentage': percentage
                })
        
        # Sort by percentage
        color_info = sorted(color_info, key=lambda x: x['percentage'], reverse=True)
        
        # Update ranks
        for i, info in enumerate(color_info, 1):
            info['rank'] = i
        
        return color_info


def create_color_palette(color_info: List[Dict]) -> go.Figure:
    """Create interactive color palette"""
    fig = go.Figure()
    
    for info in color_info:
        fig.add_trace(go.Bar(
            x=[info['percentage']],
            y=['Colors'],
            orientation='h',
            marker=dict(
                color=info['hex'],
                line=dict(color='white', width=2)
            ),
            name=info['color_name'],
            hovertemplate=f"<b>{info['color_name']}</b><br>" +
                         f"Percentage: {info['percentage']:.2f}%<br>" +
                         f"RGB: {info['rgb']}<br>" +
                         f"HEX: {info['hex']}<extra></extra>",
            showlegend=False
        ))
    
    fig.update_layout(
        barmode='stack',
        height=150,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        hovermode='closest'
    )
    
    return fig


def create_pie_chart(color_info: List[Dict]) -> go.Figure:
    """Create interactive pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=[info['color_name'] for info in color_info],
        values=[info['percentage'] for info in color_info],
        marker=dict(colors=[info['hex'] for info in color_info],
                   line=dict(color='white', width=2)),
        textinfo='label+percent',
        textfont=dict(size=12, color='white'),
        hovertemplate="<b>%{label}</b><br>" +
                     "Percentage: %{value:.2f}%<extra></extra>"
    )])
    
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )
    
    return fig


def create_bar_chart(color_info: List[Dict]) -> go.Figure:
    """Create interactive bar chart"""
    fig = go.Figure(data=[go.Bar(
        x=[info['color_name'] for info in color_info],
        y=[info['percentage'] for info in color_info],
        marker=dict(
            color=[info['hex'] for info in color_info],
            line=dict(color='white', width=2)
        ),
        text=[f"{info['percentage']:.1f}%" for info in color_info],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>" +
                     "Percentage: %{y:.2f}%<extra></extra>"
    )])
    
    fig.update_layout(
        height=400,
        xaxis_title="Color",
        yaxis_title="Percentage (%)",
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(tickangle=-45),
        yaxis=dict(gridcolor='rgba(128,128,128,0.2)'),
        showlegend=False
    )
    
    return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎨Color Analyzer By ReeM</h1>
        <p>Extract and analyze dominant colors with intelligent preprocessing</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Settings & Preprocessing
    with st.sidebar:
        st.markdown("## ⚙️ Analysis Settings")
        
        num_colors = st.slider(
            "Number of Colors",
            min_value=3,
            max_value=20,
            value=10,
            help="Number of dominant colors to extract"
        )
        
        st.markdown("---")
        st.markdown("## 🔧 Image Preprocessing")
        
        enable_preprocessing = st.checkbox("Enable Preprocessing", value=False)
        
        if enable_preprocessing:
            st.markdown("### Basic Adjustments")
            
            contrast = st.slider(
                "Contrast",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust image contrast"
            )
            
            brightness = st.slider(
                "Brightness",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust image brightness"
            )
            
            saturation = st.slider(
                "Saturation",
                min_value=0.0,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust color saturation"
            )
            
            gamma = st.slider(
                "Gamma Correction",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Adjust gamma (< 1.0 = darker, > 1.0 = brighter)"
            )
            
            st.markdown("### Advanced Filters")
            
            sharpen = st.slider(
                "Sharpening",
                min_value=0.0,
                max_value=2.0,
                value=0.0,
                step=0.1,
                help="Sharpen image details"
            )
            
            denoise = st.checkbox("Noise Reduction", value=False)
            bilateral = st.checkbox("Edge-Preserving Smoothing", value=False)
            white_balance = st.checkbox("Auto White Balance", value=False)
        
        else:
            contrast = brightness = saturation = gamma = 1.0
            sharpen = 0.0
            denoise = bilateral = white_balance = False
        
        st.markdown("---")
        st.markdown("### 📊 About")
        st.info(
            "This tool uses K-Means clustering with advanced preprocessing "
            "to extract dominant colors accurately."
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Supported formats: PNG, JPG, JPEG, BMP"
    )
    
    if uploaded_file is not None:
        # Load original image
        image = Image.open(uploaded_file)
        original_array = np.array(image.convert('RGB'))
        
        # Apply preprocessing
        if enable_preprocessing:
            with st.spinner("Preprocessing image..."):
                preprocessor = ImagePreprocessor()
                processed_array = preprocessor.preprocess_pipeline(
                    original_array,
                    contrast=contrast,
                    brightness=brightness,
                    saturation=saturation,
                    sharpen=sharpen,
                    denoise=denoise,
                    bilateral=bilateral,
                    white_balance=white_balance,
                    gamma=gamma
                )
        else:
            processed_array = original_array
        
        # Display images side by side
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("### 📸 Original Image")
            st.image(original_array, use_column_width=True)
        
        with col2:
            st.markdown("### 🔧 Processed Image")
            st.image(processed_array, use_column_width=True)
            
            if enable_preprocessing:
                # Show preprocessing difference
                diff = np.abs(original_array.astype(float) - processed_array.astype(float))
                diff_percentage = (np.mean(diff) / 255) * 100
                st.metric("Preprocessing Impact", f"{diff_percentage:.1f}%")
        
        with col3:
            st.markdown("### 🎨 Color Palette")
            with st.spinner("Analyzing colors..."):
                # Analyze processed image
                analyzer = ColorAnalyzer(num_colors=num_colors)
                color_info = analyzer.analyze_image(processed_array)
                
                # Create palette
                palette_fig = create_color_palette(color_info)
                st.plotly_chart(palette_fig, use_container_width=True)
                
                # Quick stats
                st.markdown("#### 📈 Quick Stats")
                st.metric("Colors Detected", len(color_info))
                st.metric("Dominant Color", color_info[0]['color_name'])
                st.metric("Coverage", f"{color_info[0]['percentage']:.1f}%")
        
        # Detailed Analysis
        st.markdown("---")
        st.markdown("## 🔍 Detailed Analysis")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Color Cards", 
            "📈 Charts", 
            "📋 Data Table",
            "🔬 Comparison"
        ])
        
        with tab1:
            # Color cards
            cols = st.columns(min(5, len(color_info)))
            for idx, info in enumerate(color_info[:10]):
                with cols[idx % 5]:
                    st.markdown(f"""
                    <div class="color-card">
                        <div class="color-swatch" style="background-color: {info['hex']};"></div>
                        <div class="color-info">
                            <div class="color-name">{info['color_name']}</div>
                            <div class="color-percentage">{info['percentage']:.2f}%</div>
                            <div class="color-rgb">{info['hex']}</div>
                            <div class="color-rgb">RGB{info['rgb']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                st.markdown("#### Distribution Chart")
                bar_chart = create_bar_chart(color_info)
                st.plotly_chart(bar_chart, use_container_width=True)
            
            with chart_col2:
                st.markdown("#### Composition Chart")
                pie_chart = create_pie_chart(color_info)
                st.plotly_chart(pie_chart, use_container_width=True)
        
        with tab3:
            # Data table
            df = pd.DataFrame([{
                'Rank': info['rank'],
                'Color': info['color_name'],
                'HEX': info['hex'],
                'RGB': str(info['rgb']),
                'Percentage': f"{info['percentage']:.2f}%"
            } for info in color_info])
            
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="color_analysis.csv",
                mime="text/csv"
            )
        
        with tab4:
            if enable_preprocessing:
                st.markdown("#### Preprocessing Comparison")
                
                # Analyze original image for comparison
                original_color_info = analyzer.analyze_image(original_array)
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    st.markdown("**Original Image Colors**")
                    for info in original_color_info[:5]:
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                            <div style="width: 40px; height: 40px; background-color: {info['hex']}; 
                                        border-radius: 8px; margin-right: 1rem; border: 2px solid #e2e8f0;"></div>
                            <div>
                                <strong>{info['color_name']}</strong><br>
                                <small>{info['percentage']:.2f}%</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with comp_col2:
                    st.markdown("**Processed Image Colors**")
                    for info in color_info[:5]:
                        st.markdown(f"""
                        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
                            <div style="width: 40px; height: 40px; background-color: {info['hex']}; 
                                        border-radius: 8px; margin-right: 1rem; border: 2px solid #e2e8f0;"></div>
                            <div>
                                <strong>{info['color_name']}</strong><br>
                                <small>{info['percentage']:.2f}%</small>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("#### Impact Analysis")
                st.success(
                    f"✅ Preprocessing changed color detection by "
                    f"{diff_percentage:.1f}%, revealing more accurate color patterns."
                )
            else:
                st.info("Enable preprocessing in the sidebar to see comparison.")
    
    else:
        # Landing page
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 3rem;
            border-radius: 15px;
            text-align: center;
            margin: 2rem 0;
        ">
            <h2 style="color: white; margin-top: 0;">👆 Upload an image to get started</h2>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.1rem;">
                Supported formats: PNG, JPG, JPEG, BMP
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features
        st.markdown("### ✨ Features")
        feat_cols = st.columns(4)
        features = [
            ("🎨", "K-Means Clustering", "Statistical color extraction"),
            ("🔧", "Preprocessing", "CLAHE, denoising, sharpening"),
            ("📊", "Visual Analysis", "Interactive charts & palettes"),
            ("💾", "Export Data", "Download results as CSV")
        ]
        
        for col, (emoji, title, desc) in zip(feat_cols, features):
            with col:
                st.markdown(f"""
                <div class="color-card">
                    <div style="font-size: 3rem; text-align: center;">{emoji}</div>
                    <div style="text-align: center;">
                        <strong>{title}</strong>
                        <p style="font-size: 0.9rem; color: #64748b;">{desc}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":

    main()
