# Week 5: Fashion MNIST Classifier - Streamlit Web Application
# Save this as: app.py
# Run with: streamlit run app.py
from operator import index

import pandas
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import io

from main import x_train, y_train

# Page configuration
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Class names
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Emoji mapping for classes

CLASS_EMOJIS = {
    'T-shirt/top': '👕',
    'Trouser': '👖',
    'Pullover': '🧥',
    'Dress': '👗',
    'Coat': '🧥',
    'Sandal': '👡',
    'Shirt': '👔',
    'Sneaker': '👟',
    'Bag': '👜',
    'Ankle boot': '🥾'
}


@st.cache_resource
def load_model(model_path):
    try:
        model  =  keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f'Model load error : {str(e)}')
        return  None


def preprocess_image(image):
  if (len(image.shape) == 3):
      image = cv2.cvtColor(image  , cv2.COLOR_RGB2GRAY)
  image =  cv2.resize(image,(28,28))

  image = image.astype('float32') / 255.0
  image = image.reshape(1,28,28,1)

  return image



def predict_image(image_model , image) :
    prediction = image_model.predict(image)
    predicted_class  = np.argmax(prediction[0])
    details_of_conf  =  prediction[0][predicted_class]


    return predicted_class , details_of_conf  , prediction[0]

def plot_pred_bars(predictions , class_names):
    fig , ax  = plt.subplots(figsize=(10,10))

    axis_ind  = np.arange(len(class_names))
    colors =  ['#1f77b4' if pred ==  max(predictions) else '#7f7f7f'  for pred in predictions]
    ax.barh(axis_ind , predictions , color=colors )
    ax.set_yticks(axis_ind)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('confidence' , fontsize  = 12)
    ax.set_title('Prediction Confidence for All Classes', fontsize=14, pad=20)
    ax.set_xlim([0, 1])


    for i , v in enumerate(predictions):
        ax.text(v + 0.01, i, f'{v * 100:.1f}%',
                va='center', fontsize=10)


    plt.tight_layout()
    return fig



def main():
    st.markdown('<h1 class="main-header">👕 Fashion MNIST Classifier</h1>',
                unsafe_allow_html=True)

    st.markdown("### AI-Powered Fashion Item Recognition")
    st.markdown('---')

    with st.sidebar:
        st.image('https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/doc/img/fashion-mnist-sprite.png' , use_container_width=True )
        st.markdown('## About')
        st.info(""" This app uses a Convolutional Neural Network (CNN) 
        to classify fashion items into 10 categories.
        
        **Categories:**
        - 👕 T-shirt/top
        - 👖 Trouser
        - 🧥 Pullover
        - 👗 Dress
        - 🧥 Coat
        - 👡 Sandal
        - 👔 Shirt
        - 👟 Sneaker
        - 👜 Bag
        - 🥾 Ankle boot""")

        st.markdown('## Model Selection')
        model_choice  = st.selectbox(
            "Choose a Model",
            ['DRESSES-2'  , 'DRESSES-1'  , 'DRESSES-3'],
            index=0

        )

        st.markdown('## Settings')
        show_conf  = st.checkbox("Show Confidence Bars", value=True)
        show_proccessed_img  = st.checkbox("Show Proccessed Image"  , value=True)


    model_path = None
    if model_choice == "DRESSES-2":
        model_path =  'models/fashion_mnist_best.h5'
    elif model_choice == "DRESSES-1"  :
        model_path = "models/fashion_mnist_baseline.h5"
    else :
        model_path = "models/fashion_mnist_super.h5"
    model  = load_model(model_path)

    if model is None:
            st.error('⚠️ Model not found!')
            return

    st.success(f'{model_choice} loaded successfully!')

    tab1 , tab2 , tab3  =  st.tabs(["📤 Upload Image", "🎲 Test Random Samples", "📊 Model Info"])
    with tab1:
        st.markdown('<h2 class="sub-header">Upload Your Image</h2>',
                    unsafe_allow_html=True)


        col1 , col2  =  st.columns([1,1])

        with col1:
             uploaded_file =  st.file_uploader("Upload Image" ,  type=["jpg", "jpeg", "png"], help="Upload a grayscale or color image of a fashion item")

             if uploaded_file is not None:
                  image = Image.open(uploaded_file)
                  st.image(image, caption='Uploaded Image')

                  img_aray = np.array(image)
                  proccessed_img = preprocess_image(img_aray)

                  if show_proccessed_img:
                      st.markdown("**Preprocessed Image (28x28 grayscale)**")
                      st.image(proccessed_img.reshape(28,28), caption='Preprocessed Image'  , width=200 , clamp = True)

        with col2:
            if uploaded_file is not None:


                with st.spinner('Classifying using NN...'):
                     predicted_class, details_of_conf, prediction = predict_image(model, proccessed_img)
                # Display results

                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                st.markdown("### Prediction Result")

                predicted_label = CLASS_NAMES[predicted_class]
                emoji = CLASS_EMOJIS.get(predicted_label, "🎽")

                st.markdown(f"## {emoji} **{predicted_label}**")
                st.markdown(f"### Confidence: **{details_of_conf * 100:.2f}%**")
                st.markdown('</div>', unsafe_allow_html=True)


                st.progress(float(details_of_conf))
                if (details_of_conf > 0.9) :
                    st.success("🎯 Very High Confidence!")
                elif details_of_conf > 0.7 :
                    st.info("✅ Good Confidence")
                else:
                    st.warning("⚠️ Low Confidence - Image may be unclear")

                if show_conf :
                    st.markdown("### Confidence Bars")
                    fig = plot_pred_bars(prediction , CLASS_NAMES)
                    st.pyplot(fig)
                    plt.close()

    with tab2:
        st.markdown('<h2 class="sub-header">Test with Random Samples</h2>',
                    unsafe_allow_html=True)

        st.info("Load random samples from the test dataset to see model predictions")

        if st.button("🎲 Load Random Samples"  ,  type='primary' ):
            (_ , _)  , (x_test , y_test)  = keras.datasets.fashion_mnist.load_data()
            x_test =   x_test.astype('float32') / 255.0

            choice_selection  =  np.random.choice(len(x_test) , 6  , replace  =False)
            cols  = st.columns(3)

            correct = []

            for i , img_sel in enumerate(choice_selection):
                with cols[i % 3] :
                    img = x_test[img_sel]
                    true_label = CLASS_NAMES[y_test[img_sel]]
                    proccsed = img.reshape(1,28,28,1)
                    predicted_class, details_of_conf, prediction = predict_image(model, proccsed)
                    predicted_label  = CLASS_NAMES[predicted_class]

                    st.image(img, caption=f'Predicted Label: {predicted_label}')
                    isCorrect  =   predicted_label == true_label

                    correct.append(isCorrect)
                    color =  "green" if isCorrect else "red"
                    result = "✅ Correct" if isCorrect else "❌ Incorrect"

                    st.markdown(f"**True:{true_label}**")
                    st.markdown(
                        f"""<span style='color:{color}; font-weight:bold;'>Predicted: {predicted_label}</span>""",
                        unsafe_allow_html=True
                    )

                    st.markdown(f"**Confidence:** {details_of_conf*100:.1f}%")
                    st.markdown(f"**{result}**")

            st.markdown(f"**correct {len([x for x in correct if x is True])} out of {len(choice_selection)}**")

    with tab3:
        st.markdown('<h2 class="sub-header">Model Information</h2>',
                    unsafe_allow_html=True)

        col1 , col2  = st.columns(2)


        with col1:
            st.markdown('### 📋 Model Architecture')
            st.markdown(f"""
 - **Input Shape:** 28x28x1 (grayscale)
            - **Total Layers:** {len(model.layers)}
            - **Trainable Parameters:** {model.count_params():,}
            - **Optimizer:** Adam
            - **Loss Function:** Sparse Categorical Crossentropy
 
""")
            st.markdown("### 🎯 Training Details")
            st.markdown("""
                - **Dataset:** Fashion MNIST
                - **Training Samples:** 60,000
                - **Test Samples:** 10,000
                - **Classes:** 10
                - **Image Size:** 28x28 pixels
                """)

        with col2:
            st.markdown("### 🏗️ Architecture Layers")
            with st.expander(
                "View Layer Details"
            ):

                rows  = []
                for i , layer in enumerate(model.layers):
                    rows.append({     "Index": i +1,
        "Layer Name": layer.name,
        "Layer Class": layer.__class__.__name__})

                df = pandas.DataFrame(rows)
                st.dataframe(df , hide_index= True)

                st.markdown("### 📊 Performance Metrics")
                st.markdown("""
                              The model achieves:
                              - **Training Accuracy:** ~93%
                              - **Validation Accuracy:** ~91%
                              - **Test Accuracy:** ~90%

                              Check the  `results/` folder for detailed metrics!
                              """)

    st.markdown("---")
    st.markdown("""  <div style='text-align: center; color: gray;'>
        <p>Built with ❤️ using TensorFlow and Streamlit</p>
        <p>Fashion MNIST Dataset by Zalando Research</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()




