import PIL
import streamlit as st
import tensorflow as tf
import numpy as np

st.set_page_config(
    page_title="Food Image Classification",
    page_icon="♋",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model("./model/model.h5")
    return model


st.title("Food Image Classification ")

pic = st.file_uploader(
    label="Upload a picture",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    help="Upload a picture of your food image",
)

if st.button("Predict"):
    if pic != None:
        st.header("Results")

        cols = st.columns([1, 2])
        with cols[0]:
            st.image(pic, caption=pic.name, use_column_width=True)

        with cols[1]:
            labels = ['adhirasam',
                'aloo_gobi',
                'aloo_matar',
                'aloo_methi',
                'aloo_shimla_mirch',
                'aloo_tikki',
                'anarsa',
                'ariselu',
                'bandar_laddu',
                'basundi',
                'bhatura',
                'bhindi_masala',
                'biryani',
                'boondi',
                'butter_chicken',
                'chak_hao_kheer',
                'cham_cham',
                'chana_masala',
                'chapati',
                'chhena_kheeri',
                'chicken_razala',
                'chicken_tikka',
                'chicken_tikka_masala',
                'chikki',
                'daal_baati_churma',
                'daal_puri',
                'dal_makhani',
                'dal_tadka',
                'dharwad_pedha',
                'doodhpak',
                'double_ka_meetha',
                'dum_aloo',
                'gajar_ka_halwa',
                'gavvalu',
                'ghevar',
                'gulab_jamun',
                'imarti',
                'jalebi',
                'kachori',
                'kadai_paneer',
                'kadhi_pakoda',
                'kajjikaya',
                'kakinada_khaja',
                'kalakand',
                'karela_bharta',
                'kofta',
                'kuzhi_paniyaram',
                'lassi',
                'ledikeni',
                'litti_chokha',
                'lyangcha',
                'maach_jhol',
                'makki_di_roti_sarson_da_saag',
                'malapua',
                'misi_roti',
                'misti_doi',
                'modak',
                'mysore_pak',
                'naan',
                'navrattan_korma',
                'palak_paneer',
                'paneer_butter_masala',
                'phirni',
                'pithe',
                'poha',
                'poornalu',
                'pootharekulu',
                'qubani_ka_meetha',
                'rabri',
                'rasgulla',
                'ras_malai',
                'sandesh',
                'shankarpali',
                'sheera',
                'sheer_korma',
                'shrikhand',
                'sohan_halwa',
                'sohan_papdi',
                'sutar_feni',
                'unni_appam'
            ]

            model = load_model()

            with st.spinner("Predicting..."):
                img = PIL.Image.open(pic)
                img_reshaped = img.resize((256, 256))
                
                img_rescale=np.asarray(img_reshaped)/255
                img_predict=model.predict(img_rescale[np.newaxis,...])
                item=np.argmax(img_predict)
                st.write(f"**Item name:**`{labels[item]}`")

                # prediction = model.predict(img)
                # prediction = tf.nn.softmax(prediction)

                # score = tf.reduce_max(prediction)
                # score = tf.round(score * 100, 2)

                # prediction = tf.argmax(prediction, axis=1)
                # prediction = prediction.numpy()
                # prediction = prediction[0]

                # disease = labels[prediction].title()
                # st.write(f"**Prediction:** `{disease}`")
                # st.write(f"**Confidence:** `{score:.2f}%`")
                # st.info(f"The model predicts that the lesion is a **{prediction}** with a confidence of {score}%")

        # st.warning(
        #     ":warning: This is not a Food Image. Please provide an clear food image."
        # )
    else:
        st.error("Please upload an image")
