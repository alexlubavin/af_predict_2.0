import streamlit as st
from streamlit.logger import get_logger
from tensorflow import keras
import numpy as np
from keras.models import load_model
model_loaded = load_model('af_predictor_model_4.h5')

def prediction(ao, la, lv, rv, ra, pa, lvaw, lvpw, ef, ar, mr, tr, cv):
    
    ao1 = (ao - 1.8) / (5.4 - 1.8)
    la1 = (la - 2.4) / (7.8 - 2.4)
    lv1 = (lv - 2.4) / (8.1 - 2.4)
    rv1 = (rv - 1.5) / (4.8 - 1.5)
    ra1 = (ra - 2.3) / (6.0 - 2.3)
    pa1 = (pa - 1.0) / (3.6 - 1.0)
    lvaw1 = (lvaw - 0.7) / (2.6 - 0.7)
    lvpw1 = (lvpw - 0.7) / (2.2 - 0.7)
    ef1 = (lv - 18.0) / (75.0 - 18.0)       # должна быть ef
    ar1 = (ar - 0) / (2.5 - 0)
    mr1 = (mr - 0) / (3.5 - 0)
    tr1 = (tr - 0) / (3.5 - 0)
    cv1 = (cv - 0) / (3.5 - 0)

    x = [ao1, la1, lv1, rv1, ra1, pa1, lvaw1, lvpw1, ef1, ar1, mr1, tr1, cv1]
    x = np.array(x)
    nn = np.expand_dims(x, axis=0)
    res = model_loaded.predict(nn)
    
    return res

# result = prediction(3.3, 4.4, 3.4, 4.4, 2.0, 1.2, 1,2, 34.0, 1.0, 2.0, 2.0, 2.0)

# st.write(round(result[0][0], 2))


# DESKTOP
LOGGER = get_logger(__name__)
def run():
    st.set_page_config(
        page_title="AF-predict-2.0",
        page_icon="https://img.icons8.com/?size=100&id=956&format=png&color=FA5252"
    )
    
    st.image('https://img.icons8.com/?size=100&id=956&format=png&color=FA5252',
             caption=None, 
             width=75, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    
    st.write(" # Прогнозирование развития фибрилляции предсердий (ФП)")
    st.subheader(""" при помощи нейронной сети """)
    
    st.subheader(""" ВВЕДИТЕ ДАННЫЕ ЭХОКАРДИОГРАФИИ """)
    
    st.image('positions.jpg', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
    
    #Input_data_columns
    col1, col2, col3, col4 = st.columns(4, gap="small", vertical_alignment="top")
    
    with col1:

        ao = st.number_input('1 - восходящая аорта, см', min_value=1.8, max_value=5.4, value=None, step=0.1, 
                             help='Введите диаметр восходящей аорты в см, например - 3,4', 
                             placeholder=None, disabled=False, label_visibility="visible")
        la = st.number_input('2 - левое предсердие, см', min_value=2.4, max_value=7.8, value=None, step=0.1, 
                             help='Введите размер левого предсердия в см, например - 3,6', 
                             placeholder=None, disabled=False, label_visibility="visible")        
        lv = st.number_input('3 - КДР левого желудочка, см', min_value=2.4, max_value=8.1, value=None, step=0.1, 
                             help='Введите конечный диастолический размер левого желудочка в см, например - 4,9', 
                             placeholder=None, disabled=False, label_visibility="visible")
        rv = st.number_input('4 - правый желудочек, см', min_value=1.5, max_value=4.8, value=None, step=0.1, 
                             help='Введите размер правого желудочка в см, например - 2,6', 
                             placeholder=None, disabled=False, label_visibility="visible")
        

    with col2:
     
        lvaw = st.number_input('5 - межжелудочковая перегородка, см', min_value=0.7, max_value=2.6, value=None, step=0.1, 
                             help='Введите толщину межжелудочковой перегородки в см, например - 0,9', 
                             placeholder=None, disabled=False, label_visibility="visible")
        lvpw = st.number_input('6 - задняя стенка левого желудочка, см', min_value=0.7, max_value=2.6, value=None, step=0.1, 
                             help='Введите толщину задней стенки левого желудочка в см, например - 1,0', 
                             placeholder=None, disabled=False, label_visibility="visible")        
        pa = st.number_input('7 - легочная артерия, см', min_value=1.0, max_value=3.6, value=None, step=0.1, 
                             help='Введите диаметр легочной артерии в см, например - 2,1', 
                             placeholder=None, disabled=False, label_visibility="visible")

    with col3:

        ra = st.number_input('8 - правое предсердие, см', min_value=2.3, max_value=6.0, value=None, step=0.1, 
                             help='Введите поперечный размер правого предсердия в см, например - 3,9', 
                             placeholder=None, disabled=False, label_visibility="visible")
        ef = st.number_input('9 - ФВ левого желудочка, %', min_value=18.0, max_value=75.0, value=None, step=1.0, 
                             help='Введите фракцию выброса левого желудочка (Симпсон) в %, например - 62', 
                             placeholder=None, disabled=False, label_visibility="visible")
        ar = st.number_input('10 - аортальная регургитация, степень', min_value=0.0, max_value=4.0, value=None, step=0.5, 
                             help='Введите степень аортальной регургитации ("+" = 1.0, "++" = 2.0; "++ - +++" = 2.5 и т.д.), например - 1,5', 
                             placeholder=None, disabled=False, label_visibility="visible")

    with col4:

        mr = st.number_input('11 - митральная регургитация, степень', min_value=0.0, max_value=4.0, value=None, step=0.5, 
                             help='Введите степень митральной регургитации ("+" = 1.0, "++" = 2.0; "++ - +++" = 2.5 и т.д.), например - 1,5', 
                             placeholder=None, disabled=False, label_visibility="visible")
        tr = st.number_input('12 - трикуспидальная регургитация, степень', min_value=0.0, max_value=4.0, value=None, step=0.5, 
                             help='Введите степень трикуспидальной регургитации ("+" = 1.0, "++" = 2.0; "++ - +++" = 2.5 и т.д.), например - 1,5', 
                             placeholder=None, disabled=False, label_visibility="visible")   
        cv = st.number_input('13 - нижняя полая вена, см', min_value=0.0, max_value=3.5, value=None, step=0.1, 
                             help='Введите диаметр нижней полой вены на выдохе в см, например - 1,6', 
                             placeholder=None, disabled=False, label_visibility="visible")
        
        
        
        
    if st.button("Получить прогноз"):
        try:
            result = prediction(ao, la, lv, rv, ra, pa, lvaw, lvpw, ef, ar, mr, tr, cv)
            out = round(result[0][0], 2)
            st.write('Значение на выходе нейросети - ' + str(out))
        except:
            st.error('Проверьте правильность заполнения формы!', icon="⚠️")
        
        if out <= 0.75:
            st.write('Низкая вероятность развития ФП - менее 4% в течение 18 месяцев')
        else:
            st.write('Высокая вероятность развития ФП - более 12% в течение 18 месяцев')
            
            
            
            
            
    
    st.link_button('Котляров С.Н., Любавин А.В.🖂 Прогнозирование фибрилляции предсердий по основным показателям трансторакальной эхокардиографии при помощи нейронной сети. Креативная кардиология. 2023; 17 (4): 481–90. DOI: 10.24022/1997-3187-2023-17-4-481-490', 
                   'https://cardiology-journal.com/catalog/detail.php?SECTION_ID=25942&ID=1142825', 
                   help=None, type="secondary", disabled=False, use_container_width=False)    
    
    st.image('curve.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto") 
    
    st.write('График логистической регрессии зависимости развития ФП от значения выхода нейросети')

    st.image('kmf1.png', caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto") 
    
    st.write('Кривые Каплана-Мейера развития ФП в зависимости от значения выхода нейросети')
    
    st.link_button('Котляров С.Н., Любавин А.В. Прогнозирование фибрилляции предсердий по основным показателям трансторакальной эхокардиографии при помощи нейронной сети. Креативная кардиология. 2023; 17 (4): 481–90. DOI: 10.24022/1997-3187-2023-17-4-481-490', 
               'https://cardiology-journal.com/catalog/detail.php?SECTION_ID=25942&ID=1142825', 
               help=None, type="secondary", disabled=False, use_container_width=False)
    
    
    
    col1, col2, col3 = st.columns(3, gap="small", vertical_alignment="top")
    
    with col1:
        
        if st.button('Написать e-mail', use_container_width=True):
            st.write('alexlubavin48@gmail.com')
    
    with col2:
        
        if st.button('Поддержать финансово', use_container_width=True):
            st.write('2202 2068 5584 3004')
            
    with col3:

        if st.button('Связаться в WhatsApp', use_container_width=True):
            st.write('+7(915)857-88-65')
    
if __name__ == "__main__":
    run()
