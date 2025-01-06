import numpy as np
import re
import pandas as pd
import demoji
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import streamlit as st
import os
from datetime import datetime
from collections import defaultdict
from PIL import Image
from collections import Counter

# TÍTULO
st.title('Análisis del Chat de Yarilin e Ismaelin')

# Ruta fija del archivo de WhatsApp
RUTA_CHAT = 'resources/chat_whatsapp.txt'  # Ajusta esta ruta según tu archivo

# ---------------- FUNCIONES NECESARIAS ----------------
def inicia_con_fecha_y_hora(linea):
    """Verifica si una línea comienza con fecha y hora."""
    patron = r'^\[([1-9]|1[0-2])/(0?[1-9]|[1-2][0-9]|3[0-1])/2[0-9], ([0-1]?[0-9]|2[0-3]):([0-5][0-9]):([0-5][0-9])\s*([AP][M])\]'
    return bool(re.match(patron, linea))

def limpiar_linea(linea):
    """Limpia caracteres innecesarios de la línea."""
    return linea.replace('\u202f', ' ').replace('\u200e', '').replace('\ufeff', '').strip()

def obtener_partes(linea):
    """Extrae fecha, hora, miembro y mensaje de una línea."""
    if linea.startswith('['):
        linea = linea[1:]
    
    partes = linea.split('] ', maxsplit=1)
    if len(partes) < 2:
        return None, None, None, None
    
    fecha_hora, mensaje = partes
    fecha, hora = fecha_hora.split(', ', maxsplit=1)
    miembro = None
    
    if ': ' in mensaje:
        miembro, mensaje = mensaje.split(': ', maxsplit=1)
    
    return fecha, hora, miembro, mensaje

def contar_tipos_de_contenido(archivo):
    """Cuenta los distintos tipos de contenido del chat."""
    conteo = {"Mensajes": 0, "Multimedia": 0, "Stickers": 0, "Emojis": 0, "Links": 0}
    
    with open(archivo, 'r', encoding='utf-8') as file:
        lineas = file.readlines()

    for linea in lineas:
        linea = limpiar_linea(linea)
        
        if inicia_con_fecha_y_hora(linea):
            conteo["Mensajes"] += 1

            if any(omitido in linea for omitido in ['image omitted', 'audio omitted', 'video omitted']):
                conteo["Multimedia"] += 1

            if 'sticker omitted' in linea:
                conteo["Stickers"] += 1

            if demoji.findall(linea):
                conteo["Emojis"] += len(demoji.findall(linea))

            if re.search(r'http[s]?://[^\s]+', linea):
                conteo["Links"] += 1

    return conteo

def calcular_promedio_palabras_por_miembro(archivo):
    """Calcula el promedio de palabras por mensaje por miembro."""
    datos_por_miembro = defaultdict(lambda: {"total_palabras": 0, "total_mensajes": 0, "emojis": 0, "links": 0})

    with open(archivo, 'r', encoding='utf-8') as file:
        lineas = file.readlines()

    for linea in lineas:
        if inicia_con_fecha_y_hora(linea):
            _, _, miembro, mensaje = obtener_partes(linea)
            if miembro and mensaje:
                palabras = len(mensaje.split())
                datos_por_miembro[miembro]["total_palabras"] += palabras
                datos_por_miembro[miembro]["total_mensajes"] += 1
                datos_por_miembro[miembro]["emojis"] += len(demoji.findall(mensaje))
                if re.search(r'http[s]?://[^\s]+', mensaje):
                    datos_por_miembro[miembro]["links"] += 1

    resultados = []
    for miembro, datos in datos_por_miembro.items():
        total_palabras = datos["total_palabras"]
        total_mensajes = datos["total_mensajes"]
        emojis = datos["emojis"]
        links = datos["links"]

        promedio_palabras = total_palabras / total_mensajes if total_mensajes > 0 else 0

        resultados.append({
            "Miembro": miembro,
            "Mensajes": total_mensajes,
            "Promedio Palabras/Mensaje": promedio_palabras,
            "Emojis": emojis,
            "Links": links
        })

    return pd.DataFrame(resultados)

def contar_mensajes_por_hora(archivo):
    """Cuenta los mensajes por hora en un día promedio."""
    mensajes_por_hora = defaultdict(int)

    with open(archivo, 'r', encoding='utf-8') as file:
        lineas = file.readlines()

    for linea in lineas:
        if inicia_con_fecha_y_hora(linea):
            fecha_hora = linea.split('] ')[0].strip('[')
            hora_str = fecha_hora.split(', ')[1]
            
            try:
                # Aseguramos de que el AM/PM esté bien formateado
                hora_24 = datetime.strptime(hora_str.strip(), '%I:%M:%S %p').hour
                mensajes_por_hora[hora_24] += 1  # Incrementa el contador de la hora correspondiente
            except ValueError:
                continue

    # Asegúrate de que todas las horas de 0 a 23 estén presentes en el diccionario, aunque no haya mensajes en alguna de ellas
    for hora in range(24):
        if hora not in mensajes_por_hora:
            mensajes_por_hora[hora] = 0

    # Convertir el diccionario a DataFrame directamente
    df_mensajes_por_hora = pd.DataFrame(list(mensajes_por_hora.items()), columns=["Hora", "Cantidad"])

    # Convertir las horas a formato adecuado para el gráfico
    df_mensajes_por_hora["Hora"] = [f"{hora:02d}:00 - {hora+1:02d}:00" for hora in df_mensajes_por_hora["Hora"]]
    
    df_mensajes_por_hora.set_index("Hora", inplace=True)

    return df_mensajes_por_hora
def contar_mensajes_por_dia(archivo):
    """Cuenta los mensajes por día de la semana (lunes, martes, etc.)."""
    mensajes_por_dia = defaultdict(int)

    with open(archivo, 'r', encoding='utf-8') as file:
        lineas = file.readlines()

    for linea in lineas:
        if inicia_con_fecha_y_hora(linea):
            fecha_hora = linea.split('] ')[0].strip('[')
            fecha, hora = fecha_hora.split(', ', maxsplit=1)
            
            try:
                # Convertir la fecha al formato adecuado y obtener el día de la semana
                dia_semana = datetime.strptime(fecha, '%m/%d/%y').strftime('%A')  # Día de la semana
                mensajes_por_dia[dia_semana] += 1
            except ValueError:
                continue

    # Asegúrate de que todos los días de la semana estén presentes, aunque no haya mensajes en alguno de ellos
    dias_semana = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for dia in dias_semana:
        if dia not in mensajes_por_dia:
            mensajes_por_dia[dia] = 0

    # Convertir el diccionario a DataFrame
    df_mensajes_por_dia = pd.DataFrame(list(mensajes_por_dia.items()), columns=["Día", "Cantidad"])

    # Reordenar los días de la semana en orden correcto
    df_mensajes_por_dia["Día"] = pd.Categorical(df_mensajes_por_dia["Día"], categories=dias_semana, ordered=True)
    df_mensajes_por_dia = df_mensajes_por_dia.sort_values("Día")

    return df_mensajes_por_dia

def crear_nube_de_palabras(archivo):
    """Crea una nube de palabras a partir del archivo del chat."""
    palabras_comunes = {
        "y", "ya", "de", "la", "que", "el", "en", "a", "no", "es", "con", "por", 
        "un", "una", "los", "las", "al", "del", "me", "te", "se", "lo", "le", "mi", 
        "sí", "tu", "su", "ha", "va", "muy", "más", "como", "pero", "si", "o", "e", 
        "este", "esta", "ese", "eso", "también", "porque","pue","yo","qué","Pues"
    }

    with open(archivo, 'r', encoding='utf-8') as file:
        lineas = file.readlines()

    texto = ''
    for linea in lineas:
        if inicia_con_fecha_y_hora(linea):
            _, _, _, mensaje = obtener_partes(linea)
            if mensaje:
                texto += mensaje + ' '
    mask = np.array(Image.open('resources/corazon.jpg'))
    stopwords = set(STOPWORDS).union(palabras_comunes)
    nube = WordCloud(width = 800, height = 800, background_color ='black', stopwords = stopwords,
                      max_words=100, min_font_size = 5,
                      mask = mask, colormap='OrRd',).generate(texto)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(nube, interpolation='bilinear')
    ax.axis('off')
    return fig
def contar_mensajes_por_mes(archivo):
    """Cuenta los mensajes por mes en el chat."""
    mensajes_por_mes = defaultdict(int)

    with open(archivo, 'r', encoding='utf-8') as file:
        lineas = file.readlines()

    for linea in lineas:
        if inicia_con_fecha_y_hora(linea):
            fecha_hora = linea.split('] ')[0].strip('[')
            fecha = fecha_hora.split(', ')[0]
            # Extraemos el mes y año de la fecha
            mes_anio = datetime.strptime(fecha, '%m/%d/%y').strftime('%Y-%m')
            mensajes_por_mes[mes_anio] += 1  # Incrementamos el contador de mensajes para ese mes

    # Convertir el diccionario a DataFrame
    df_mensajes_por_mes = pd.DataFrame(list(mensajes_por_mes.items()), columns=["Mes-Año", "Cantidad"])
    
    # Ordenamos por Mes-Año para que esté en orden cronológico
    df_mensajes_por_mes["Mes-Año"] = pd.to_datetime(df_mensajes_por_mes["Mes-Año"])
    df_mensajes_por_mes = df_mensajes_por_mes.sort_values("Mes-Año")

    # Convertimos de nuevo la fecha a formato string para mostrarla mejor
    df_mensajes_por_mes["Mes-Año"] = df_mensajes_por_mes["Mes-Año"].dt.strftime('%Y-%m')

    df_mensajes_por_mes.set_index("Mes-Año", inplace=True)
    
    return df_mensajes_por_mes
def contar_palabras_por_miembro(archivo):
    """Cuenta las palabras más comunes de cada miembro y la cantidad de veces que se usan, excluyendo las stopwords."""
    palabras_comunes = {
        "y", "ya", "de", "la", "que", "el", "en", "a", "no", "es", "con", "por", 
        "un", "una", "los", "las", "al", "del", "me", "te", "se", "lo", "le", "mi", 
        "sí", "tu", "su", "ha", "va", "muy", "más", "como", "pero", "si", "o", "e", 
        "este", "esta", "ese", "eso", "también", "porque","pue","yo","qué","Pues",
        "ps", "para", "pues"
    }

    palabras_por_miembro = defaultdict(list)

    with open(archivo, 'r', encoding='utf-8') as file:
        lineas = file.readlines()

    for linea in lineas:
        if inicia_con_fecha_y_hora(linea):
            _, _, miembro, mensaje = obtener_partes(linea)
            if miembro and mensaje:
                # Limpiar y dividir el mensaje en palabras
                palabras = re.findall(r'\b\w+\b', mensaje.lower())  # Usamos \b\w+\b para obtener solo palabras
                palabras_filtradas = [palabra for palabra in palabras if palabra not in palabras_comunes]
                palabras_por_miembro[miembro].extend(palabras_filtradas)

    # Crear un diccionario para almacenar el top de palabras por miembro
    top_palabras_por_miembro = {}

    for miembro, palabras in palabras_por_miembro.items():
        contador_palabras = Counter(palabras)
        top_palabras_por_miembro[miembro] = contador_palabras.most_common(10)  # Obtener las 10 más comunes

    # Convertir el resultado a un formato más adecuado para el dataframe
    data = []
    for miembro, palabras in top_palabras_por_miembro.items():
        for palabra, frecuencia in palabras:
            data.append({"Miembro": miembro, "Palabra": palabra, "Frecuencia": frecuencia})

    return data


def mostrar_top_palabras_por_miembro(top_palabras_por_miembro):
    """Muestra un top de las palabras más comunes de cada miembro y la cantidad de veces que se usaron."""
    top_palabras_df = []

    for miembro, palabras in top_palabras_por_miembro.items():
        for palabra, frecuencia in palabras:
            top_palabras_df.append({
                "Miembro": miembro,
                "Palabra": palabra,
                "Frecuencia": frecuencia
            })

    return pd.DataFrame(top_palabras_df)

# ---------------- PROCESAMIENTO Y VISUALIZACIÓN ----------------
if os.path.exists(RUTA_CHAT):
    st.subheader('Estadísticas de cada uno')
    df_resultados = calcular_promedio_palabras_por_miembro(RUTA_CHAT)
    st.table(df_resultados)
    
    st.subheader('Estadísticas en general')
    conteo = contar_tipos_de_contenido(RUTA_CHAT)
    df_estadisticas = pd.DataFrame({
        "Tipo": ["Mensajes", "Multimedia", "Stickers", "Emojis", "Links"],
        "Cantidad": [conteo["Mensajes"], conteo["Multimedia"], conteo["Stickers"], conteo["Emojis"], conteo["Links"]]
    })
    st.table(df_estadisticas)

    st.subheader('Promedio de mensajes por hora')
    promedio_horas = contar_mensajes_por_hora(RUTA_CHAT)
    
    # No usamos from_dict, directamente pasamos el DataFrame
    st.bar_chart(promedio_horas)

    st.subheader('Nube de Palabras')
    fig = crear_nube_de_palabras(RUTA_CHAT)
    st.pyplot(fig)
    st.subheader('Mensajes por Día de la Semana')
    mensajes_por_dia = contar_mensajes_por_dia(RUTA_CHAT)
    st.bar_chart(mensajes_por_dia.set_index('Día'))
    st.subheader('Top de Palabras por Miembro')
    top_palabras_por_miembro = contar_palabras_por_miembro(RUTA_CHAT)
    
    # Convertir la lista de diccionarios a un DataFrame para mostrarlo
    df_top_palabras = pd.DataFrame(top_palabras_por_miembro)
    st.table(df_top_palabras)
    
    # Gráfico de barras para las palabras más usadas
    miembro = st.selectbox('Selecciona un miembro', df_top_palabras['Miembro'].unique())
    df_miembro = df_top_palabras[df_top_palabras['Miembro'] == miembro]
    st.bar_chart(df_miembro.set_index('Palabra')['Frecuencia'])
if os.path.exists(RUTA_CHAT):
    
    
    st.subheader('Mensajes por Mes')
    df_mensajes_por_mes = contar_mensajes_por_mes(RUTA_CHAT)
    st.table(df_mensajes_por_mes)

    # Puedes hacer también un gráfico de barras:
    st.subheader('Gráfico de Mensajes por Mes')
    st.line_chart(df_mensajes_por_mes['Cantidad'])
else:
    st.write("El archivo de chat no se encuentra en la ruta especificada.")
