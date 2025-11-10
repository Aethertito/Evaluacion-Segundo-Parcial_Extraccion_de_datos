# Water Potability Predictor

**Autor:** Ernestro Garcia Valenzuela  
**Curso:** Machine Learning - Parte 2

## Descripción

Este proyecto predice si una muestra de agua es potable o no usando machine learning. Analiza 9 características químicas del agua y usa un modelo Random Forest para hacer la predicción.

La idea es que puedas ingresar los valores de laboratorio de una muestra de agua y el sistema te diga si es segura para beber.

## Cómo usar

```bash
pip install -r requirements.txt
python app.py
```

Abre http://localhost:8501 en tu navegador.

## Funcionamiento

1. Ingresa los 9 valores de las características del agua (o usa el botón de datos aleatorios para probar)
2. Presiona "Predecir" 
3. El sistema te muestra si el agua es potable o no, con gráficos de confianza

## Características que evalúa

- **pH**: Nivel de acidez (0-14)
- **Hardness**: Qué tan "dura" es el agua 
- **Solids**: Sólidos disueltos totales
- **Chloramines**: Nivel de cloraminas
- **Sulfate**: Contenido de sulfatos
- **Conductivity**: Conductividad eléctrica
- **Organic_carbon**: Carbono orgánico presente
- **Trihalomethanes**: Compuestos químicos específicos
- **Turbidity**: Qué tan turbia está el agua

## Archivos importantes

- `app.py` - La aplicación web principal
- `templates/index.html` - La página web
- `static/style.css` - Los estilos
- `models/` - Los modelos entrenados
- `archive/` - Los datos originales que usé para entrenar

## Tecnología

Uso Flask para la web, scikit-learn para el machine learning, y matplotlib para los gráficos. El modelo es Random Forest porque me dio buenos resultados en las pruebas.

## Notas

El dataset original tenía como 3000 muestras de agua con sus características y si eran potables o no. Entrené el modelo con esos datos y ahora puede predecir nuevas muestras.

Los gráficos muestran la distribución de datos y qué tan seguro está el modelo de su predicción.