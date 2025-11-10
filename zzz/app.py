"""
Water Potability Predictor
Ernestro Garcia Valenzuela
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import seaborn as sns  # Comentado temporalmente
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / 'models'
PIPELINE_PATH = MODELS_DIR / 'pipeline_full.pkl'
MODEL_PATH = MODELS_DIR / 'best_model.pkl'

FEATURE_NAMES = [
    'ph','Hardness','Solids','Chloramines','Sulfate',
    'Conductivity','Organic_carbon','Trihalomethanes','Turbidity'
]

FEATURE_RANGES = {
    'ph': (0, 14),
    'Hardness': (47, 323),
    'Solids': (320, 61227), 
    'Chloramines': (0.35, 13.13),
    'Sulfate': (129, 481),
    'Conductivity': (181, 753),
    'Organic_carbon': (2.2, 28.3),
    'Trihalomethanes': (0.74, 124),
    'Turbidity': (1.45, 6.74)
}


def generate_random_sample():
    """Generar valores aleatorios para pruebas"""
    sample = {}
    for feature in FEATURE_NAMES:
        min_val, max_val = FEATURE_RANGES[feature]
        mean = (min_val + max_val) / 2
        std = (max_val - min_val) / 6
        value = np.random.normal(mean, std)
        value = np.clip(value, min_val, max_val)
        sample[feature] = round(value, 3)
    return sample


def create_feature_distribution_plot():
    """Crear gráfico de distribución de características"""
    try:
        csv_path = BASE_DIR / 'archive' / 'train_dataset.csv'
        if not csv_path.exists():
            return None
            
        df = pd.read_csv(csv_path)
        
        plt.style.use('default')
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(FEATURE_NAMES):
            if feature in df.columns:
                axes[i].hist(df[feature].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_title(f'Distribución de {feature}', fontsize=10)
                axes[i].set_xlabel(feature, fontsize=9)
                axes[i].set_ylabel('Frecuencia', fontsize=9)
                axes[i].tick_params(labelsize=8)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='white', edgecolor='black', dpi=100)
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode()
    except Exception as e:
        print(f'Error creando gráfico de distribución: {e}')
        return None


def create_prediction_confidence_plot(prediction_proba):
    """Crear gráfico de confianza de predicción"""
    try:
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(8, 5))
        
        categories = ['No Potable', 'Potable']
        
        # Manejar diferentes tipos de entrada de probabilidades
        if prediction_proba is not None:
            if hasattr(prediction_proba, '__len__') and len(prediction_proba) >= 2:
                # Es un array de probabilidades [prob_no_potable, prob_potable]
                probabilities = list(prediction_proba)
            elif isinstance(prediction_proba, (int, float)):
                # Es solo la probabilidad de potable
                prob_potable = float(prediction_proba)
                probabilities = [1 - prob_potable, prob_potable]
            else:
                probabilities = [0.5, 0.5]
        else:
            probabilities = [0.5, 0.5]
        
        colors = ['#ff6b6b', '#4ecdc4']
        bars = ax.bar(categories, probabilities, color=colors, alpha=0.8, edgecolor='black')
        
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_ylim(0, 1)
        ax.set_ylabel('Probabilidad', fontsize=12)
        ax.set_title('Confianza de Predicción', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', facecolor='white', edgecolor='black', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        buffer.close()
        plt.close()
        
        return base64.b64encode(plot_data).decode()
    except Exception as e:
        print(f'Error creando gráfico de confianza: {e}')
        return None


def load_pipeline_and_model():
    """Intentar cargar un pipeline todo-en-uno, o preprocesador + modelo por separado.
    Devuelve (predictor, is_pipeline_flag) donde predictor es un objeto con predict/predict_proba
    """
    # Preferir pipeline_full si existe
    if PIPELINE_PATH.exists():
        try:
            pipeline = joblib.load(PIPELINE_PATH)
            return pipeline, True
        except Exception as e:
            print('Error cargando pipeline_full.pkl:', e)

    # Fallback: intentar cargar modelo y (opcional) preprocesador
    model = None
    preproc = None
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            print('Error cargando best_model.pkl:', e)

    # Si el modelo no es None y el pipeline no fue cargado, intentar cargar un preprocesador
    # con nombre convencionado (pipeline_full contiene a menudo el preprocesador, pero por si acaso)
    if model is not None:
        return model, False

    raise FileNotFoundError('No se encontró un pipeline o modelo válido en models/. Coloca pipeline_full.pkl o best_model.pkl')
    """Intentar cargar un pipeline todo-en-uno, o preprocesador + modelo por separado.
    Devuelve (predictor, is_pipeline_flag) donde predictor es un objeto con predict/predict_proba
    """
    # Preferir pipeline_full si existe
    if PIPELINE_PATH.exists():
        try:
            pipeline = joblib.load(PIPELINE_PATH)
            return pipeline, True
        except Exception as e:
            print('Error cargando pipeline_full.pkl:', e)

    # Fallback: intentar cargar modelo y (opcional) preprocesador
    model = None
    preproc = None
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            print('Error cargando best_model.pkl:', e)

    # Si el modelo no es None y el pipeline no fue cargado, intentar cargar un preprocesador
    # con nombre convencionado (pipeline_full contiene a menudo el preprocesador, pero por si acaso)
    if model is not None:
        return model, False

    raise FileNotFoundError('No se encontró un pipeline o modelo válido en models/. Coloca pipeline_full.pkl o best_model.pkl')


# Cargar predictor al iniciarse la app
try:
    PREDICTOR, PRED_IS_PIPELINE = load_pipeline_and_model()
    print('Predictor cargado. Pipeline integrado?:', PRED_IS_PIPELINE)
except Exception as e:
    PREDICTOR = None
    PRED_IS_PIPELINE = False
    print('Advertencia: predictor no cargado:', e)


@app.route('/api/random-sample')
def get_random_sample():
    """API endpoint para obtener una muestra aleatoria"""
    sample = generate_random_sample()
    return jsonify(sample)


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    details = None
    confidence_plot = None
    # valores por defecto vacíos
    values = {f: '' for f in FEATURE_NAMES}

    if request.method == 'POST':
        # Leer valores enviados
        try:
            for f in FEATURE_NAMES:
                v = request.form.get(f, '')
                values[f] = v

            # Convertir a floats (si falla, se propagará)
            row = {f: float(values[f]) for f in FEATURE_NAMES}
            X = pd.DataFrame([row])

            if PREDICTOR is None:
                raise RuntimeError('Predictor no cargado. Revisa la carpeta models/ y los pickles.')

            # Si el objeto cargado es un pipeline (prefiere transform+predict internamente)
            if PRED_IS_PIPELINE:
                pred = PREDICTOR.predict(X)
                proba = None
                if hasattr(PREDICTOR, 'predict_proba'):
                    proba_array = PREDICTOR.predict_proba(X)
                    proba = proba_array[0]  # Guardar array completo
            else:
                # predictor es solo el modelo; asumimos que acepta datos ya transformados
                # intentar aplicar predict directly (si pipeline no está disponible, esperamos que el modelo acepte raw X)
                pred = PREDICTOR.predict(X)
                proba = None
                if hasattr(PREDICTOR, 'predict_proba'):
                    proba_array = PREDICTOR.predict_proba(X)
                    proba = proba_array[0]  # Guardar array completo

            pred_label = int(pred[0]) if hasattr(pred, '__len__') else int(pred)
            
            # Crear gráfico de confianza
            confidence_plot = create_prediction_confidence_plot(proba)
            
            if proba is not None:
                potable_prob = proba[1]  # Probabilidad de potable
                details = f'Probabilidad de potabilidad ≈ {potable_prob:.3f} ({potable_prob:.1%})'
            else:
                details = 'Probabilidad no disponible para este modelo.'

            result = 'Potable' if pred_label == 1 else 'No potable'

        except ValueError as ve:
            result = None
            details = f'Error de entrada: asegúrate de ingresar valores numéricos válidos. ({ve})'
        except Exception as e:
            result = None
            details = f'Error al predecir: {e}'

    distribution_plot = create_feature_distribution_plot()
    
    return render_template('index.html', 
                         features=FEATURE_NAMES, 
                         values=values, 
                         result=result, 
                         details=details,
                         confidence_plot=confidence_plot,
                         distribution_plot=distribution_plot,
                         feature_ranges=FEATURE_RANGES)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=False)
