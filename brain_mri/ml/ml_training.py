import os  # Configuração de ambiente/threads para libs nativas
import pickle  # Serialização de objetos Python
import time  # Funções relacionadas a tempo e pausas
import tkinter as tk  # Interface gráfica
from tkinter import messagebox  # Diálogos do Tkinter

import numpy as np  # Operações numéricas e arrays

# Limita threads do OpenMP/MKL para evitar conflitos de múltiplos libomp em macOS/ARM
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    import pandas as pd  # Manipulação de dados tabulares
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
try:  # Dependências pesadas opcionais
    import torch  # Framework de deep learning PyTorch
    import torch.nn as nn  # Submódulo de camadas neurais do PyTorch
    import torch.nn.functional as F  # Funções auxiliares para pooling/ReLU
    import torch.optim as optim  # Otimizadores do PyTorch
    from torch.optim.lr_scheduler import CosineAnnealingLR  # Scheduler para annealing do LR
    from torch.utils.data import DataLoader  # Carregador de dados PyTorch
    from .training_utils import (
        ExponentialMovingAverage,
        build_densenet,
        build_transforms,
        focal_loss,
        mixup_data,
        select_device,
    )
    TORCH_AVAILABLE = True
except ImportError:
    torch = nn = optim = DataLoader = None
    ExponentialMovingAverage = build_densenet = build_transforms = focal_loss = mixup_data = select_device = None
    TORCH_AVAILABLE = False
try:
    import xgboost as xgb  # Biblioteca de gradient boosting XGBoost
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
try:
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                                 precision_score, r2_score, recall_score)  # Métricas de avaliação
    from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split  # Divisão de dados e validação cruzada
    from sklearn.preprocessing import StandardScaler  # Normalização de características
    from sklearn.svm import SVC  # Suporte a máquinas de vetor para classificação
    SKLEARN_AVAILABLE = True
except ImportError:
    accuracy_score = confusion_matrix = f1_score = mean_absolute_error = mean_squared_error = precision_score = r2_score = recall_score = None
    GridSearchCV = GroupKFold = train_test_split = StandardScaler = SVC = None
    SKLEARN_AVAILABLE = False
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # Canvas e barra de ferramentas do Matplotlib integrados ao Tkinter
from matplotlib.figure import Figure  # Objeto de figura do Matplotlib

from .datasets import MRIDataset  # Dataset PyTorch específico


class MLTrainingMixin:  # Métodos de criação de dataset e treinamento de modelos
    def _list_orientation_paths(self, mri_id: str):
        """Lista caminhos disponíveis para o mesmo exame nas orientações axl/cor/sag."""
        base = self.dataset_dir.parent
        paths = []
        for orient in ["axl", "cor", "sag"]:
            for ext in (".nii.gz", ".nii"):
                cand = base / orient / f"{mri_id}_{orient}{ext}"
                if cand.exists():
                    try:
                        rel = cand.relative_to(base)
                        paths.append(str(rel))
                    except ValueError:
                        paths.append(str(cand))
        return paths

    def _expand_with_orientations(self, df_subset):
        """Duplica linhas para cada orientação disponível; mantém metadados originais."""
        if df_subset is None or df_subset.empty:
            return df_subset
        rows = []
        for _, row in df_subset.iterrows():
            mri_id = row.get("MRI_ID")
            if not isinstance(mri_id, str):
                continue
            orient_paths = self._list_orientation_paths(mri_id)
            # Se não houver nenhuma outra orientação, mantém a linha original
            if not orient_paths:
                if row.get("original_path"):
                    rows.append(row.copy())
                continue
            # Uma linha por exame, guardando lista de caminhos para empilhamento downstream
            r = row.copy()
            r["orientation_paths"] = orient_paths
            # Define original_path como axial se existir, senão primeiro disponível
            axial = [p for p in orient_paths if "_axl" in p]
            r["original_path"] = axial[0] if axial else orient_paths[0]
            rows.append(r)
        return type(df_subset)(rows)
    def create_exam_level_dataset(self):  # Constrói dataset unindo descritores e dados demográficos com split
        """Junta CSV demográfico com descritores e faz split."""  # Docstring do processo de criação do dataset
        if not PANDAS_AVAILABLE:
            messagebox.showerror("Dependência ausente", "O módulo 'pandas' é necessário para criar o dataset.\nInstale com 'pip install pandas'.")
            return
        if not self.descriptors_csv.exists():  # Garante que descritores foram gerados
            messagebox.showwarning("Aviso", "Gere descritores antes de criar o dataset.")  # Alerta se CSV não existe
            return  # Interrompe execução

        df_desc = pd.read_csv(self.descriptors_csv)  # Carrega descritores já calculados
        if df_desc.empty:  # Verifica se há registros
            messagebox.showwarning("Aviso", "CSV de descritores vazio.")  # Alerta se vazio
            return  # Sai
        if 'viable' not in df_desc.columns:
            df_desc['viable'] = True  # Garante flag de viabilidade

        df_demo = pd.read_csv(self.csv_path, sep=';', decimal=',')  # Lê CSV demográfico com separador ';'
        df_demo.columns = [c.strip() for c in df_demo.columns]  # Remove espaços dos nomes de coluna
        if 'MRI ID' in df_demo.columns:
            df_demo.rename(columns={'MRI ID': 'MRI_ID'}, inplace=True)  # Normaliza nome da coluna de MRI
        if 'Subject ID' in df_demo.columns:
            df_demo.rename(columns={'Subject ID': 'Subject_ID'}, inplace=True)  # Normaliza nome da coluna de sujeito

        def _as_numeric(series):  # Converte série para numérico tratando vírgula decimal
            return pd.to_numeric(series.astype(str).str.replace(',', '.').str.strip(), errors='coerce')  # Retorna números ou NaN

        numeric_map = {  # Mapeia colunas originais para nomes padronizados
            'Age': 'age',
            'EDUC': 'education',
            'MMSE': 'mmse',
            'CDR': 'cdr',
            'eTIV': 'etiv',
            'nWBV': 'nwbv',
            'ASF': 'asf'
        }
        for src, dst in numeric_map.items():  # Itera pares de mapeamento
            if src in df_demo.columns:  # Se coluna existe
                df_demo[dst] = _as_numeric(df_demo[src])  # Cria coluna numérica convertida

        if 'M/F' in df_demo.columns:
            df_demo['sex'] = df_demo['M/F'].map({'M': 0, 'F': 1})  # Codifica sexo como binário

        merged = pd.merge(df_desc, df_demo, on='MRI_ID', how='left', suffixes=('', '_demo'))  # Faz merge descritores+demografia
        merged['viable'] = merged['viable'].fillna(True)  # Preenche viabilidade ausente como verdadeira
        merged = merged[merged['viable'] == True]  # Filtra apenas exames viáveis

        if 'Subject_ID_x' in merged.columns:
            merged['Subject_ID'] = merged['Subject_ID_x']  # Prefere coluna de sujeito x
        if 'Subject_ID_y' in merged.columns:
            merged['Subject_ID'] = merged['Subject_ID'].fillna(merged['Subject_ID_y'])  # Preenche faltantes com coluna y
            merged.drop(columns=['Subject_ID_y'], inplace=True)  # Remove coluna duplicada
        if 'Subject_ID_x' in merged.columns:
            merged.drop(columns=['Subject_ID_x'], inplace=True)  # Remove coluna duplicada

        merged['Original_Group'] = merged.get('Group')  # Guarda grupo original

        def _resolve_final_group(row):  # Resolve grupo final para casos "Converted"
            grp = row.get('Group')  # Grupo original
            if isinstance(grp, str) and grp == 'Converted':  # Se convertido
                cdr_val = row.get('cdr') if 'cdr' in row else row.get('CDR')  # Busca CDR numérico
                if pd.notna(cdr_val) and float(cdr_val) > 0:  # Se CDR > 0 considera Demente
                    return 'Demented'
                return 'Nondemented'  # Caso contrário, Nondemented
            return grp  # Retorna grupo original

        merged['Final_Group'] = merged.apply(_resolve_final_group, axis=1)  # Aplica regra de conversão de grupo
        merged['Final_Group'] = merged['Final_Group'].fillna(merged['Original_Group'])  # Preenche faltantes com original

        merged['original_path'] = merged['MRI_ID'].apply(self._resolve_original_path)  # Resolve caminho do exame
        merged = merged[merged['original_path'] != ""]  # Remove registros sem caminho válido
        merged = merged[merged['Subject_ID'].notna()]  # Filtra registros com Subject_ID

        subjects = merged['Subject_ID'].unique()  # Lista de sujeitos únicos
        if len(subjects) < 3:  # Requer mínimo de 3 sujeitos para split
            messagebox.showwarning("Aviso", "Dados insuficientes para split (mínimo 3 sujeitos).")  # Alerta insuficiência
            return  # Interrompe

        train_sub, test_sub = train_test_split(subjects, test_size=0.2)  # Separa sujeitos em treino/teste
        train_sub, val_sub = train_test_split(train_sub, test_size=0.2)  # Separa subset de validação

        def get_split(sid):  # Função auxiliar para mapear sujeito para split
            if sid in val_sub: return 'validation'  # Sujeitos de validação
            if sid in test_sub: return 'test'  # Sujeitos de teste
            return 'train'  # Demais em treino

        merged['split'] = merged['Subject_ID'].apply(get_split)  # Aplica divisão por sujeito

        cols_to_drop = ['Age', 'EDUC', 'SES', 'MMSE', 'CDR', 'eTIV', 'nWBV', 'ASF', 'Visit', 'MR Delay', 'M/F']  # Colunas redundantes
        cols_to_drop = [col for col in cols_to_drop if col in merged.columns]  # Filtra apenas existentes
        if cols_to_drop:
            merged.drop(columns=cols_to_drop, inplace=True)  # Remove colunas duplicadas ou não padronizadas

        out = self.output_dir / "exam_level_dataset_split.csv"  # Caminho de saída do dataset combinado
        merged.to_csv(out, index=False)  # Salva CSV final
        messagebox.showinfo("Sucesso", f"Dataset criado em {out.name}\nTotal: {len(merged)} exames.")  # Informa sucesso

    def open_feature_selection_dialog(self):  # Abre seleção de features para SVM
        self._generic_feature_selector("SVM", self.train_svm_classifier)  # Reaproveita diálogo genérico apontando para SVM

    def open_feature_selection_dialog_xgboost(self):  # Abre seleção de features para XGBoost
        self._generic_feature_selector("XGBoost", self.train_xgboost_regressor)  # Chama diálogo genérico apontando XGBoost

    def _generic_feature_selector(self, title, callback):  # Diálogo genérico para escolher colunas de features
        win = tk.Toplevel(self.root)  # Cria janela de nível superior
        win.title(f"Features para {title}")  # Define título contextual

        vars_dict = {}  # Dicionário para armazenar variáveis das checkboxes
        features = [  # Lista de features disponíveis
            'ventricle_area', 'ventricle_perimeter', 'ventricle_circularity',
            'ventricle_eccentricity', 'ventricle_solidity', 'ventricle_major_axis_length',
            'ventricle_minor_axis_length', 'age', 'sex', 'education', 'mmse', 'cdr',
            'nwbv', 'etiv', 'asf'
        ]

        for f in features:  # Cria checkbox para cada feature
            v = tk.BooleanVar(value=True)  # Variável booleana iniciando marcada
            tk.Checkbutton(win, text=f, variable=v).pack(anchor='w')  # Adiciona checkbox à janela
            vars_dict[f] = v  # Armazena referência da variável

        def run():  # Ação ao confirmar seleção
            selected = [k for k, v in vars_dict.items() if v.get()]  # Filtra features selecionadas
            win.destroy()  # Fecha diálogo
            callback(selected)  # Invoca callback com lista selecionada

        tk.Button(win, text="Treinar", command=run).pack(pady=10)  # Botão para iniciar treino

    def train_svm_classifier(self, features=None):  # Treina classificador SVM com seleção de features
        if not SKLEARN_AVAILABLE:
            messagebox.showerror("Dependência ausente", "O módulo 'scikit-learn' é necessário para treinar o SVM.\nInstale com 'pip install scikit-learn'.")
            return
        if not PANDAS_AVAILABLE:
            messagebox.showerror("Dependência ausente", "O módulo 'pandas' é necessário para treinar o SVM.\nInstale com 'pip install pandas'.")
            return
        start_time = time.time()  # Marca início para medir tempo de treino

        df_path = self.output_dir / "exam_level_dataset_split.csv"  # Caminho do dataset combinado
        if not df_path.exists():  # Garante existência do dataset
            messagebox.showwarning("Aviso", "Crie o dataset primeiro.")  # Alerta se não existir
            return  # Interrompe

        df = pd.read_csv(df_path)  # Lê dataset
        if not features:  # Se features não especificadas
            features = ['ventricle_area', 'ventricle_perimeter', 'ventricle_circularity',
                        'ventricle_eccentricity', 'mmse', 'cdr', 'age']  # Features padrão

        tmp = df.copy()  # Copia para ajustes
        if 'sex' in features and 'sex' not in tmp.columns:  # Se sexo solicitado mas ausente
            if 'M/F' in tmp.columns:
                tmp['sex'] = tmp['M/F'].map({'M': 0, 'F': 1})  # Converte M/F para binário
            else:
                tmp['sex'] = np.nan  # Preenche com NaN caso não exista

        missing = [f for f in features if f not in tmp.columns]  # Checa colunas faltantes
        if missing:  # Se houver faltantes
            messagebox.showerror("Erro", f"Colunas ausentes no dataset: {missing}")  # Erro informativo
            return  # Sai

        X = tmp[features].copy()  # Seleciona features
        X = X.fillna(X.mean())  # Preenche NaN com média
        X = X.values  # Converte para array
        y = (tmp['Final_Group'] == 'Demented').astype(int).values  # Alvo binário: demente ou não

        train_mask = df['split'] == 'train'  # Máscara de treino
        val_mask = df['split'] == 'validation'  # Máscara de validação
        test_mask = df['split'] == 'test'  # Máscara de teste

        if not val_mask.any():  # Exige validação
            messagebox.showwarning("Aviso", "Split de validação vazio.")  # Alerta ausência
            return  # Sai

        scaler = StandardScaler()  # Normalizador
        X_train = scaler.fit_transform(X[train_mask])  # Ajusta e transforma treino
        X_val = scaler.transform(X[val_mask])  # Transforma validação
        X_test = scaler.transform(X[test_mask]) if test_mask.any() else None  # Transforma teste se existir

        grid = {  # Espaço de busca de hiperparâmetros
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        }
        gs = GridSearchCV(SVC(), grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)  # Grid search com CV
        gs.fit(X_train, y[train_mask])  # Ajusta grid no conjunto de treino
        clf = gs.best_estimator_  # Recupera melhor estimador

        y_train_pred = clf.predict(X_train)  # Predições treino
        y_val_pred = clf.predict(X_val)  # Predições validação
        acc_tr = accuracy_score(y[train_mask], y_train_pred)  # Acurácia treino
        acc_val = accuracy_score(y[val_mask], y_val_pred)  # Acurácia validação

        test_cm = None  # Inicializa matriz de confusão de teste
        msg = f"Acurácia (Treino): {acc_tr:.2%}\nAcurácia (Val): {acc_val:.2%}\nMelhor: {gs.best_params_}"  # Mensagem base
        if X_test is not None:  # Se há conjunto de teste
            y_test_pred = clf.predict(X_test)  # Predições de teste
            acc_test = accuracy_score(y[test_mask], y_test_pred)  # Acurácia teste
            test_cm = confusion_matrix(y[test_mask], y_test_pred)  # Matriz de confusão teste

            test_precision = precision_score(y[test_mask], y_test_pred, average='binary', zero_division=0)  # Precisão teste
            test_recall = recall_score(y[test_mask], y_test_pred, average='binary', zero_division=0)  # Recall teste
            test_f1 = f1_score(y[test_mask], y_test_pred, average='binary', zero_division=0)  # F1 teste

            msg += f"\n\n=== TESTE ===\nAcurácia: {acc_test:.2%}\nPrecision: {test_precision:.2%}\nRecall: {test_recall:.2%}\nF1-Score: {test_f1:.2%}"  # Adiciona métricas de teste
        messagebox.showinfo("Resultado Treino (SVM)", msg)  # Exibe resultados

        if test_cm is not None:  # Se matriz de confusão disponível
            try:
                fig_cm = Figure(figsize=(6, 5))  # Figura para matriz de confusão
                ax = fig_cm.add_subplot(1, 1, 1)  # Eixo único
                self.plot_confusion_matrix(ax, test_cm, ['0', '1'], "Teste")  # Desenha matriz
                fig_cm.tight_layout()  # Ajusta layout
                fig_cm.savefig(self.output_dir / "confusion_svm.png", dpi=300, bbox_inches='tight')  # Salva imagem
                self._show_plot_window("Matriz de Confusão SVM - Teste", fig_cm)  # Exibe janela com matriz
            except Exception:
                pass  # Silencia falhas na geração do gráfico

        with open(self.output_dir / "svm_scaler.pkl", "wb") as f:  # Salva scaler
            pickle.dump(scaler, f)  # Serializa scaler
        with open(self.output_dir / "svm_model.pkl", "wb") as f:  # Salva modelo SVM
            pickle.dump(clf, f)  # Serializa modelo

        training_time = time.time() - start_time  # Calcula tempo total de treino

        exp_data = {  # Dados do experimento para histórico
            'model': 'SVM',
            'features': features,
            'best_params': gs.best_params_,
            'train_accuracy': float(acc_tr),
            'val_accuracy': float(acc_val),
            'training_time_seconds': float(training_time),
        }

        if X_test is not None:  # Registra métricas de teste se houver
            exp_data['test_accuracy'] = float(acc_test)
            exp_data['test_precision'] = float(test_precision)
            exp_data['test_recall'] = float(test_recall)
            exp_data['test_f1'] = float(test_f1)
            if test_cm is not None:
                exp_data['test_confusion_matrix'] = test_cm.tolist()  # Salva matriz como lista

        self._save_experiment(exp_data)  # Persiste histórico do experimento

    def train_xgboost_regressor(self, features=None):  # Treina regressor XGBoost para predição de idade
        if not SKLEARN_AVAILABLE:
            messagebox.showerror("Dependência ausente", "O módulo 'scikit-learn' é necessário para treinar o XGBoost.\nInstale com 'pip install scikit-learn'.")
            return
        if not XGBOOST_AVAILABLE:
            messagebox.showerror("Dependência ausente", "O módulo 'xgboost' é necessário para este treino.\nInstale com 'pip install xgboost'.")
            return
        if not PANDAS_AVAILABLE:
            messagebox.showerror("Dependência ausente", "O módulo 'pandas' é necessário para treinar o XGBoost.\nInstale com 'pip install pandas'.")
            return
        start_time = time.time()  # Marca início do treino

        df_path = self.output_dir / "exam_level_dataset_split.csv"  # Caminho do dataset combinado
        if not df_path.exists():  # Verifica existência
            messagebox.showwarning("Aviso", "Crie o dataset primeiro.")  # Alerta ausência
            return  # Sai

        df = pd.read_csv(df_path)  # Lê dataset
        if not features:  # Seleção padrão de features
            features = ['ventricle_area', 'ventricle_perimeter', 'ventricle_circularity',
                        'ventricle_eccentricity', 'mmse', 'cdr', 'nwbv', 'etiv', 'asf', 'sex', 'education']  # Lista padrão

        tmp = df.copy()  # Cópia para manipulação
        if 'sex' in features and 'sex' not in tmp.columns and 'M/F' in tmp.columns:  # Cria sexo binário se necessário
            tmp['sex'] = tmp['M/F'].map({'M': 0, 'F': 1})  # Mapeia sexo

        missing = [f for f in features if f not in tmp.columns]  # Verifica colunas faltantes
        if missing:
            messagebox.showerror("Erro", f"Colunas ausentes no dataset: {missing}")  # Erro se faltar coluna
            return  # Sai

        X = tmp[features].fillna(tmp[features].mean()).values  # Preenche NaN com média e converte para array
        y = tmp['age'].values  # Alvo: idade

        train_mask = df['split'] == 'train'  # Máscara de treino
        val_mask = df['split'] == 'validation'  # Máscara de validação
        if not val_mask.any():  # Exige validação
            messagebox.showwarning("Aviso", "Split de validação vazio.")  # Alerta ausência
            return  # Sai

        groups = df.loc[train_mask, 'Subject_ID']  # Grupos para CV em nível de sujeito

        base = xgb.XGBRegressor(  # Modelo base XGBoost
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=1,
            verbosity=0
        )
        grid = {  # Espaço de busca de hiperparâmetros
            'n_estimators': [200, 300, 500],
            'max_depth': [6, 8, 10],
            'learning_rate': [0.05, 0.1, 0.15],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }

        gkf = GroupKFold(n_splits=3)  # CV estratificado por sujeito
        gs = GridSearchCV(base, grid, cv=gkf.split(X[train_mask], y[train_mask], groups),
                          scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)  # GridSearch com MAE negativo
        gs.fit(X[train_mask], y[train_mask])  # Ajusta grid
        model = gs.best_estimator_  # Melhor modelo encontrado

        val_preds = model.predict(X[val_mask])  # Predições na validação
        mae_val = mean_absolute_error(y[val_mask], val_preds)  # MAE de validação
        r2_val = r2_score(y[val_mask], val_preds)  # R² de validação
        mse_val = mean_squared_error(y[val_mask], val_preds)  # MSE de validação
        rmse_val = float(np.sqrt(mse_val))  # RMSE de validação

        messagebox.showinfo(
            "XGBoost",
            f"Val MAE={mae_val:.2f} | Val RMSE={rmse_val:.2f} | Val MSE={mse_val:.2f} | Val R²={r2_val:.4f}"
        )  # Exibe métricas de validação

        with open(self.output_dir / "xgb_age.pkl", "wb") as f:  # Salva modelo treinado
            pickle.dump(model, f)  # Serializa regressor

        training_time = time.time() - start_time  # Tempo total de treino

        self._save_experiment({  # Registra experimento
            'model': 'XGBoost',  # Nome do modelo
            'target': 'age',  # Variável alvo prevista
            'features': features,  # Lista de features usadas
            'val_mae': float(mae_val),  # MAE na validação
            'val_mse': float(mse_val),  # MSE na validação
            'val_rmse': float(rmse_val),  # RMSE na validação
            'val_r2': float(r2_val),  # R² na validação
            'best_params': gs.best_params_,  # Hiperparâmetros ótimos
            'training_time_seconds': float(training_time),  # Duração do treino em segundos
        })  # Salva experimento no histórico

    def train_densenet_classifier(self):  # Wrapper para treinar DenseNet em modo classificação
        if not TORCH_AVAILABLE:
            messagebox.showerror("Dependência ausente", "PyTorch/torchvision são necessários para treinar a DenseNet.\nInstale com 'pip install torch torchvision'.")
            return
        self._train_pytorch_model(mode='classification')  # Chama treino genérico com modo de classificação

    def train_densenet_regressor(self):  # Wrapper para treinar DenseNet em modo regressão
        if not TORCH_AVAILABLE:
            messagebox.showerror("Dependência ausente", "PyTorch/torchvision são necessários para treinar a DenseNet.\nInstale com 'pip install torch torchvision'.")
            return
        self._train_pytorch_model(mode='regression')  # Chama treino genérico com modo de regressão

    def _train_pytorch_model(self, mode='classification', hparams=None):  # Treina DenseNet para classificação ou regressão
        headless = not hasattr(self, 'root') or self.root is None
        if not SKLEARN_AVAILABLE:
            try:
                messagebox.showerror("Dependência ausente", "O módulo 'scikit-learn' é necessário para normalização e métricas.\nInstale com 'pip install scikit-learn'.")
            except Exception:
                print("[WARN] scikit-learn ausente")
            return
        if not TORCH_AVAILABLE:
            try:
                messagebox.showerror("Dependência ausente", "PyTorch/torchvision são necessários para este treino.\nInstale com 'pip install torch torchvision'.")
            except Exception:
                print("[WARN] PyTorch/torchvision ausentes")
            return
        if not PANDAS_AVAILABLE:
            try:
                messagebox.showerror("Dependência ausente", "O módulo 'pandas' é necessário para preparar os datasets de treino.\nInstale com 'pip install pandas'.")
            except Exception:
                print("[WARN] pandas ausente")
            return
        start_time = time.time()  # Marca início do treinamento

        df_path = self.output_dir / "exam_level_dataset_split.csv"  # Caminho do dataset combinado
        if not df_path.exists(): return  # Sai se o dataset ainda não foi criado

        df = pd.read_csv(df_path)  # Carrega dataset consolidado
        device = select_device()
        print(f"Dispositivo selecionado: {device} | Torch threads: {torch.get_num_threads()}")  # Log para debugging

        # Hiperparâmetros (podem ser sobrescritos por hparams)
        defaults = {
            "lr": 1e-4 if mode == 'classification' else 0.001,
            "weight_decay": 1e-4 if mode == 'classification' else 0.0,
            "dropout": 0.3,
            "label_smoothing": 0.05 if mode == 'classification' else 0.0,
            "mixup_alpha": 0.2 if mode == 'classification' else 0.0,
            "freeze_backbone": False,
            "class_balance": False,
            "freeze_warmup_epochs": 0,
            "warmup_lr": None,
            "balance_penalty": 0.0,
            "thresholds_eval": [0.5, 0.6, 0.4, 0.7],
            "seed": 42,
        }
        if hparams:
            for k, v in hparams.items():
                if k in defaults:
                    defaults[k] = v

        lr = defaults["lr"]
        weight_decay = defaults["weight_decay"]
        dropout_rate = defaults["dropout"]
        label_smoothing = defaults["label_smoothing"]
        mixup_alpha = defaults["mixup_alpha"]
        freeze_backbone = bool(defaults["freeze_backbone"])
        use_class_balance = bool(defaults["class_balance"])

        # Hiperparâmetros de regularização/augmentação
        use_mixup = mode == 'classification'
        mixup_alpha = 0.4

        age_scaler = None  # Normalizador para alvo de regressão
        if mode == 'regression':  # Fluxo específico para regressão de idade
            age_scaler = StandardScaler()  # Inicializa scaler para idade
            df_train = df[df['split']=='train'].copy()  # Subconjunto de treino
            df_val = df[df['split']=='validation'].copy()  # Subconjunto de validação
            df_test = df[df['split']=='test'].copy()  # Subconjunto de teste

            df_train['age_normalized'] = age_scaler.fit_transform(df_train[['age']])  # Normaliza idade no treino
            df_val['age_normalized'] = age_scaler.transform(df_val[['age']])  # Normaliza idade na validação
            df_test['age_normalized'] = age_scaler.transform(df_test[['age']])  # Normaliza idade no teste

            df.loc[df['split']=='train', 'age_normalized'] = df_train['age_normalized']  # Atribui idade normalizada ao DF original (treino)
            df.loc[df['split']=='validation', 'age_normalized'] = df_val['age_normalized']  # Atribui idade normalizada (validação)
            df.loc[df['split']=='test', 'age_normalized'] = df_test['age_normalized']  # Atribui idade normalizada (teste)

            print(f"Age normalization - Original range: [{df['age'].min():.1f}, {df['age'].max():.1f}]")  # Log de faixa original de idades
            print(f"                    Normalized range: [{df_train['age_normalized'].min():.2f}, {df_train['age_normalized'].max():.2f}]")  # Log de faixa normalizada

        train_tf, val_tf = build_transforms()  # Transforms separados em módulo utilitário

        lbl_col = 'age_normalized' if mode == 'regression' else 'Final_Group'  # Define coluna alvo conforme modo

        # Expande cada linha para todas as orientações disponíveis (axl/cor/sag) para DenseNet
        train_df = self._expand_with_orientations(df[df['split']=='train'])
        val_df = self._expand_with_orientations(df[df['split']=='validation'])
        test_df = self._expand_with_orientations(df[df['split']=='test'])

        train_ds = MRIDataset(train_df, train_tf, self.dataset_dir.parent, 'original_path', lbl_col)  # Dataset de treino
        val_ds = MRIDataset(val_df, val_tf, self.dataset_dir.parent, 'original_path', lbl_col)  # Dataset de validação
        test_ds = MRIDataset(test_df, val_tf, self.dataset_dir.parent, 'original_path', lbl_col)  # Dataset de teste

        if len(val_ds) == 0:  # Validação obrigatória
            messagebox.showwarning("Aviso", "Split de validação vazio.")  # Alerta ausência de validação
            return  # Sai

        epochs = 40 if mode == 'classification' else 20  # Número de épocas (mais longo para class)
        if hparams and 'max_epochs' in hparams:
            try:
                epochs = int(hparams['max_epochs'])
            except Exception:
                pass
        batch_size = 16  # Tamanho do batch
        early_stop_patience = 7 if mode == 'classification' else None  # Early stopping para class
        use_mixup = mode == 'classification' and mixup_alpha > 0
        use_focal = mode == 'classification' and not use_mixup  # Focal loss para lidar com desbalanceamento (desligado se mixup ativo)
        focal_gamma = 1.5  # Gamma padrão
        use_ema = mode == 'classification'  # EMA para suavizar pesos na classificação
        ema_decay = 0.999

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)  # Loader de treino com embaralhamento
        val_loader = DataLoader(val_ds, batch_size=batch_size)  # Loader de validação
        test_loader = DataLoader(test_ds, batch_size=batch_size)  # Loader de teste

        model = build_densenet(mode=mode, dropout_rate=dropout_rate).to(device)  # Move modelo para CPU/GPU
        if freeze_backbone and hasattr(model, "features"):
            for p in model.features.parameters():
                p.requires_grad = False

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # Otimizador Adam
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)  # Annealing de LR
        class_weights = None
        if mode == 'classification' and use_class_balance and 'Final_Group' in df.columns:
            counts = df[df['split'] == 'train']['Final_Group'].value_counts()
            if len(counts) >= 1:
                total = counts.sum()
                w0 = total / (2 * counts.get('Nondemented', max(counts.max(), 1)))
                w1 = total / (2 * counts.get('Demented', max(counts.max(), 1)))
                class_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

        label_smoothing = label_smoothing if mode == 'classification' else 0.0
        criterion = nn.MSELoss() if mode == 'regression' else nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=class_weights
        )  # Função de perda conforme modo
        ema = ExponentialMovingAverage(model, decay=ema_decay) if use_ema else None  # EMA opcional

        popup = None
        if not headless and hasattr(self, 'root'):
            popup = tk.Toplevel(self.root)
            lbl = tk.Label(popup, text=f"Treinando... aguarde ({epochs} épocas)"); lbl.pack(padx=20, pady=20)
            try: self.root.update()
            except Exception: pass

        def _mixup_data(x, y, alpha=0.4):
            if alpha <= 0:
                return x, y, y, 1.0
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(x.size(0), device=x.device)
            mixed_x = lam * x + (1 - lam) * x[index, :]
            y_a, y_b = y, y[index]
            return mixed_x, y_a, y_b, lam

        history = []  # Histórico de losses
        history_train_loss, history_val_loss = [], []  # Loss de treino/validação
        history_train_acc, history_val_acc = [], []  # Acurácia de treino/validação (classificação)
        history_train_mae, history_val_mae = [], []  # MAE de treino/validação (regressão)
        val_metric_value = None  # Métrica principal em validação
        best_state, best_epoch = None, 0  # Controle early stopping
        best_val_metric = -float('inf') if mode == 'classification' else float('inf')
        no_improve = 0  # Contador para early stopping

        for epoch in range(epochs):  # Loop de épocas
            model.train()  # Coloca modelo em modo de treino
            running_loss = 0  # Acumulador de loss
            total_train = 0  # Contador de amostras de treino
            correct_train, total_train_cls = 0, 0  # Acertos/total para classificação
            mae_sum_train, total_train_reg = 0.0, 0  # Soma MAE/total para regressão

            for imgs, lbls in train_loader:  # Itera batches de treino
                imgs, lbls = imgs.to(device), lbls.to(device)  # Move dados para CPU/GPU
                optimizer.zero_grad()  # Zera gradientes
                if mode == 'regression':  # Lógica para regressão
                    out = model(imgs)
                    preds_batch = out.squeeze()  # Ajusta forma da saída
                    loss = criterion(preds_batch, lbls)  # Calcula loss MSE
                    mae_sum_train += torch.abs(preds_batch - lbls).sum().item()  # Acumula MAE do batch
                    total_train_reg += lbls.size(0)  # Conta amostras de regressão
                else:  # Lógica para classificação
                    if use_mixup:
                        imgs_mix, targets_a, targets_b, lam = _mixup_data(imgs, lbls.long(), mixup_alpha)
                        out = model(imgs_mix)
                        loss = lam * criterion(out, targets_a) + (1 - lam) * criterion(out, targets_b)
                        preds_batch = out.argmax(dim=1)
                        correct_train += (
                            lam * (preds_batch == targets_a).sum().item()
                            + (1 - lam) * (preds_batch == targets_b).sum().item()
                        )
                    else:
                        out = model(imgs)
                        if use_focal:
                            loss = focal_loss(out, lbls.long(), gamma=focal_gamma)  # Focal loss ajuda com desbalanceamento
                        else:
                            loss = criterion(out, lbls.long())  # Calcula loss CE
                        preds_batch = out.argmax(dim=1)  # Classe predita
                        correct_train += (preds_batch == lbls.long()).sum().item()  # Acertos no batch
                    total_train_cls += lbls.size(0)  # Conta amostras de classe

                loss.backward()  # Backprop
                optimizer.step()  # Atualiza pesos
                if ema: ema.update(model)  # Atualiza EMA
                running_loss += loss.item() * imgs.size(0)  # Soma loss ponderada pelo batch
                total_train += imgs.size(0)  # Atualiza total de amostras

            model.eval()  # Modo avaliação
            running_val = 0  # Acumulador de loss de validação
            preds_list, targs_list = [], []  # Listas para métricas de regressão
            correct_val, total_val = 0, 0  # Acertos/total na validação (classificação)
            if ema: ema.apply_shadow(model)  # Avalia com pesos suavizados
            with torch.no_grad():  # Sem gradientes
                for imgs, lbls in val_loader:  # Itera batches de validação
                    imgs, lbls = imgs.to(device), lbls.to(device)  # Move dados
                    out = model(imgs)  # Forward validação
                    if mode == 'regression':  # Métrica regressão
                        loss = criterion(out.squeeze(), lbls)  # Loss MSE validação
                        running_val += loss.item() * imgs.size(0)  # Soma loss ponderada
                        preds_list.append(out.squeeze().cpu().numpy())  # Guarda predições
                        targs_list.append(lbls.cpu().numpy())  # Guarda rótulos verdadeiros
                    else:  # Métrica classificação
                        if use_focal:
                            loss = focal_loss(out, lbls.long(), gamma=focal_gamma)  # Focal na validação
                        else:
                            loss = criterion(out, lbls.long())  # Loss CE validação
                        running_val += loss.item() * imgs.size(0)  # Soma loss ponderada
                        preds = out.argmax(dim=1)  # Predições de classe
                        correct_val += (preds == lbls.long()).sum().item()  # Acertos no batch
                        total_val += lbls.size(0)  # Conta amostras
            if ema: ema.restore(model)  # Restaura pesos originais

            train_loss = running_loss / max(total_train, 1)  # Loss médio de treino
            val_loss = running_val / max(len(val_ds), 1)  # Loss médio de validação

            history_train_loss.append(train_loss)  # Armazena loss de treino
            history_val_loss.append(val_loss)  # Armazena loss de validação

            if mode == 'regression':  # Métricas para regressão
                train_mae = mae_sum_train / max(total_train_reg, 1)  # MAE médio de treino
                history_train_mae.append(train_mae)  # Guarda MAE treino
                if preds_list:  # Se há predições de validação
                    preds = np.concatenate(preds_list)  # Concatena predições
                    targets = np.concatenate(targs_list)  # Concatena verdadeiros
                    val_metric_value = mean_absolute_error(targets, preds)  # Calcula MAE validação
                    history_val_mae.append(val_metric_value)  # Guarda MAE val
            else:  # Métricas para classificação
                train_acc = correct_train / max(total_train_cls, 1) if total_train_cls else 0.0  # Acurácia treino
                history_train_acc.append(train_acc)  # Guarda acurácia treino
                if total_val:  # Se há validação
                    val_metric_value = correct_val / total_val  # Acurácia validação
                    history_val_acc.append(val_metric_value)  # Guarda acurácia val

            print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")  # Log de progresso
            history.append((train_loss, val_loss))  # Armazena histórico simples

            # Early stopping baseado em val accuracy (class) ou val loss (reg)
            improved = False
            if mode == 'classification' and val_metric_value is not None:
                if val_metric_value > best_val_metric:
                    improved = True
                    best_val_metric = val_metric_value
            elif mode == 'regression' and val_loss < best_val_metric:
                improved = True
                best_val_metric = val_loss

            if improved:
                best_epoch = epoch + 1
                no_improve = 0
                best_state = {k: v.cpu() for k, v in (ema.shadow if (ema and ema.shadow) else model.state_dict()).items()}
            else:
                no_improve += 1
                if early_stop_patience and no_improve >= early_stop_patience:
                    print(f"Early stopping ativado na época {epoch+1}. Melhor época: {best_epoch}")
                    break

            scheduler.step()  # Atualiza LR

        if popup is not None:
            try: popup.destroy()
            except Exception: pass

        # Se early stopping foi ativado, restaura melhor estado
        if best_state is not None:
            model.load_state_dict(best_state, strict=False)

        history_train_mae_denorm = history_train_mae  # Inicializa MAE denormalizado (treino)
        history_val_mae_denorm = history_val_mae  # Inicializa MAE denormalizado (val)
        if mode == 'regression' and age_scaler is not None:  # Para regressão, converte MAE para escala original
            mae_scale_factor = age_scaler.scale_[0]  # Fator de escala do StandardScaler
            history_train_mae_denorm = [mae * mae_scale_factor for mae in history_train_mae]  # MAE treino em anos
            history_val_mae_denorm = [mae * mae_scale_factor for mae in history_val_mae]  # MAE val em anos
            print(f"\nMAE em escala normalizada → original:")  # Log informativo
            print(f"Train MAE: {history_train_mae[-1]:.4f} → {history_train_mae_denorm[-1]:.4f} anos")  # Log MAE treino
            print(f"Val MAE: {history_val_mae[-1]:.4f} → {history_val_mae_denorm[-1]:.4f} anos")  # Log MAE val

        if history_train_loss:  # Gera curvas de aprendizagem se houver histórico
            epochs_range = range(1, len(history_train_loss) + 1)  # Eixo de épocas

            if mode == 'classification':  # Plots para classificação
                fig = Figure(figsize=(10, 4))  # Figura com dois subplots

                ax1 = fig.add_subplot(121)  # Subplot de loss
                ax1.plot(epochs_range, history_train_loss, 'b-', label='Treino')  # Loss de treino
                ax1.plot(epochs_range, history_val_loss, 'r-', label='Validação')  # Loss de validação
                ax1.set_title("Loss")  # Título
                ax1.set_xlabel("Época")  # Label eixo x
                ax1.legend()  # Legenda
                ax1.grid(True, alpha=0.3)  # Grade leve

                ax2 = fig.add_subplot(122)  # Subplot de acurácia
                if history_train_acc:
                    ax2.plot(epochs_range, history_train_acc, 'b-', label='Treino')  # Acurácia treino
                if history_val_acc:
                    ax2.plot(epochs_range, history_val_acc, 'r-', label='Validação')  # Acurácia val
                ax2.set_title("Acurácia")  # Título
                ax2.set_xlabel("Época")  # Label eixo x
                ax2.legend()  # Legenda
                ax2.grid(True, alpha=0.3)  # Grade leve
            else:  # Plots para regressão
                fig = Figure(figsize=(10, 4))  # Figura com dois subplots

                ax1 = fig.add_subplot(121)  # Subplot de loss
                ax1.plot(epochs_range, history_train_loss, 'b-', label='Treino')  # Loss treino
                ax1.plot(epochs_range, history_val_loss, 'r-', label='Validação')  # Loss val
                ax1.set_title("Loss")  # Título
                ax1.set_xlabel("Época")  # Label eixo x
                ax1.legend()  # Legenda
                ax1.grid(True, alpha=0.3)  # Grade leve

                ax2 = fig.add_subplot(122)  # Subplot de MAE
                if history_train_mae_denorm:
                    ax2.plot(epochs_range, history_train_mae_denorm, 'b-', label='Treino')  # MAE treino
                if history_val_mae_denorm:
                    ax2.plot(epochs_range, history_val_mae_denorm, 'r-', label='Validação')  # MAE val
                ax2.set_title("MAE (anos)")  # Título
                ax2.set_xlabel("Época")  # Label eixo x
                ax2.legend()  # Legenda
                ax2.grid(True, alpha=0.3)  # Grade leve

            fig.tight_layout()  # Ajusta layout

            curves_name = f"densenet_{mode}_learning_curves.png"  # Nome do arquivo de curvas
            fig.savefig(self.output_dir / curves_name, dpi=300, bbox_inches='tight')  # Salva curvas
            if not headless:
                try: self._show_plot_window("Resultados", fig)
                except Exception: pass

        torch.save(model.state_dict(), self.output_dir / f"densenet_{mode}.pth")  # Salva pesos do modelo (melhor estado)

        if mode == 'regression' and age_scaler is not None:  # Pós-processamento específico de regressão
            model.eval()  # Modo avaliação
            all_train_preds_norm, all_train_true_norm = [], []  # Armazenam predições/verdadeiros normalizados (treino)
            all_val_preds_norm, all_val_true_norm = [], []  # Armazenam predições/verdadeiros normalizados (val)
            all_test_preds_norm, all_test_true_norm = [], []  # Armazenam predições/verdadeiros normalizados (teste)

            with torch.no_grad():  # Sem gradientes
                for imgs, ages in train_loader:  # Loop treino
                    imgs = imgs.to(device)  # Move imagens
                    preds = model(imgs).squeeze()  # Predições normalizadas
                    all_train_preds_norm.extend(np.atleast_1d(preds.cpu().numpy()))  # Guarda predições
                    all_train_true_norm.extend(np.atleast_1d(ages.numpy()))  # Guarda idades reais

                for imgs, ages in val_loader:  # Loop validação
                    imgs = imgs.to(device)  # Move imagens
                    preds = model(imgs).squeeze()  # Predições
                    all_val_preds_norm.extend(np.atleast_1d(preds.cpu().numpy()))  # Guarda predições
                    all_val_true_norm.extend(np.atleast_1d(ages.numpy()))  # Guarda idades reais

                for imgs, ages in test_loader:  # Loop teste
                    imgs = imgs.to(device)  # Move imagens
                    preds = model(imgs).squeeze()  # Predições
                    all_test_preds_norm.extend(np.atleast_1d(preds.cpu().numpy()))  # Guarda predições
                    all_test_true_norm.extend(np.atleast_1d(ages.numpy()))  # Guarda idades reais

            all_train_preds_norm = np.array(all_train_preds_norm).reshape(-1, 1)  # Converte para array 2D
            all_train_true_norm = np.array(all_train_true_norm).reshape(-1, 1)  # Converte rótulos treino
            all_val_preds_norm = np.array(all_val_preds_norm).reshape(-1, 1)  # Converte val preds
            all_val_true_norm = np.array(all_val_true_norm).reshape(-1, 1)  # Converte val true
            all_test_preds_norm = np.array(all_test_preds_norm).reshape(-1, 1)  # Converte test preds
            all_test_true_norm = np.array(all_test_true_norm).reshape(-1, 1)  # Converte test true

            all_train_preds = age_scaler.inverse_transform(all_train_preds_norm).flatten()  # Desnormaliza predições treino
            all_train_true = age_scaler.inverse_transform(all_train_true_norm).flatten()  # Desnormaliza alvos treino
            all_val_preds = age_scaler.inverse_transform(all_val_preds_norm).flatten()  # Desnormaliza predições val
            all_val_true = age_scaler.inverse_transform(all_val_true_norm).flatten()  # Desnormaliza alvos val
            all_test_preds = age_scaler.inverse_transform(all_test_preds_norm).flatten()  # Desnormaliza predições teste
            all_test_true = age_scaler.inverse_transform(all_test_true_norm).flatten()  # Desnormaliza alvos teste

            train_mae_orig = mean_absolute_error(all_train_true, all_train_preds)  # MAE treino (escala original)
            train_r2 = r2_score(all_train_true, all_train_preds)  # R² treino
            train_rmse = np.sqrt(mean_squared_error(all_train_true, all_train_preds))  # RMSE treino

            val_mae_orig = mean_absolute_error(all_val_true, all_val_preds)  # MAE val
            val_r2 = r2_score(all_val_true, all_val_preds)  # R² val
            val_rmse = np.sqrt(mean_squared_error(all_val_true, all_val_preds))  # RMSE val

            test_mae_orig = mean_absolute_error(all_test_true, all_test_preds)  # MAE teste
            test_r2 = r2_score(all_test_true, all_test_preds)  # R² teste
            test_rmse = np.sqrt(mean_squared_error(all_test_true, all_test_preds))  # RMSE teste

            print(f"\n=== RESULTADOS FINAIS (escala original) ===")  # Log de resultados finais
            print(f"Train - MAE: {train_mae_orig:.4f} anos, R²: {train_r2:.4f}, RMSE: {train_rmse:.4f} anos")  # Treino
            print(f"Val   - MAE: {val_mae_orig:.4f} anos, R²: {val_r2:.4f}, RMSE: {val_rmse:.4f} anos")  # Validação
            print(f"Test  - MAE: {test_mae_orig:.4f} anos, R²: {test_r2:.4f}, RMSE: {test_rmse:.4f} anos")  # Teste

            fig_scatter = Figure(figsize=(8, 7))  # Figura para dispersão predito vs real

            ax = fig_scatter.add_subplot(111)  # Subplot único
            ax.scatter(all_test_true, all_test_preds, alpha=0.6, s=80, c='green',
                       edgecolors='darkgreen', linewidths=0.5)  # Pontos do conjunto de teste
            min_val = min(all_test_true.min(), all_test_preds.min())  # Menor valor para linha de referência
            max_val = max(all_test_true.max(), all_test_preds.max())  # Maior valor para linha de referência
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3,
                    label='Predição Perfeita', zorder=10)  # Linha de identidade
            ax.text(0.05, 0.95,
                    f'R² = {test_r2:.4f}\n'
                    f'MAE = {test_mae_orig:.2f} anos\n'
                    f'RMSE = {test_rmse:.2f} anos\n'
                    f'N = {len(all_test_true)} amostras',
                    transform=ax.transAxes, fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9,
                             edgecolor='darkgreen', linewidth=2))  # Caixa de métricas
            ax.set_xlabel('Idade Real (anos)', fontsize=13, fontweight='bold')  # Label X
            ax.set_ylabel('Idade Predita (anos)', fontsize=13, fontweight='bold')  # Label Y
            ax.set_title('Teste: Predito vs Real', fontsize=14, fontweight='bold', pad=15)  # Título gráfico
            ax.legend(loc='lower right', fontsize=11, framealpha=0.9)  # Legenda
            ax.grid(True, alpha=0.3, linestyle='--')  # Grade tracejada

            fig_scatter.tight_layout()  # Ajusta layout
            fig_scatter.savefig(self.output_dir / f"densenet_regression_scatter.png", dpi=300, bbox_inches='tight')  # Salva gráfico
            if not headless:
                try: self._show_plot_window("Gráfico de Dispersão - Teste", fig_scatter)
                except Exception: pass

            val_metric_value = test_mae_orig  # Métrica final para regressão

        test_cm = None  # Matriz de confusão de teste (classificação)
        test_acc = test_precision = test_recall = test_f1 = None
        if mode == 'classification':  # Avaliação extra para classificação
            model.eval()  # Modo avaliação
            y_true_test, y_pred_test = [], []  # Listas para rótulos verdadeiros e predições
            with torch.no_grad():  # Sem gradientes
                for imgs, lbls in test_loader:  # Loop de teste
                    imgs, lbls = imgs.to(device), lbls.to(device)  # Move dados
                    out = model(imgs)  # Forward
                    preds = out.argmax(dim=1)  # Classe predita
                    y_true_test.append(lbls.cpu().numpy())  # Coleta rótulos verdadeiros
                    y_pred_test.append(preds.cpu().numpy())  # Coleta predições

            if y_true_test:  # Calcula métricas se houver dados de teste
                y_true_test = np.concatenate(y_true_test)  # Concatena rótulos
                y_pred_test = np.concatenate(y_pred_test)  # Concatena predições
                test_cm = confusion_matrix(y_true_test, y_pred_test)  # Matriz de confusão

                test_acc = accuracy_score(y_true_test, y_pred_test)  # Acurácia teste
                test_precision = precision_score(y_true_test, y_pred_test, average='binary', zero_division=0)  # Precisão teste
                test_recall = recall_score(y_true_test, y_pred_test, average='binary', zero_division=0)  # Recall teste
                test_f1 = f1_score(y_true_test, y_pred_test, average='binary', zero_division=0)  # F1 teste

                print(f"\n=== RESULTADOS TESTE (Classificação) ===")  # Log cabeçalho
                print(f"Accuracy: {test_acc:.4f}")  # Log acurácia
                print(f"Precision: {test_precision:.4f}")  # Log precisão
                print(f"Recall: {test_recall:.4f}")  # Log recall
                print(f"F1-Score: {test_f1:.4f}")  # Log F1

                val_metric_value = test_acc  # Usa acurácia de teste como métrica final

            if test_cm is not None:  # Gera gráfico de matriz de confusão
                fig_cm = Figure(figsize=(6, 5))  # Figura da matriz
                ax = fig_cm.add_subplot(1, 1, 1)  # Eixo único
                self.plot_confusion_matrix(ax, test_cm, ['0', '1'], "Teste")  # Plota matriz
                fig_cm.tight_layout()  # Ajusta layout
                fig_cm.savefig(self.output_dir / f"confusion_densenet_{mode}.png", dpi=300, bbox_inches='tight')  # Salva imagem
                if not headless:
                    try: self._show_plot_window("Matriz de Confusão - Teste", fig_cm)
                    except Exception: pass

        learning_curves = {  # Dicionário com curvas de aprendizagem
            'train_loss': history_train_loss,
            'val_loss': history_val_loss,
        }
        if mode == 'classification':
            learning_curves['train_acc'] = history_train_acc  # Acurácia treino
            learning_curves['val_acc'] = history_val_acc  # Acurácia val
        else:
            learning_curves['train_mae'] = history_train_mae_denorm  # MAE treino desnormalizado
            learning_curves['val_mae'] = history_val_mae_denorm  # MAE val desnormalizado

        training_time = time.time() - start_time  # Tempo total do processo

        exp_payload = {  # Payload para histórico do experimento
            'model': f'DenseNet_{mode}',
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'train_loss': float(history_train_loss[-1]) if history_train_loss else None,
            'val_loss': float(history_val_loss[-1]) if history_val_loss else None,
            'learning_curves': learning_curves,
            'training_time_seconds': float(training_time),
            'best_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': lr,
            }
        }
        if mode == 'classification':
            if val_metric_value is not None:
                exp_payload['val_accuracy'] = float(val_metric_value)  # Acurácia de teste (usada como final)
                if history_train_acc:
                    exp_payload['train_accuracy'] = float(history_train_acc[-1])  # Acurácia final de treino
            if best_val_metric != -float('inf'):
                exp_payload['best_val_accuracy'] = float(best_val_metric)
                exp_payload['best_epoch'] = best_epoch
        if mode == 'regression' and val_metric_value is not None:
            exp_payload['type'] = 'regression'  # Marca tipo do experimento
            exp_payload['test_mae'] = float(val_metric_value)  # MAE final (teste)
            exp_payload['train_mae'] = float(train_mae_orig)  # MAE treino original
            exp_payload['val_mae'] = float(val_mae_orig)  # MAE val original
            exp_payload['y_train'] = all_train_true.tolist()  # Verdadeiros treino
            exp_payload['y_train_pred'] = all_train_preds.tolist()  # Predições treino
            exp_payload['y_val'] = all_val_true.tolist()  # Verdadeiros val
            exp_payload['y_val_pred'] = all_val_preds.tolist()  # Predições val
            exp_payload['y_test'] = all_test_true.tolist()  # Verdadeiros teste
            exp_payload['y_test_pred'] = all_test_preds.tolist()  # Predições teste
            exp_payload['train_r2'] = float(train_r2)  # R² treino
            exp_payload['val_r2'] = float(val_r2)  # R² val
            exp_payload['test_r2'] = float(test_r2)  # R² teste
            exp_payload['train_rmse'] = float(train_rmse)  # RMSE treino
            exp_payload['val_rmse'] = float(val_rmse)  # RMSE val
            exp_payload['test_rmse'] = float(test_rmse)  # RMSE teste

        if test_cm is not None:
            exp_payload['test_confusion_matrix'] = test_cm.tolist()  # Matriz de confusão serializada
            if mode == 'classification':
                exp_payload['test_accuracy'] = float(test_acc)  # Acurácia teste
                exp_payload['test_precision'] = float(test_precision)  # Precisão teste
                exp_payload['test_recall'] = float(test_recall)  # Recall teste
                exp_payload['test_f1'] = float(test_f1)  # F1 teste

        # Exporta embeddings antes de finalizar
        if PANDAS_AVAILABLE:
            try:
                def _export_embeddings(split_name, dataset_obj):
                    if len(dataset_obj) == 0:
                        return
                    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False)
                    emb_list, target_list, ids = [], [], []
                    model.eval()
                    idx_offset = 0
                    with torch.no_grad():
                        for imgs, lbls in loader:
                            imgs = imgs.to(device)
                            feats = model.features(imgs)  # Extrai mapas de features
                            feats = F.relu(feats, inplace=False)
                            feats = F.adaptive_avg_pool2d(feats, (1, 1)).view(feats.size(0), -1)
                            emb_list.append(feats.cpu().numpy())
                            target_list.append(lbls.cpu().numpy())
                            # Recupera meta-informação
                            rows = dataset_obj.df.iloc[idx_offset: idx_offset + len(lbls)]
                            ids.extend(rows.get('MRI_ID', rows.index).tolist())
                            idx_offset += len(lbls)

                    emb_arr = np.concatenate(emb_list)
                    tgt_arr = np.concatenate(target_list)
                    df_emb = pd.DataFrame(emb_arr)
                    df_emb.insert(0, 'MRI_ID', ids)
                    # Alvo bruto para interpretação
                    if mode == 'regression' and 'age' in dataset_obj.df.columns:
                        df_emb['target'] = dataset_obj.df.loc[:len(df_emb)-1, 'age'].values
                    else:
                        df_emb['target'] = tgt_arr
                    out_path = self.output_dir / f"densenet_embeddings_{mode}_{split_name}.csv"
                    df_emb.to_csv(out_path, index=False)

                _export_embeddings('train', train_ds)
                _export_embeddings('val', val_ds)
                _export_embeddings('test', test_ds)
            except Exception as e:
                print(f"Falha ao exportar embeddings: {e}")
        self._save_experiment(exp_payload)  # Salva experimento no histórico

        metric_msg = "Test Acc" if mode == 'classification' else "Test MAE"  # Label de métrica principal
        if val_metric_value is not None:
            fmt = "{:.2%}" if mode == 'classification' else "{:.4f}"  # Formatação por modo
            metric_msg += f": {fmt.format(val_metric_value)}"  # Anexa valor final
        if mode == 'classification':
            metric_msg += f"\nBest Val Acc: {best_val_metric:.2%} @ epoch {best_epoch}"
        try:
            if not headless:
                messagebox.showinfo("DenseNet", f"Treino concluído. {metric_msg}")
            else:
                print(f"[DenseNet] {metric_msg}")
        except Exception:
            print(f"[DenseNet] {metric_msg}")

    def refine_densenet_with_rl(self, episodes=8, horizon=4, micro_epochs=1,
                                train_subset=120, val_subset=80):  # Refinamento via RL
        if not (SKLEARN_AVAILABLE and TORCH_AVAILABLE and PANDAS_AVAILABLE):
            messagebox.showerror("Dependência ausente",
                                 "PyTorch, scikit-learn e pandas são necessários para o refinamento com RL.")
            return

        df_path = self.output_dir / "exam_level_dataset_split.csv"
        if not df_path.exists():
            messagebox.showwarning("Aviso", "Crie o dataset (Criar Dataset) antes de rodar o RL.")
            return

        base_ckpt = self.output_dir / "densenet_classification.pth"
        if not base_ckpt.exists():
            messagebox.showwarning("Aviso", "Treine a DenseNet de classificação antes de refinar com RL.")
            return

        df = pd.read_csv(df_path)
        df_train = df[df['split'] == 'train'].copy()
        df_val = df[df['split'] == 'validation'].copy()
        df_test = df[df['split'] == 'test'].copy()
        if df_train.empty or df_val.empty:
            messagebox.showwarning("Aviso", "Splits de treino/validação vazios para classificação.")
            return

        device = select_device()
        train_tf, val_tf = build_transforms()

        def _sample(df_split, n):
            if len(df_split) <= n:
                return df_split
            return df_split.sample(n=n, random_state=42)

        df_train_small = _sample(df_train, train_subset)
        df_val_small = _sample(df_val, val_subset)

        batch_small = 8
        train_loader_small = DataLoader(
            MRIDataset(df_train_small, train_tf, self.dataset_dir.parent, 'original_path', 'Final_Group'),
            batch_size=batch_small, shuffle=True
        )
        val_loader_small = DataLoader(
            MRIDataset(df_val_small, val_tf, self.dataset_dir.parent, 'original_path', 'Final_Group'),
            batch_size=batch_small, shuffle=False
        )

        # Pesos de classe para lidar com desbalanceamento
        class_weights = None
        class_counts = df_train['Final_Group'].value_counts()
        if len(class_counts) >= 1:
            total = class_counts.sum()
            w0 = total / (2 * class_counts.get('Nondemented', max(class_counts.max(), 1)))
            w1 = total / (2 * class_counts.get('Demented', max(class_counts.max(), 1)))
            class_weights = torch.tensor([w0, w1], dtype=torch.float32)

        popup = tk.Toplevel(self.root)
        tk.Label(popup, text=f"Refinando (RL)... {episodes} episódios").pack(padx=20, pady=20)
        self.root.update()

        rl_history = {"episodes": [], "actions": []}
        curve_train_loss, curve_val_loss = [], []
        curve_train_acc, curve_val_acc = [], []
        curve_rewards = []
        best_state, best_hparams = None, None
        try:
            env = DenseNetRefineEnv(
                train_loader=train_loader_small,
                val_loader=val_loader_small,
                device=device,
                base_checkpoint=base_ckpt,
                class_weights=class_weights,
                micro_epochs=micro_epochs,
                max_batches_per_epoch=3
            )
            agent = PPOAgent(state_dim=env.state_dim, action_dim=env.action_dim, device=device)
            rl_history["actions"] = env.actions

            state = env.reset()
            for ep in range(episodes):
                ep_reward = 0.0
                ep_steps = []
                for _ in range(horizon):
                    action_idx, log_prob, value_est = agent.select_action(state)
                    next_state, reward, info = env.step(action_idx)
                    agent.store(state, action_idx, log_prob, value_est, reward)
                    ep_reward += reward
                    ep_steps.append(info)
                    curve_train_loss.append(info["train_loss"])
                    curve_val_loss.append(info["val_loss"])
                    curve_train_acc.append(info["train_acc"])
                    curve_val_acc.append(info["val_acc"])
                    curve_rewards.append(reward)
                    state = next_state

                update_stats = agent.update()
                best_state, best_hparams = env.get_best_checkpoint()
                rl_history["episodes"].append({
                    "episode": ep + 1,
                    "reward_sum": float(ep_reward),
                    "last_val_acc": float(env.last_val_acc),
                    "last_val_loss": float(env.last_val_loss),
                    "best_val_acc": float(env.best_val_acc),
                    "steps": ep_steps,
                    "update": update_stats,
                })
                state = env.reset()
        finally:
            try:
                popup.destroy()
            except Exception:
                pass

        if best_state is None:
            best_state = {k: v.cpu() for k, v in env.model.state_dict().items()}
            best_hparams = env.state

        # Reconstroi modelo com melhores hiperparâmetros para avaliação completa
        env.state.update(best_hparams or {})
        eval_model = env._build_model()
        eval_model.load_state_dict(best_state, strict=False)
        eval_model = eval_model.to(device)

        val_loader_full = DataLoader(
            MRIDataset(df_val, val_tf, self.dataset_dir.parent, 'original_path', 'Final_Group'),
            batch_size=16, shuffle=False
        )
        test_loader_full = DataLoader(
            MRIDataset(df_test, val_tf, self.dataset_dir.parent, 'original_path', 'Final_Group'),
            batch_size=16, shuffle=False
        )

        val_metrics = evaluate_full_model(eval_model, val_loader_full, device)
        test_metrics = evaluate_full_model(eval_model, test_loader_full, device)

        def _collect_preds(loader):
            eval_model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for imgs, labels in loader:
                    imgs = imgs.to(device)
                    labels = labels.to(device).long()
                    logits = eval_model(imgs)
                    preds = logits.argmax(dim=1)
                    y_true.append(labels.cpu().numpy())
                    y_pred.append(preds.cpu().numpy())
            if not y_true:
                return np.array([]), np.array([])
            return np.concatenate(y_true), np.concatenate(y_pred)

        y_val, y_val_pred = _collect_preds(val_loader_full)
        y_test, y_test_pred = _collect_preds(test_loader_full)
        val_cm = confusion_matrix(y_val, y_val_pred) if y_val.size else None
        test_cm = confusion_matrix(y_test, y_test_pred) if y_test.size else None

        best_model_path = self.output_dir / "densenet_classification_rl_best.pth"
        torch.save(best_state, best_model_path)
        policy_path = self.output_dir / "rl_policy_densenet.pth"
        torch.save(agent.policy.state_dict(), policy_path)

        history_writer = TrainHistoryWriter(self.output_dir)
        rl_history["meta"] = {
            "episodes": episodes,
            "horizon": horizon,
            "micro_epochs": micro_epochs,
            "train_subset": len(df_train_small),
            "val_subset": len(df_val_small),
            "best_val_acc": float(env.best_val_acc),
            "base_checkpoint": base_ckpt.name,
        }
        history_file = history_writer.save(rl_history)

        exp_payload = {
            'model': 'DenseNet_classification_RL',
            'episodes': episodes,
            'horizon': horizon,
            'micro_epochs': micro_epochs,
            'train_subset': len(df_train_small),
            'val_subset': len(df_val_small),
            'best_val_acc': float(env.best_val_acc),
            'val_accuracy': float(val_metrics.get("acc", 0.0)),
            'test_accuracy': float(test_metrics.get("acc", 0.0)),
            'val_loss': float(val_metrics.get("loss", 0.0)),
            'test_loss': float(test_metrics.get("loss", 0.0)),
            'history_file': history_file.name,
            'best_model_path': best_model_path.name,
            'policy_path': policy_path.name,
            'best_hparams': best_hparams,
        }
        if curve_train_loss and len(curve_train_loss) == len(curve_val_loss):
            exp_payload['learning_curves'] = {
                'train_loss': curve_train_loss,
                'val_loss': curve_val_loss,
                'train_acc': curve_train_acc,
                'val_acc': curve_val_acc,
                'reward': curve_rewards,
            }
        if val_cm is not None:
            exp_payload['val_confusion_matrix'] = val_cm.tolist()
            exp_payload['val_classes'] = ['Nondemented', 'Demented']
        if test_cm is not None:
            exp_payload['test_confusion_matrix'] = test_cm.tolist()
            exp_payload['test_classes'] = ['Nondemented', 'Demented']

        self._save_experiment(exp_payload)
        # Curvas de aprendizagem do ciclo de RL
        if curve_train_loss and curve_val_loss:
            from matplotlib.figure import Figure
            fig_rl = Figure(figsize=(10, 4))
            ax1 = fig_rl.add_subplot(121)
            steps = range(1, len(curve_train_loss) + 1)
            ax1.plot(steps, curve_train_loss, 'b-', label='Treino')
            ax1.plot(steps, curve_val_loss, 'r-', label='Validação')
            ax1.set_title("Loss (passos RL)")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = fig_rl.add_subplot(122)
            if curve_train_acc: ax2.plot(steps, curve_train_acc, 'b-', label='Treino')
            if curve_val_acc: ax2.plot(steps, curve_val_acc, 'r-', label='Validação')
            ax2.set_title("Acurácia (passos RL)")
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            fig_rl.tight_layout()
            curves_path = self.output_dir / "densenet_classification_rl_learning_curves.png"
            fig_rl.savefig(curves_path, dpi=300, bbox_inches='tight')
            try:
                self._show_plot_window("Curvas RL", fig_rl)
            except Exception:
                pass

        msg = (
            f"Melhor Val Acc (RL): {env.best_val_acc:.2%}\n"
            f"Val (full) Acc: {val_metrics.get('acc', 0):.2%}\n"
            f"Teste Acc: {test_metrics.get('acc', 0):.2%}\n"
            f"Histórico: {history_file.name}\n"
            f"Curvas: densenet_classification_rl_learning_curves.png"
        )

        # Treino final robusto usando hiperparâmetros encontrados pelo RL no dataset completo
        if best_hparams:
            try:
                print("Treinando DenseNet final com hiperparâmetros do RL (split completo)...")
                self._train_pytorch_model(mode='classification', hparams=best_hparams)
            except Exception as e:
                print(f"Falha ao treinar modelo final com hparams do RL: {e}")

        messagebox.showinfo("DenseNet + RL", msg)
