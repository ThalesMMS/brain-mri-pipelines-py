import os  # Configuração de ambiente/threads para libs nativas
import pickle  # Serialização de objetos Python
import random  # Controle determinístico de seeds
import time  # Funções relacionadas a tempo e pausas
from contextlib import nullcontext  # Contexto vazio para fallback de autocast
from pathlib import Path  # Manipulação conveniente de caminhos
import hashlib  # Hash reprodutível de arquivos (split)

# Tkinter é opcional: scripts headless/CI podem não ter _tkinter instalado.
try:
    import tkinter as tk  # Interface gráfica
    from tkinter import messagebox  # Diálogos do Tkinter
    TK_AVAILABLE = True
except Exception:
    tk = None
    TK_AVAILABLE = False

    class _HeadlessMessageBox:
        def showinfo(self, *args, **kwargs):
            return None

        def showwarning(self, *args, **kwargs):
            return None

        def showerror(self, *args, **kwargs):
            return None

        def askyesno(self, *args, **kwargs):
            return True

    messagebox = _HeadlessMessageBox()

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
    from .medicalnet_models import MultimodalMedicalNet
    from .multistream_models import MultiOrientTabularFusionNet
    from .training_utils import (
        ExponentialMovingAverage,


        build_medicalnet,
        build_transforms,
        focal_loss,
        mixup_data,
        select_device,
    )
    try:
        from .debug_tools import debug_batch, debug_one_step
    except Exception:
        debug_batch = debug_one_step = None
    TORCH_AVAILABLE = True
except ImportError:
    torch = nn = optim = DataLoader = None
    ExponentialMovingAverage = build_medicalnet = build_transforms = focal_loss = mixup_data = select_device = None
    debug_batch = debug_one_step = None
    TORCH_AVAILABLE = False
try:
    import xgboost as xgb  # Biblioteca de gradient boosting XGBoost
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
try:
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                                 precision_score, r2_score, recall_score)  # Métricas de avaliação
    from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split  # Divisão de dados e validação cruzada
    from sklearn.preprocessing import StandardScaler  # Normalização de características
    from sklearn.svm import SVC  # Suporte a máquinas de vetor para classificação
    SKLEARN_AVAILABLE = True
except ImportError:
    accuracy_score = balanced_accuracy_score = confusion_matrix = f1_score = mean_absolute_error = mean_squared_error = precision_score = r2_score = recall_score = None
    GridSearchCV = GroupKFold = train_test_split = StandardScaler = SVC = None
    SKLEARN_AVAILABLE = False
try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # Canvas e barra de ferramentas do Matplotlib integrados ao Tkinter
except Exception:
    FigureCanvasTkAgg = None
    NavigationToolbar2Tk = None

from matplotlib.figure import Figure  # Objeto de figura do Matplotlib

from .datasets import MRIDataset, MultiOrientMRIDataset  # Dataset PyTorch específico


class MLTrainingMixin:  # Métodos de criação de dataset e treinamento de modelos
    def _list_orientation_paths(self, mri_id: str):
        """Lista caminhos disponíveis para o mesmo exame nas orientações axl/cor/sag."""
        base = self.dataset_dir.parent
        paths = []
        # Expande para incluir formatos de imagem comuns além de NIfTI
        valid_exts = (".nii.gz", ".nii", ".png", ".jpg", ".jpeg")
        for orient in ("axl", "cor", "sag"):
            found = False
            for ext in valid_exts:
                cand = base / orient / f"{mri_id}_{orient}{ext}"
                if cand.exists():
                    try:
                        rel = cand.relative_to(base)
                        paths.append(str(rel))
                        found = True
                        break # Encontrou para esta orientação, passa para próxima ext
                    except ValueError:
                        paths.append(str(cand))
                        found = True
                        break
        return paths

    def _populate_orientation_paths(self, df_subset):
        """Assegura que existe campo orientation_paths para cada exame (1 linha por exame)."""
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
            # Mantém 1 linha por exame, mas enriquece com lista de paths
            r = row.copy()
            r["orientation_paths"] = orient_paths
            # Define original_path como axial se existir, senão primeiro disponível
            axial = [p for p in orient_paths if "_axl" in p]
            r["original_path"] = axial[0] if axial else orient_paths[0]
            rows.append(r)
        return type(df_subset)(rows)
    def create_exam_level_dataset(self):  # Constrói dataset unindo descritores e dados demográficos com split
        """Junta CSV demográfico com descritores e faz split.

        Versão atual: o conjunto de exames/pacientes é definido pela união dos arquivos existentes em
        `axl/`, `cor/`, `sag/`, tolerando ausência de algumas orientações (1–2 planos).
        Descritores ventriculares (quando disponíveis) são incorporados por left join.
        """
        if not PANDAS_AVAILABLE:
            messagebox.showerror(
                "Dependência ausente",
                "O módulo 'pandas' é necessário para criar o dataset.\nInstale com 'pip install pandas'.",
            )
            return

        def _parse_mri_subject_ids(filename: str):
            # Ex.: OAS2_0001_MR1_axl.nii.gz -> (MRI_ID=OAS2_0001_MR1, Subject_ID=OAS2_0001)
            name = filename
            for ext in (".nii.gz", ".nii", ".png", ".jpg", ".jpeg"):
                if name.endswith(ext):
                    name = name[: -len(ext)]
                    break
            parts = name.split('_')
            if len(parts) < 3:
                return None, None
            subj_id = '_'.join(parts[:2])
            mri_id = '_'.join(parts[:3])
            return mri_id, subj_id

        def _build_union_index(base_dir: Path):
            """Cria índice (união) de exames presentes em axl/cor/sag."""
            valid_exts = (".nii.gz", ".nii", ".png", ".jpg", ".jpeg")
            by_mri = {}
            for orient in ("axl", "cor", "sag"):
                orient_dir = base_dir / orient
                if not orient_dir.exists():
                    continue
                for ext in valid_exts:
                    for f in orient_dir.glob(f"*{ext}"):
                        mri_id, subj_id = _parse_mri_subject_ids(f.name)
                        if not mri_id or not subj_id:
                            continue
                        rec = by_mri.setdefault(
                            mri_id,
                            {
                                'MRI_ID': mri_id,
                                'Subject_ID': subj_id,
                                'paths': [],
                                'has_axl': False,
                                'has_cor': False,
                                'has_sag': False,
                            },
                        )
                        try:
                            rel = f.relative_to(base_dir)
                            rel_s = str(rel)
                        except ValueError:
                            rel_s = str(f)
                        if rel_s not in rec['paths']:
                            rec['paths'].append(rel_s)
                        rec[f'has_{orient}'] = True

            rows = []
            for rec in by_mri.values():
                paths = rec['paths']
                axial = [p for p in paths if "_axl" in p]
                original_path = axial[0] if axial else (paths[0] if paths else "")
                rows.append(
                    {
                        'MRI_ID': rec['MRI_ID'],
                        'Subject_ID': rec['Subject_ID'],
                        'orientation_paths': rec['paths'],
                        'original_path': original_path,
                        'has_axl': bool(rec['has_axl']),
                        'has_cor': bool(rec['has_cor']),
                        'has_sag': bool(rec['has_sag']),
                    }
                )
            return pd.DataFrame(rows)

        base_dir = self.dataset_dir.parent
        df_union = _build_union_index(base_dir)
        if df_union.empty:
            messagebox.showwarning("Aviso", "Nenhuma imagem encontrada em axl/cor/sag para montar o dataset.")
            return

        # Descritores ventriculares: podem existir apenas para axial. Mantemos por left join.
        if self.descriptors_csv.exists():
            df_desc_raw = pd.read_csv(self.descriptors_csv)
            if df_desc_raw.empty:
                messagebox.showwarning("Aviso", "CSV de descritores vazio. O dataset será criado sem descritores.")
                df_desc = df_union.copy()
                df_desc['viable'] = True
            else:
                if 'viable' not in df_desc_raw.columns:
                    df_desc_raw['viable'] = True
                df_desc = pd.merge(df_union, df_desc_raw, on='MRI_ID', how='left', suffixes=('', '_desc'))
                df_desc['viable'] = df_desc.get('viable', True).fillna(True)
        else:
            df_desc = df_union.copy()
            df_desc['viable'] = True

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

        # Mantém o caminho base já calculado a partir da união axl/cor/sag.
        # Não deve sobrescrever usando _resolve_original_path (pode assumir apenas axial).
        if 'original_path' in merged.columns:
            merged['original_path'] = merged['original_path'].fillna("")
        else:
            merged['original_path'] = ""
        merged = merged[merged['original_path'] != ""]  # Remove registros sem caminho válido

        # Flag de disponibilidade de descritores ventriculares (útil para modelos tabulares).
        descriptor_cols = [c for c in merged.columns if c.startswith('ventricle_')]
        if descriptor_cols:
            merged['has_descriptors'] = merged[descriptor_cols].notna().any(axis=1)
        else:
            merged['has_descriptors'] = False
        merged = merged[merged['Subject_ID'].notna()]  # Filtra registros com Subject_ID

        subjects = merged['Subject_ID'].dropna().unique()  # Lista de sujeitos únicos
        if len(subjects) < 3:  # Requer mínimo de 3 sujeitos para split
            messagebox.showwarning("Aviso", "Dados insuficientes para split (mínimo 3 sujeitos).")  # Alerta insuficiência
            return  # Interrompe

        split_seed = int(os.getenv("DENSENET_SEED", "42"))

        # Split por sujeito estratificado por classe para evitar val/test sem classe minoritária
        subj_label = (
            merged[['Subject_ID', 'Final_Group']]
            .dropna(subset=['Subject_ID', 'Final_Group'])
            .groupby('Subject_ID')['Final_Group']
            .apply(lambda s: int((s == 'Demented').mean() >= 0.5))
        )
        subj_ids = subj_label.index.to_numpy()
        subj_y = subj_label.values

        def _split_has_both(sub_ids):
            vc = merged[merged['Subject_ID'].isin(sub_ids)]['Final_Group'].value_counts()
            return vc.get('Nondemented', 0) > 0 and vc.get('Demented', 0) > 0

        train_sub = val_sub = test_sub = None
        for attempt in range(500):
            rs = split_seed + attempt
            try:
                tr, te = train_test_split(
                    subj_ids, test_size=0.2, random_state=rs,
                    stratify=subj_y if len(np.unique(subj_y)) > 1 else None
                )
                y_tr = np.array([subj_label[sid] for sid in tr])
                tr, va = train_test_split(
                    tr, test_size=0.2, random_state=rs,
                    stratify=y_tr if len(np.unique(y_tr)) > 1 else None
                )
            except Exception:
                continue
            if _split_has_both(tr) and _split_has_both(va) and _split_has_both(te):
                train_sub, val_sub, test_sub = tr, va, te
                break

        if train_sub is None:
            # fallback: garante pelo menos treino/val com ambas as classes
            for attempt in range(500):
                rs = split_seed + attempt
                try:
                    tr, te = train_test_split(
                        subj_ids, test_size=0.2, random_state=rs,
                        stratify=subj_y if len(np.unique(subj_y)) > 1 else None
                    )
                    y_tr = np.array([subj_label[sid] for sid in tr])
                    tr, va = train_test_split(
                        tr, test_size=0.2, random_state=rs,
                        stratify=y_tr if len(np.unique(y_tr)) > 1 else None
                    )
                except Exception:
                    continue
                if _split_has_both(tr) and _split_has_both(va):
                    train_sub, val_sub, test_sub = tr, va, te
                    print("[WARN] Não foi possível garantir ambas as classes no TESTE; mantendo split com ambas em TREINO/VAL.")
                    break

        if train_sub is None:
            print("[WARN] Split estratificado por sujeito falhou; usando split aleatório simples.")
            train_sub, test_sub = train_test_split(subjects, test_size=0.2, random_state=split_seed)
            train_sub, val_sub = train_test_split(train_sub, test_size=0.2, random_state=split_seed)

        def get_split(sid):  # Função auxiliar para mapear sujeito para split
            if sid in val_sub: return 'validation'  # Sujeitos de validação
            if sid in test_sub: return 'test'  # Sujeitos de teste
            return 'train'  # Demais em treino

        merged['split'] = merged['Subject_ID'].apply(get_split)  # Aplica divisão por sujeito

        # Diagnóstico: distribuição por split
        try:
            print("\n[Diagnóstico] Final_Group por split:")
            print(merged.groupby('split')['Final_Group'].value_counts())
        except Exception:
            pass

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

    def train_svm_classifier(self, features=None, scenario=None):  # Treina classificador SVM com seleção de features
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

        uses_mmse = any(str(f).lower() == 'mmse' for f in features)
        uses_cdr = any(str(f).lower() == 'cdr' for f in features)
        scenario_label = scenario or ("svm_with_mmse_cdr" if (uses_mmse or uses_cdr) else "svm_without_mmse_cdr")

        tmp = df.copy()  # Copia para ajustes

        # Se o modelo usa descritores ventriculares, filtra para linhas com descritores.
        uses_descriptors = any(str(f).startswith('ventricle_') for f in features)
        if uses_descriptors:
            if 'has_descriptors' in tmp.columns:
                tmp = tmp[tmp['has_descriptors'] == True]
            else:
                desc_cols = [c for c in tmp.columns if c.startswith('ventricle_')]
                if desc_cols:
                    tmp = tmp[tmp[desc_cols].notna().any(axis=1)]
        if 'sex' in features and 'sex' not in tmp.columns:  # Se sexo solicitado mas ausente
            if 'M/F' in tmp.columns:
                tmp['sex'] = tmp['M/F'].map({'M': 0, 'F': 1})  # Converte M/F para binário
            else:
                tmp['sex'] = np.nan  # Preenche com NaN caso não exista

        missing = [f for f in features if f not in tmp.columns]  # Checa colunas faltantes
        if missing:  # Se houver faltantes
            messagebox.showerror("Erro", f"Colunas ausentes no dataset: {missing}")  # Erro informativo
            return  # Sai

        X_df = tmp[features].copy()  # Seleciona features
        # Preenche NaNs com médias calculadas apenas no treino (evita vazamento).
        train_mask = tmp['split'] == 'train'
        train_means = X_df.loc[train_mask].mean(numeric_only=True)
        train_means = train_means.fillna(0.0)
        X_df = X_df.fillna(train_means)
        X_df = X_df.fillna(0.0)
        X = X_df.values  # Converte para array
        y = (tmp['Final_Group'] == 'Demented').astype(int).values  # Alvo binário: demente ou não

        val_mask = tmp['split'] == 'validation'  # Máscara de validação
        test_mask = tmp['split'] == 'test'  # Máscara de teste

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

        # Evita sobrescrever artefatos quando rodando múltiplos cenários.
        if scenario is None:
            scaler_path = self.output_dir / "svm_scaler.pkl"
            model_path = self.output_dir / "svm_model.pkl"
        else:
            safe = str(scenario_label).replace(' ', '_').replace('/', '-')
            scaler_path = self.output_dir / f"svm_scaler_{safe}.pkl"
            model_path = self.output_dir / f"svm_model_{safe}.pkl"

        with open(scaler_path, "wb") as f:  # Salva scaler
            pickle.dump(scaler, f)  # Serializa scaler
        with open(model_path, "wb") as f:  # Salva modelo SVM
            pickle.dump(clf, f)  # Serializa modelo

        training_time = time.time() - start_time  # Calcula tempo total de treino

        exp_data = {  # Dados do experimento para histórico
            'model': 'SVM',
            'scenario': scenario_label,
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

    def train_xgboost_regressor(self, features=None, scenario=None, seed=None):  # Treina regressor XGBoost para predição de idade
        if not SKLEARN_AVAILABLE:
            messagebox.showerror("Dependência ausente", "O módulo 'scikit-learn' é necessário para treinar o XGBoost.\nInstale com 'pip install scikit-learn'.")
            return
        if not XGBOOST_AVAILABLE:
            messagebox.showerror("Dependência ausente", "O módulo 'xgboost' é necessário para este treino.\nInstale com 'pip install xgboost'.")
            return
        if not PANDAS_AVAILABLE:
            messagebox.showerror("Dependência ausente", "O módulo 'pandas' é necessário para treinar o XGBoost.\nInstale com 'pip install pandas'.")
            return
        # Reprodutibilidade: fixa seed local e no XGBoost.
        if seed is None:
            try:
                seed = int(os.getenv("XGB_SEED", "42"))
            except Exception:
                seed = 42

        try:
            random.seed(seed)
            np.random.seed(seed)
        except Exception:
            pass

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

        # Se o modelo usa descritores ventriculares, filtra para linhas com descritores.
        uses_descriptors = any(str(f).startswith('ventricle_') for f in features)
        if uses_descriptors:
            if 'has_descriptors' in tmp.columns:
                tmp = tmp[tmp['has_descriptors'] == True]
            else:
                desc_cols = [c for c in tmp.columns if c.startswith('ventricle_')]
                if desc_cols:
                    tmp = tmp[tmp[desc_cols].notna().any(axis=1)]
        if 'sex' in features and 'sex' not in tmp.columns and 'M/F' in tmp.columns:  # Cria sexo binário se necessário
            tmp['sex'] = tmp['M/F'].map({'M': 0, 'F': 1})  # Mapeia sexo

        missing = [f for f in features if f not in tmp.columns]  # Verifica colunas faltantes
        if missing:
            messagebox.showerror("Erro", f"Colunas ausentes no dataset: {missing}")  # Erro se faltar coluna
            return  # Sai

        # Preenche NaNs usando médias do TREINO (evita vazamento).
        train_mask = tmp['split'] == 'train'
        X_df = tmp[features].copy()
        train_means = X_df.loc[train_mask].mean(numeric_only=True).fillna(0.0)
        X_df = X_df.fillna(train_means).fillna(0.0)
        X = X_df.values
        y = tmp['age'].values  # Alvo: idade

        val_mask = tmp['split'] == 'validation'  # Máscara de validação
        test_mask = tmp['split'] == 'test'  # Máscara de teste
        if not val_mask.any():  # Exige validação
            messagebox.showwarning("Aviso", "Split de validação vazio.")  # Alerta ausência
            return  # Sai

        groups = tmp.loc[train_mask, 'Subject_ID']  # Grupos para CV em nível de sujeito

        base = xgb.XGBRegressor(  # Modelo base XGBoost
            objective='reg:squarederror',
            tree_method='hist',
            n_jobs=1,
            verbosity=0,
            random_state=int(seed),
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

        test_mae = test_rmse = test_r2 = test_mse = None
        if test_mask.any():
            test_preds = model.predict(X[test_mask])
            test_mae = mean_absolute_error(y[test_mask], test_preds)
            test_mse = mean_squared_error(y[test_mask], test_preds)
            test_rmse = float(np.sqrt(test_mse))
            test_r2 = r2_score(y[test_mask], test_preds)

        msg = f"Val MAE={mae_val:.2f} | Val RMSE={rmse_val:.2f} | Val MSE={mse_val:.2f} | Val R²={r2_val:.4f}"
        if test_mae is not None:
            msg += f"\nTest MAE={test_mae:.2f} | Test RMSE={test_rmse:.2f} | Test MSE={test_mse:.2f} | Test R²={test_r2:.4f}"

        messagebox.showinfo("XGBoost", msg)  # Exibe métricas de validação/teste

        with open(self.output_dir / "xgb_age.pkl", "wb") as f:  # Salva modelo treinado
            pickle.dump(model, f)  # Serializa regressor

        training_time = time.time() - start_time  # Tempo total de treino

        scenario_label = scenario or "xgb_train_and_test_current_split"

        # Fingerprint do dataset/split para auditoria (não depende de rede).
        split_sha256 = None
        try:
            import hashlib
            split_sha256 = hashlib.sha256(df_path.read_bytes()).hexdigest()
        except Exception:
            split_sha256 = None

        exp_payload = {  # Registra experimento
            'model': 'XGBoost',  # Nome do modelo
            'scenario': scenario_label,
            'target': 'age',  # Variável alvo prevista
            'features': features,  # Lista de features usadas
            'val_mae': float(mae_val),  # MAE na validação
            'val_mse': float(mse_val),  # MSE na validação
            'val_rmse': float(rmse_val),  # RMSE na validação
            'val_r2': float(r2_val),  # R² na validação
            'best_params': gs.best_params_,  # Hiperparâmetros ótimos
            'training_time_seconds': float(training_time),  # Duração do treino em segundos
            'seed': int(seed),
            'split_csv_sha256': split_sha256,
        }

        if test_mae is not None:
            exp_payload.update({
                'test_mae': float(test_mae),
                'test_mse': float(test_mse),
                'test_rmse': float(test_rmse),
                'test_r2': float(test_r2),
            })

        self._save_experiment(exp_payload)  # Salva experimento no histórico

    def train_efficientnet_classifier(self):  # Wrapper para treinar EfficientNet em modo classificação
        if not TORCH_AVAILABLE:
            messagebox.showerror("Dependência ausente", "PyTorch/torchvision são necessários para treinar a EfficientNet.\nInstale com 'pip install torch torchvision'.")
            return
        self._train_pytorch_model(mode='classification', backbone='efficientnet')

    def train_efficientnet_regressor(self):  # Wrapper para treinar EfficientNet em modo regressão
        if not TORCH_AVAILABLE:
            messagebox.showerror("Dependência ausente", "PyTorch/torchvision são necessários para treinar a EfficientNet.\nInstale com 'pip install torch torchvision'.")
            return
        self._train_pytorch_model(mode='regression', backbone='efficientnet')

    def train_densenet_classifier(self):
        if not TORCH_AVAILABLE:
             messagebox.showerror("Dependência ausente", "PyTorch necessário.")
             return
        self._train_pytorch_model(mode='classification', backbone='densenet')

    def train_densenet_regressor(self):
        if not TORCH_AVAILABLE:
             messagebox.showerror("Dependência ausente", "PyTorch necessário.")
             return
        self._train_pytorch_model(mode='regression', backbone='densenet')

    def train_medicalnet_classifier(self):
        if not TORCH_AVAILABLE:
             messagebox.showerror("Dependência ausente", "PyTorch necessário.")
             return
        self._train_pytorch_model(mode='classification', backbone='medicalnet')

    def train_medicalnet_regressor(self):
        if not TORCH_AVAILABLE:
             messagebox.showerror("Dependência ausente", "PyTorch necessário.")
             return
        self._train_pytorch_model(mode='regression', backbone='medicalnet')

    def _train_pytorch_model(self, mode='classification', backbone='medicalnet', hparams=None):  # Treina modelo PyTorch (DenseNet, MedicalNet)
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
        checkpoint_suffix = "classifier" if mode == 'classification' else 'regressor'
        best_checkpoint_path = self.output_dir / f"best_{backbone}_{checkpoint_suffix}.pth"
        legacy_checkpoint_path = self.output_dir / f"{backbone}_{mode}.pth"

        split_override = os.environ.get("SPLIT_CSV_PATH", "").strip()
        if split_override:
            df_path = Path(split_override)
            df = pd.read_csv(df_path)
            print(f"[DATA] USING SPLIT CSV: {df_path} | shape={df.shape}")
        else:
            df_path = self.output_dir / "exam_level_dataset_split.csv"  # Caminho do dataset combinado
            if not df_path.exists():
                print("[DATA] USING DEFAULT SPLIT SOURCE - arquivo ausente.")
                return  # Sai se o dataset ainda não foi criado
            df = pd.read_csv(df_path)  # Carrega dataset consolidado
            print(f"[DATA] USING DEFAULT SPLIT SOURCE: {df_path} | shape={df.shape}")

        assert 'split' in df.columns, f"CSV sem coluna split. Colunas: {df.columns.tolist()}"
        assert 'Final_Group' in df.columns, f"CSV sem coluna Final_Group. Colunas: {df.columns.tolist()}"

        for s in ["train", "validation", "test"]:
            sub = df[df["split"] == s]
            vc = sub["Final_Group"].value_counts(dropna=False).to_dict()
            print(f"[DATA] {s}: n={len(sub)} | by_class={vc}")
            if len(sub) == 0:
                raise ValueError(f"Split {s} está vazio (treino inválido).")

        device = select_device()
        print(f"Dispositivo selecionado: {device} | Torch threads: {torch.get_num_threads()} | Backbone: {backbone}")  # Log para debugging

        # Hiperparâmetros (podem ser sobrescritos por hparams)
        # Ajuste de LR: Se backbone congelado, heads precisam de LR maior (1e-3). Se fine-tuning total, menor (1e-4/5e-5).
        # Assumiremos freeze="1" -> 1e-3 default, senão 5e-5.
        default_freeze = os.getenv("RESNET_FREEZE", "1") == "1"
        default_lr = 1e-3 if default_freeze else (5e-5 if mode == 'classification' else 1e-3)
        
        defaults = {
            "lr": float(os.getenv("RESNET_LR", str(default_lr))),
            "weight_decay": float(os.getenv("RESNET_WEIGHT_DECAY", 1e-4 if mode == 'classification' else 0.0)),
            "dropout": float(os.getenv("RESNET_DROPOUT", 0.25)),
            "label_smoothing": float(os.getenv("RESNET_LABEL_SMOOTH", 0.05 if mode == 'classification' else 0.0)),
            "mixup_alpha": float(os.getenv("RESNET_MIXUP", 0.0)),
            "freeze_backbone": os.getenv("RESNET_FREEZE", "1") == "1",
            "class_balance": os.getenv("RESNET_CLASS_WEIGHTS", "0") == "1",
            "freeze_warmup_epochs": int(os.getenv("RESNET_WARMUP_EPOCHS", "0")),
            "warmup_lr": float(os.getenv("RESNET_WARMUP_LR", "0")) or None,
            "balance_penalty": float(os.getenv("RESNET_BALANCE_PENALTY", 0.0)),
            "thresholds_eval": [float(x) for x in os.getenv("RESNET_THRESHOLDS", "0.5,0.6,0.4,0.7").split(',')],
            "seed": int(os.getenv("RESNET_SEED", "42")),
            "medicalnet_depth": 18,
        }
        if hparams:
            for k, v in hparams.items():
                if k in defaults or k == 'medicalnet_depth':
                    defaults[k] = v

        # Fixar seeds para reprodutibilidade
        seed = defaults.get("seed", 42)
        try:
            seed = int(seed)
        except Exception:
            seed = 42
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Hash do CSV de split usado (auditoria/reprodutibilidade)
        split_csv_sha256 = None
        try:
            split_csv_sha256 = hashlib.sha256(df_path.read_bytes()).hexdigest()
        except Exception:
            split_csv_sha256 = None

        deep_scenario_base = (os.getenv('DEEP_SCENARIO', 'deep_current_split') or 'deep_current_split').strip()
        deep_scenario_label = f"{deep_scenario_base}_{backbone}_{mode}_seed{seed}"

        lr = defaults["lr"]
        weight_decay = defaults["weight_decay"]
        dropout_rate = defaults["dropout"]
        label_smoothing = defaults["label_smoothing"]
        mixup_alpha = float(defaults["mixup_alpha"])
        freeze_backbone = bool(defaults["freeze_backbone"])
        freeze_warmup_epochs = int(defaults.get("freeze_warmup_epochs", 0) or 0)
        use_class_balance = bool(defaults["class_balance"])

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

        # Configuração multimodal: define quais colunas clínicas usar
        # Se desejar ativar por padrão, mude para "1"; aqui deixamos controlado por env ou "1" já que o user pediu
        clinical_features = ['age', 'education', 'nwbv', 'etiv', 'asf'] if os.getenv("USE_MULTIMODAL", "1") == "1" else None
        if clinical_features:
            print(f"[Multimodal] Integrando dados clínicos: {clinical_features}")

        lbl_col = 'age_normalized' if mode == 'regression' else 'Final_Group'  # Define coluna alvo conforme modo

        # Garante que cada linha tenha a lista de orientações (sem duplicar linhas)
        train_df = self._populate_orientation_paths(df[df['split']=='train'])
        val_df = self._populate_orientation_paths(df[df['split']=='validation'])
        test_df = self._populate_orientation_paths(df[df['split']=='test'])

        train_ds = MultiOrientMRIDataset(train_df, train_tf, self.dataset_dir.parent, 'original_path', lbl_col, clinical_features=clinical_features)
        val_ds = MultiOrientMRIDataset(val_df, val_tf, self.dataset_dir.parent, 'original_path', lbl_col, clinical_features=clinical_features)
        test_ds = MultiOrientMRIDataset(test_df, val_tf, self.dataset_dir.parent, 'original_path', lbl_col, clinical_features=clinical_features)  # Dataset de teste

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
        early_stop_patience = int(os.getenv("RESNET_PATIENCE", "7")) if mode == 'classification' else None  # Early stopping para class
        # Mixup desativado para Multimodal/Multistream por enquanto para garantir corretude
        use_mixup = False if clinical_features else (mode == 'classification' and mixup_alpha > 0)
        use_focal = mode == 'classification' and not use_mixup  # Focal loss para lidar com desbalanceamento (desligado se mixup ativo)
        focal_gamma = float(os.getenv("RESNET_FOCAL_GAMMA", 2.0))  # Gamma ajustável para focal loss
        use_ema = mode == 'classification'  # EMA para suavizar pesos na classificação
        ema_decay = 0.999

        # Configura sampler balanceado para evitar colapso de classe
        train_sampler = None
        shuffle_train = True
        if mode == 'classification' and 'Final_Group' in train_df.columns:
            # Calcula pesos por amostra para WeightedRandomSampler
            train_labels = train_df['Final_Group'].replace({'Nondemented': 0, 'Demented': 1}).fillna(0).astype(int).values
            class_counts = np.bincount(train_labels)
            # Evita divisão por zero
            class_counts = np.maximum(class_counts, 1)
            # Peso inversamente proporcional à frequência
            class_weights_sampler = 1.0 / class_counts
            sample_weights = class_weights_sampler[train_labels]
            sample_weights = torch.from_numpy(sample_weights).double()
            train_sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle_train = False  # WeightedRandomSampler é incompatível com shuffle
            print(f"[Anti-Colapso] WeightedRandomSampler ativado - Contagem por classe: {dict(zip(['Nondemented', 'Demented'], class_counts))}")

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, sampler=train_sampler)  # Loader de treino com balanceamento
        val_loader = DataLoader(val_ds, batch_size=batch_size)  # Loader de validação
        test_loader = DataLoader(test_ds, batch_size=batch_size)  # Loader de teste

        debug_batch_flag = os.getenv("DEBUG_BATCH", "0") == "1"
        debug_done = False

        # Seleção de Modelo (Multi-Stream)
        num_tab = len(clinical_features) if clinical_features else 0
        model = MultiOrientTabularFusionNet(
            backbone=backbone,
            mode=mode,
            num_tabular_features=num_tab,
            medicalnet_depth=defaults.get("medicalnet_depth", 18),
            pretrained=True,
            share_encoder=True,
            dropout=dropout_rate,
        )
        model = model.to(device)

        trainable_params_initial = None
        trainable_params_after_unfreeze = None
        did_unfreeze_backbone = False
        unfreeze_epoch_1based = None

        if freeze_backbone:
             # Congela os encoders (axl, cor, sag)
             for enc in [model.enc_axl, model.enc_cor, model.enc_sag]:
                 for p in enc.parameters():
                     p.requires_grad = False
             print("[INFO] Backbone encoders congelados. Apenas heads são treináveis.")
             if freeze_warmup_epochs > 0:
                 print(
                     f"[INFO] Agenda de ajuste: congelar encoders por {freeze_warmup_epochs} época(s) e depois liberar para ajuste fino."
                 )

        try:
            trainable_params_initial = int(sum(1 for p in model.parameters() if p.requires_grad))
        except Exception:
            trainable_params_initial = None

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
        # OneCycleLR is better for super-convergence and stability
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3, # 30% of time warming up
            div_factor=25.0, # Initial LR = max_lr / 25
            final_div_factor=10000.0, # Final LR = Initial LR / 10000
            anneal_strategy='cos'
        )
        
        # Calcula pesos de classe para loss (evita dupla compensação com sampler)
        loss_weights = None
        if mode == 'classification' and 'Final_Group' in df.columns:
            if train_sampler is None:
                counts = df[df['split'] == 'train']['Final_Group'].value_counts()
                n_nondemented = max(counts.get('Nondemented', counts.get(0, 0)), 1)
                n_demented = max(counts.get('Demented', counts.get(1, 0)), 1)
                total = n_nondemented + n_demented
                w0 = total / (2.0 * n_nondemented)
                w1 = total / (2.0 * n_demented)
                loss_weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)
                print(f"[Anti-Colapso] Sampler OFF. Pesos na Loss ATIVADOS: {w0:.3f}/{w1:.3f}")
            else:
                print("[Anti-Colapso] Sampler ON. Pesos na Loss DESATIVADOS para evitar viés.")

        label_smoothing = label_smoothing if mode == 'classification' else 0.0
        criterion = nn.MSELoss() if mode == 'regression' else nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            weight=loss_weights
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
        history_train_acc, history_val_acc, history_val_acc_raw = [], [], []  # Métricas de treino/validação (classificação)
        history_train_mae, history_val_mae = [], []  # MAE de treino/validação (regressão)
        val_metric_value = None  # Métrica principal em validação
        best_state, best_epoch = None, 0  # Controle early stopping
        best_bal_acc_raw = -float('inf') if mode == 'classification' else None
        best_bal_acc_adj = -float('inf') if mode == 'classification' else None
        best_val_acc_raw = -float('inf') if mode == 'classification' else None
        best_val_metric = float('inf') if mode == 'regression' else None
        no_improve = 0  # Contador para early stopping

        amp_available = hasattr(torch, "amp")
        scaler = None
        if amp_available and device.type != 'cpu':
            try:
                scaler = torch.amp.GradScaler(device_type=device.type)
            except TypeError:
                if device.type == 'cuda' and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
                    scaler = torch.cuda.amp.GradScaler()
                else:
                    scaler = None
        use_amp = scaler is not None

        for epoch in range(epochs):  # Loop de épocas
            if freeze_backbone and freeze_warmup_epochs > 0 and (not did_unfreeze_backbone) and epoch >= freeze_warmup_epochs:
                # Libera encoders após a fase de aquecimento congelada.
                for enc in [model.enc_axl, model.enc_cor, model.enc_sag]:
                    for p in enc.parameters():
                        p.requires_grad = True
                did_unfreeze_backbone = True
                unfreeze_epoch_1based = int(epoch + 1)
                try:
                    trainable_params_after_unfreeze = int(
                        sum(1 for p in model.parameters() if p.requires_grad)
                    )
                except Exception:
                    trainable_params_after_unfreeze = None
                print(
                    f"[INFO] Encoders liberados na época {unfreeze_epoch_1based}. "
                    f"Trainable params: {trainable_params_after_unfreeze}"
                )

            model.train()  # Coloca modelo em modo de treino
            running_loss = 0  # Acumulador de loss
            total_train = 0  # Contador de amostras de treino
            correct_train, total_train_cls = 0, 0  # Acertos/total para classificação
            mae_sum_train, total_train_reg = 0.0, 0  # Soma MAE/total para regressão

            for step, (batch_x, lbls) in enumerate(train_loader):  # Itera batches de treino
                axl = batch_x["axl"].to(device)
                cor = batch_x["cor"].to(device)
                sag = batch_x["sag"].to(device)
                clin = batch_x.get("clin")
                if clin is not None:
                    clin = clin.to(device)
                lbls = lbls.to(device)  # Move dados para CPU/GPU

                if debug_batch_flag and not debug_done and debug_batch is not None and epoch == 0 and step == 0:
                    debug_batch(axl, lbls, out_dir=os.path.join(self.output_dir, "debug"), prefix="train_batch0")
                    if debug_one_step is not None:
                        # Se for multistream (tem clinical_features), debug_one_step padrão pode quebrar
                        # pois espera model(imgs). Vamos pular ou adaptar se possível.
                        # Por segurança, executamos apenas se NÃO for multistream complexo
                        if not clinical_features:
                            debug_one_step(
                                model,
                                criterion,
                                optimizer,
                                imgs=axl, 
                                lbls=lbls,
                                clin=None,
                                mode=mode,
                                use_focal=use_focal,
                                focal_gamma=focal_gamma,
                                loss_weights=loss_weights,
                            )
                        else:
                            print("[DEBUG] Pulo debug_one_step pois arquitetura multistream requer inputs complexos.")
                    debug_done = True
                    raise SystemExit("DEBUG_BATCH done")

                optimizer.zero_grad()  # Zera gradientes
                try:
                    autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=use_amp)
                except TypeError:
                    if device.type == 'cuda' and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
                        autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)
                    else:
                        autocast_ctx = nullcontext()
                with autocast_ctx:
                    if mode == 'regression':  # Lógica para regressão
                        out = model(axl, cor, sag, clin)
                        preds_batch = out.squeeze()  # Ajusta forma da saída
                        loss = criterion(preds_batch, lbls)  # Calcula loss MSE
                        mae_sum_train += torch.abs(preds_batch - lbls).sum().item()  # Acumula MAE do batch
                        total_train_reg += lbls.size(0)  # Conta amostras de regressão
                    else:  # Lógica para classificação
                        if use_mixup:
                            # Mixup não suportado trivialmente com multistream neste código
                            # Fallback para sem mixup ou implementar mixup em cada ramo
                            # Por simplicidade, desativamos ou passamos direto
                            out = model(axl, cor, sag, clin)
                            loss = criterion(out, lbls.long())
                            preds_batch = out.argmax(dim=1)
                            correct_train += (preds_batch == lbls.long()).sum().item()
                        else:
                            out = model(axl, cor, sag, clin)
                            if use_focal:
                                loss = focal_loss(out, lbls.long(), gamma=focal_gamma, alpha=loss_weights)  # Focal loss com class weights
                            else:
                                loss = criterion(out, lbls.long())  # Calcula loss CE
                            preds_batch = out.argmax(dim=1)  # Classe predita
                            correct_train += (preds_batch == lbls.long()).sum().item()  # Acertos no batch
                        total_train_cls += lbls.size(0)  # Conta amostras de classe

                if scaler and use_amp:
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=defaults.get("grad_clip", 1.0))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=defaults.get("grad_clip", 1.0))
                    optimizer.step()
                
                # Step scheduler per batch for OneCycleLR
                scheduler.step()
                if ema: ema.update(model)  # Atualiza EMA
                running_loss += loss.item() * axl.size(0)  # Soma loss ponderada pelo batch
                total_train += axl.size(0)  # Atualiza total de amostras

            model.eval()  # Modo avaliação
            running_val = 0  # Acumulador de loss de validação
            preds_list, targs_list = [], []  # Listas para métricas de regressão
            correct_val, total_val = 0, 0  # Acertos/total na validação (classificação)
            val_preds_cls, val_targs_cls = [], []  # Armazenam predições e alvos para balanced accuracy
            if ema: ema.apply_shadow(model)  # Avalia com pesos suavizados
            with torch.no_grad():  # Sem gradientes
                for batch_x, lbls in val_loader:  # Itera batches de validação
                    axl = batch_x["axl"].to(device)
                    cor = batch_x["cor"].to(device)
                    sag = batch_x["sag"].to(device)
                    clin = batch_x.get("clin")
                    if clin is not None:
                        clin = clin.to(device)
                    lbls = lbls.to(device)
                    
                    out = model(axl, cor, sag, clin)  # Forward validação
                    if mode == 'regression':  # Métrica regressão
                        loss = criterion(out.squeeze(), lbls)  # Loss MSE validação
                        running_val += loss.item() * axl.size(0)  # Soma loss ponderada
                        preds_list.append(out.squeeze().cpu().numpy())  # Guarda predições
                        targs_list.append(lbls.cpu().numpy())  # Guarda rótulos verdadeiros
                    else:  # Métrica classificação
                        if use_focal:
                            loss = focal_loss(out, lbls.long(), gamma=focal_gamma, alpha=loss_weights)  # Focal na validação
                        else:
                            loss = criterion(out, lbls.long())
                        running_val += loss.item() * axl.size(0)  # Soma loss ponderada
                        preds = out.argmax(dim=1)  # Predições de classe
                        correct_val += (preds == lbls.long()).sum().item()  # Acertos no batch
                        total_val += lbls.size(0)  # Conta amostras
                        val_preds_cls.append(preds.cpu().numpy())
                        val_targs_cls.append(lbls.cpu().numpy())
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
                bal_acc_adj = -float('inf')
                if val_preds_cls:  # Balanced accuracy penaliza colapso para uma classe
                    y_true_val = np.concatenate(val_targs_cls)
                    y_pred_val = np.concatenate(val_preds_cls)
                    bal_acc_raw = balanced_accuracy_score(y_true_val, y_pred_val)
                    bal_acc_adj = 0.0 if len(np.unique(y_true_val)) < 2 else (2.0 * bal_acc_raw - 1.0)
                    val_metric_value = bal_acc_raw
                    val_acc_raw = correct_val / total_val if total_val else 0.0
                    history_val_acc.append(bal_acc_raw)  # Guarda balanced accuracy de validação
                    history_val_acc_raw.append(val_acc_raw)  # Guarda accuracy simples de validação
                    
                    # Monitoramento anti-colapso: verifica distribuição de predições
                    pred_counts = np.bincount(y_pred_val, minlength=2)
                    pred_ratio = pred_counts / max(pred_counts.sum(), 1)
                    collapse_threshold = 0.95  # Se >95% das predições são de uma classe, alerta colapso
                    if pred_ratio.max() > collapse_threshold:
                        dominant_class = 'Nondemented' if pred_ratio[0] > pred_ratio[1] else 'Demented'
                        print(f"[ALERTA COLAPSO] Época {epoch+1}: {pred_ratio.max()*100:.1f}% das predições são '{dominant_class}'")
                else:
                    val_metric_value = 0.0
                    history_val_acc.append(val_metric_value)
                    history_val_acc_raw.append(0.0)

            # Log de progresso com métricas relevantes
            if mode == 'classification':
                bal_acc_str = f", Bal Acc {val_metric_value:.4f}" if val_metric_value else ""
                print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}{bal_acc_str}")
            else:
                print(f"Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")  # Log de progresso
            history.append((train_loss, val_loss))  # Armazena histórico simples

            # Early stopping baseado em balanced accuracy (class) ou val loss (reg)
            improved = False
            if mode == 'classification' and val_metric_value is not None:
                metric_es = bal_acc_adj if mode == 'classification' else -float('inf')
                if metric_es > best_bal_acc_adj:
                    improved = True
                    best_bal_acc_adj = metric_es
                    best_bal_acc_raw = history_val_acc[-1] if history_val_acc else 0.0
                    best_val_acc_raw = history_val_acc_raw[-1] if history_val_acc_raw else 0.0
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

            curves_name = f"{backbone}_{mode}_learning_curves.png"  # Nome do arquivo de curvas
            fig.savefig(self.output_dir / curves_name, dpi=300, bbox_inches='tight')  # Salva curvas
            if not headless:
                try: self._show_plot_window("Resultados", fig)
                except Exception: pass

        torch.save(model.state_dict(), best_checkpoint_path)  # Salva pesos do melhor estado
        # Mantém nome legado para compatibilidade com fluxos existentes (ex: RL)
        torch.save(model.state_dict(), legacy_checkpoint_path)

        if mode == 'regression' and age_scaler is not None:  # Pós-processamento específico de regressão
            model.eval()  # Modo avaliação
            all_train_preds_norm, all_train_true_norm = [], []  # Armazenam predições/verdadeiros normalizados (treino)
            all_val_preds_norm, all_val_true_norm = [], []  # Armazenam predições/verdadeiros normalizados (val)
            all_test_preds_norm, all_test_true_norm = [], []  # Armazenam predições/verdadeiros normalizados (teste)

            with torch.no_grad():  # Sem gradientes
                for batch_x, ages in train_loader:  # Loop treino
                    axl = batch_x["axl"].to(device)
                    cor = batch_x["cor"].to(device)
                    sag = batch_x["sag"].to(device)
                    clin = batch_x.get("clin")
                    if clin is not None: clin = clin.to(device)

                    preds = model(axl, cor, sag, clin).squeeze()  # Predições normalizadas
                    all_train_preds_norm.extend(np.atleast_1d(preds.cpu().numpy()))  # Guarda predições
                    all_train_true_norm.extend(np.atleast_1d(ages.numpy()))  # Guarda idades reais

                for batch_x, ages in val_loader:  # Loop validação
                    axl = batch_x["axl"].to(device)
                    cor = batch_x["cor"].to(device)
                    sag = batch_x["sag"].to(device)
                    clin = batch_x.get("clin")
                    if clin is not None: clin = clin.to(device)

                    preds = model(axl, cor, sag, clin).squeeze()  # Predições
                    all_val_preds_norm.extend(np.atleast_1d(preds.cpu().numpy()))  # Guarda predições
                    all_val_true_norm.extend(np.atleast_1d(ages.numpy()))  # Guarda idades reais

                for batch_x, ages in test_loader:  # Loop teste
                    axl = batch_x["axl"].to(device)
                    cor = batch_x["cor"].to(device)
                    sag = batch_x["sag"].to(device)
                    clin = batch_x.get("clin")
                    if clin is not None: clin = clin.to(device)

                    preds = model(axl, cor, sag, clin).squeeze()  # Predições
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
            fig_scatter.savefig(self.output_dir / f"{backbone}_regression_scatter.png", dpi=300, bbox_inches='tight')  # Salva gráfico
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
                for batch_x, lbls in test_loader:  # Loop de teste
                    axl = batch_x["axl"].to(device)
                    cor = batch_x["cor"].to(device)
                    sag = batch_x["sag"].to(device)
                    clin = batch_x.get("clin")
                    if clin is not None: clin = clin.to(device)
                    lbls = lbls.to(device)  # Move dados
                    out = model(axl, cor, sag, clin)  # Forward
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
                if hasattr(self, "plot_confusion_matrix"):
                    fig_cm = Figure(figsize=(6, 5))  # Figura da matriz
                    ax = fig_cm.add_subplot(1, 1, 1)  # Eixo único
                    self.plot_confusion_matrix(ax, test_cm, ['0', '1'], "Teste")  # Plota matriz
                    fig_cm.tight_layout()  # Ajusta layout
                    fig_cm.savefig(self.output_dir / f"confusion_{backbone}_{mode}.png", dpi=300, bbox_inches='tight')  # Salva imagem
                    if not headless:
                        try: self._show_plot_window("Matriz de Confusão - Teste", fig_cm)
                        except Exception: pass

        learning_curves = {  # Dicionário com curvas de aprendizagem
            'train_loss': history_train_loss,
            'val_loss': history_val_loss,
        }
        if mode == 'classification':
            learning_curves['train_acc'] = history_train_acc  # Acurácia treino
            learning_curves['val_acc'] = history_val_acc  # Alias: balanced accuracy val
            learning_curves['val_acc_balanced'] = history_val_acc  # Balanced accuracy val
            learning_curves['val_acc_raw'] = history_val_acc_raw  # Accuracy simples val
        else:
            learning_curves['train_mae'] = history_train_mae_denorm  # MAE treino desnormalizado
            learning_curves['val_mae'] = history_val_mae_denorm  # MAE val desnormalizado

        training_time = time.time() - start_time  # Tempo total do processo

        exp_payload = {  # Payload para histórico do experimento
            'model': f'{backbone}_{mode}',
            'scenario': deep_scenario_label,
            'seed': int(seed),
            'split_csv_sha256': split_csv_sha256,
            'use_multimodal': bool(clinical_features),
            'clinical_features': clinical_features,
            'pretrained': True,
            'freeze_backbone_initial': bool(freeze_backbone),
            'freeze_warmup_epochs': int(freeze_warmup_epochs),
            'unfreeze_epoch': unfreeze_epoch_1based,
            'trainable_params_initial': trainable_params_initial,
            'trainable_params_after_unfreeze': trainable_params_after_unfreeze,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': lr,
            'train_loss': float(history_train_loss[-1]) if history_train_loss else None,
            'val_loss': float(history_val_loss[-1]) if history_val_loss else None,
            'learning_curves': learning_curves,
            'training_time_seconds': float(training_time),
            'best_checkpoint': best_checkpoint_path.name,
            'legacy_checkpoint': legacy_checkpoint_path.name,
            'best_params': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': lr,
            }
        }
        if mode == 'classification':
            if history_train_acc:
                exp_payload['train_accuracy'] = float(history_train_acc[-1])  # Acurácia final de treino
            if history_val_acc_raw:
                exp_payload['val_accuracy_raw'] = float(history_val_acc_raw[-1])  # Accuracy simples (última época)
            if history_val_acc:
                exp_payload['val_balanced_accuracy'] = float(history_val_acc[-1])  # Balanced accuracy (última época)
            if best_bal_acc_raw is not None and best_bal_acc_raw != -float('inf'):
                exp_payload['best_val_balanced_accuracy'] = float(best_bal_acc_raw)
                exp_payload['best_val_balanced_accuracy_adj'] = float(best_bal_acc_adj if best_bal_acc_adj is not None else 0.0)
                exp_payload['best_val_accuracy'] = float(best_val_acc_raw if best_val_acc_raw is not None else 0.0)
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
                        for batch_x, lbls in loader:
                            axl = batch_x["axl"].to(device)
                            cor = batch_x["cor"].to(device)
                            sag = batch_x["sag"].to(device)
                            clin = batch_x.get("clin")
                            if clin is not None: clin = clin.to(device)

                            # Usa método público de extração se disponível
                            if hasattr(model, "extract_features"):
                                feats = model.extract_features(axl, cor, sag, clin)
                            else:
                                # Fallback legado (improvavel cair aqui se for o multistream_model atual)
                                f_axl = model._encode_one(model.enc_axl, model.proj_axl, axl)
                                f_cor = model._encode_one(model.enc_cor, model.proj_cor, cor)
                                f_sag = model._encode_one(model.enc_sag, model.proj_sag, sag)
                                feats_list = [f_axl, f_cor, f_sag]
                                if clin is not None and model.tab_mlp:
                                    feats_list.append(model.tab_mlp(clin))
                                feats = torch.cat(feats_list, dim=1)
                            
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
                    out_path = self.output_dir / f"{backbone}_embeddings_{mode}_{split_name}.csv"
                    df_emb.to_csv(out_path, index=False)

                _export_embeddings('train', train_ds)
                _export_embeddings('val', val_ds)
                _export_embeddings('test', test_ds)
            except Exception as e:
                print(f"Falha ao exportar embeddings: {e}")
        if hasattr(self, "_save_experiment"):
            self._save_experiment(exp_payload)  # Salva experimento no histórico
        else:
            print("[INFO] _save_experiment indisponível no modo headless; pulando registro.")

        metric_msg = "Test Acc" if mode == 'classification' else "Test MAE"  # Label de métrica principal
        if val_metric_value is not None:
            fmt = "{:.2%}" if mode == 'classification' else "{:.4f}"  # Formatação por modo
            metric_msg += f": {fmt.format(val_metric_value)}"  # Anexa valor final
        if mode == 'classification':
            metric_msg += f"\nBest Val Balanced Acc: {best_bal_acc_raw:.2%} (ajustada {best_bal_acc_adj:.3f}) @ epoch {best_epoch}"
        try:
            if not headless:
                messagebox.showinfo(f"{backbone}", f"Treino concluído. {metric_msg}")
            else:
                print(f"[{backbone}] {metric_msg}")
        except Exception:
            print(f"[{backbone}] {metric_msg}")

    def refine_efficientnet_with_rl(self, episodes=8, horizon=4, micro_epochs=1,
                                    train_subset=120, val_subset=80):  # Refinamento via RL
        if not (SKLEARN_AVAILABLE and TORCH_AVAILABLE and PANDAS_AVAILABLE):
            messagebox.showerror("Dependência ausente",
                                 "PyTorch, scikit-learn e pandas são necessários para o refinamento com RL.")
            return

        df_path = self.output_dir / "exam_level_dataset_split.csv"
        if not df_path.exists():
            messagebox.showwarning("Aviso", "Crie o dataset (Criar Dataset) antes de rodar o RL.")
            return

        base_ckpt = self.output_dir / "efficientnet_classification.pth"
        if not base_ckpt.exists():
            messagebox.showwarning("Aviso", "Treine a EfficientNet de classificação antes de refinar com RL.")
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
            MultiOrientMRIDataset(df_train_small, train_tf, self.dataset_dir.parent, 'original_path', 'Final_Group'),
            batch_size=batch_small, shuffle=True
        )
        val_loader_small = DataLoader(
            MultiOrientMRIDataset(df_val_small, val_tf, self.dataset_dir.parent, 'original_path', 'Final_Group'),
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
            MultiOrientMRIDataset(df_val, val_tf, self.dataset_dir.parent, 'original_path', 'Final_Group'),
            batch_size=16, shuffle=False
        )
        test_loader_full = DataLoader(
            MultiOrientMRIDataset(df_test, val_tf, self.dataset_dir.parent, 'original_path', 'Final_Group'),
            batch_size=16, shuffle=False
        )

        val_metrics = evaluate_full_model(eval_model, val_loader_full, device)
        test_metrics = evaluate_full_model(eval_model, test_loader_full, device)

        def _collect_preds(loader):
            eval_model.eval()
            y_true, y_pred = [], []
            with torch.no_grad():
                for batch_x, lbls in loader:
                    axl = batch_x["axl"].to(device)
                    cor = batch_x["cor"].to(device)
                    sag = batch_x["sag"].to(device)
                    clin = batch_x.get("clin")
                    if clin is not None: clin = clin.to(device)
                    lbls = lbls.to(device)

                    out = eval_model(axl, cor, sag, clin)
                    preds = out.argmax(dim=1)
                    y_true.append(lbls.cpu().numpy())
                    y_pred.append(preds.cpu().numpy())
            if not y_true:
                return np.array([]), np.array([])
            return np.concatenate(y_true), np.concatenate(y_pred)

        y_val, y_val_pred = _collect_preds(val_loader_full)
        y_test, y_test_pred = _collect_preds(test_loader_full)
        val_cm = confusion_matrix(y_val, y_val_pred) if y_val.size else None
        test_cm = confusion_matrix(y_test, y_test_pred) if y_test.size else None

        best_model_path = self.output_dir / "efficientnet_classification_rl_best.pth"
        torch.save(best_state, best_model_path)
        policy_path = self.output_dir / "rl_policy_efficientnet.pth"
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
            'model': 'EfficientNet_classification_RL',
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
            curves_path = self.output_dir / "efficientnet_classification_rl_learning_curves.png"
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
            f"Curvas: efficientnet_classification_rl_learning_curves.png"
        )

        # Treino final robusto usando hiperparâmetros encontrados pelo RL no dataset completo
        if best_hparams:
            try:
                print("Treinando EfficientNet final com hiperparâmetros do RL (split completo)...")
                self._train_pytorch_model(mode='classification', hparams=best_hparams)
            except Exception as e:
                print(f"Falha ao treinar modelo final com hparams do RL: {e}")

        messagebox.showinfo("EfficientNet + RL", msg)

    # Alias para compatibilidade
    def refine_densenet_with_rl(self, episodes=8, horizon=4, micro_epochs=1,
                                train_subset=120, val_subset=80):
        return self.refine_efficientnet_with_rl(episodes=episodes, horizon=horizon, micro_epochs=micro_epochs,
                                                train_subset=train_subset, val_subset=val_subset)
