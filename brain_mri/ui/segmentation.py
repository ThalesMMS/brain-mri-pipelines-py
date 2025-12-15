import re  # Expressões regulares para parsing de texto
import shutil  # Operações de alto nível com arquivos e diretórios
from pathlib import Path  # Manipulação de caminhos
import tkinter as tk  # Interface gráfica
from tkinter import messagebox  # Caixas de diálogo

import numpy as np  # Operações numéricas
try:
    import nibabel as nib  # Leitura de arquivos NIfTI
    NIB_AVAILABLE = True
except ImportError:
    nib = None
    NIB_AVAILABLE = False
try:
    import pandas as pd  # Manipulação de dados
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
try:
    from PIL import Image  # Manipulação de imagens via Pillow
except ImportError:
    Image = None

from ..utils.image_utils import ImageUtils  # Utilitário de imagens


class SegmentationMixin:  # Métodos relacionados à segmentação e gestão de viabilidade
    def apply_grow_region(self):  # Executa segmentação por crescimento de região e atualiza estados
        if self.current_image is None: return  # Sai se nenhuma imagem estiver carregada
        try:
            mask, _, _, roi_mask = ImageUtils.grow_region(self.original_image)  # Gera máscara segmentada e ROI
            self.roi_mask = roi_mask  # Armazena ROI usada na segmentação
            self.original_segmented = mask  # Guarda máscara original resultante
            self.current_segmented = mask.copy()  # Copia máscara para exibição/edição

            if self.image_path:  # Se existe caminho atual
                self.segmented_images[self.image_path] = mask  # Guarda máscara em cache de segmentações
                self._save_segmentation_results(mask)  # Salva arquivos e atualiza CSV

            self.display_image()  # Re-renderiza imagens com segmentação

            desc = ImageUtils.calculate_descriptors(mask)  # Calcula descritores morfológicos da máscara
            self.info_label.config(text=f"Segmentado.\nÁrea: {desc['area']:.0f} px\nCirc: {desc['circularity']:.3f}")  # Exibe métricas básicas

        except Exception as e:
            messagebox.showerror("Erro na Segmentação", str(e))  # Exibe mensagem de erro de segmentação

    def _save_segmentation_results(self, mask):  # Salva máscara em PNG e registra descritores no CSV
        """Salva imagem PNG e atualiza CSV."""  # Docstring mantém descrição da função
        if Image is None:
            raise ImportError("O módulo Pillow é necessário para salvar segmentações. Instale com 'pip install pillow'.")
        fname = Path(self.image_path).stem.replace('.nii', '')  # Extrai nome base do arquivo atual
        out_png = self.output_dir / f"{fname}_segmented.png"  # Define caminho de saída do PNG segmentado
        Image.fromarray((mask * 255).astype(np.uint8)).save(out_png)  # Converte máscara binária e salva como PNG

        desc = ImageUtils.calculate_descriptors(mask)  # Recalcula descritores para registro
        self._update_csv(fname, desc, f"output/{out_png.name}")  # Atualiza CSV com caminhos e métricas

    def _update_csv(self, mri_id, descriptors, seg_path):  # Atualiza/insere linha no CSV de descritores
        if not PANDAS_AVAILABLE:
            raise ImportError("O módulo pandas é necessário para atualizar o CSV. Instale com 'pip install pandas'.")
        mri_clean = mri_id.replace('_axl', '')  # Remove sufixo de orientação do nome
        subj_id = mri_clean.split('_MR')[0] if '_MR' in mri_clean else mri_clean  # Extrai ID do sujeito

        data = {  # Monta dict com todos os campos do registro
            'MRI_ID': mri_clean, 'Subject_ID': subj_id, 'viable': True,
            'segmented_path': seg_path,
            'ventricle_area': descriptors['area'],
            'ventricle_perimeter': descriptors['perimeter'],
            'ventricle_circularity': descriptors['circularity'],
            'ventricle_eccentricity': descriptors['eccentricity'],
            'ventricle_solidity': descriptors['solidity'],
            'ventricle_major_axis_length': descriptors['major_axis_length'],
            'ventricle_minor_axis_length': descriptors['minor_axis_length']
        }

        df = pd.read_csv(self.descriptors_csv) if self.descriptors_csv.exists() else pd.DataFrame()  # Carrega CSV existente ou cria vazio
        if df.empty and not self.descriptors_csv.exists():  # Se não existe arquivo, cria colunas
            df = pd.DataFrame(columns=data.keys())
        if 'viable' not in df.columns:
            df['viable'] = True  # Garante coluna de viabilidade
        for key in data.keys():
            if key not in df.columns:
                df[key] = np.nan  # Adiciona colunas faltantes com NaN

        if 'MRI_ID' in df.columns and mri_clean in df['MRI_ID'].values:
            for k, v in data.items():
                df.loc[df['MRI_ID'] == mri_clean, k] = v  # Atualiza linha correspondente
        else:
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)  # Anexa novo registro

        df = self._calc_longitudinal(df)  # Calcula variações longitudinais
        df.to_csv(self.descriptors_csv, index=False)  # Persiste CSV atualizado

    def _set_viable_flag(self, mri_id: str, subject_id: str, viable: bool):  # Marca exame como viável/inviável no CSV
        if not PANDAS_AVAILABLE:
            raise ImportError("O módulo pandas é necessário para atualizar o CSV. Instale com 'pip install pandas'.")
        if self.descriptors_csv.exists():  # Verifica existência do arquivo
            df = pd.read_csv(self.descriptors_csv)  # Lê CSV existente
        else:
            df = pd.DataFrame(columns=['MRI_ID', 'Subject_ID', 'viable'])  # Cria DataFrame vazio com colunas básicas

        if 'viable' not in df.columns:
            df['viable'] = True  # Garante coluna de viabilidade

        if 'MRI_ID' in df.columns and mri_id in df['MRI_ID'].values:
            df.loc[df['MRI_ID'] == mri_id, 'viable'] = viable  # Atualiza flag para exame existente
        else:
            df = pd.concat([df, pd.DataFrame([{  # Adiciona novo registro se não existir
                'MRI_ID': mri_id,
                'Subject_ID': subject_id,
                'viable': viable
            }])], ignore_index=True)

        df = self._calc_longitudinal(df)  # Atualiza métricas longitudinais
        df.to_csv(self.descriptors_csv, index=False)  # Salva CSV com nova flag

    def _calc_longitudinal(self, df: pd.DataFrame) -> pd.DataFrame:  # Calcula variações longitudinais entre exames
        if not PANDAS_AVAILABLE:
            raise ImportError("O módulo pandas é necessário para cálculos longitudinais. Instale com 'pip install pandas'.")
        if df.empty:
            return df  # Retorna se não houver dados

        if 'viable' not in df.columns:
            df['viable'] = True  # Garante coluna de viabilidade

        required_cols = [  # Lista de colunas de mudança longitudinal
            'area_change', 'area_change_percent', 'perimeter_change', 'circularity_change',
            'eccentricity_change', 'solidity_change', 'major_axis_change',
            'minor_axis_change', 'visit_number'
        ]
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan  # Adiciona colunas faltantes com NaN

        for idx, row in df.iterrows():  # Itera linhas para extrair número da visita
            m = re.search(r'MR(\d+)', str(row.get('MRI_ID', '')))  # Captura número após MR
            if m:
                df.at[idx, 'visit_number'] = int(m.group(1))  # Define número da visita

        change_map = {  # Mapeia colunas originais para colunas de mudança
            'ventricle_area': 'area_change',
            'ventricle_perimeter': 'perimeter_change',
            'ventricle_circularity': 'circularity_change',
            'ventricle_eccentricity': 'eccentricity_change',
            'ventricle_solidity': 'solidity_change',
            'ventricle_major_axis_length': 'major_axis_change',
            'ventricle_minor_axis_length': 'minor_axis_change'
        }

        if 'Subject_ID' not in df.columns:
            return df  # Sem ID de sujeito não há longitudinal

        for sid in df['Subject_ID'].dropna().unique():  # Percorre cada paciente
            subj_df = df[df['Subject_ID'] == sid].sort_values('visit_number')  # Ordena visitas por número
            prev_idx = None  # Índice da visita anterior viável
            for idx in subj_df.index:  # Itera visitas do paciente
                if prev_idx is None:
                    prev_idx = idx  # Define primeira visita como referência
                    continue
                if not (bool(df.at[idx, 'viable']) and bool(df.at[prev_idx, 'viable'])):
                    prev_idx = idx  # Pula se alguma visita não for viável
                    continue
                for src_col, dst_col in change_map.items():  # Calcula diferenças para cada descritor
                    if src_col in df.columns:
                        prev_val = df.at[prev_idx, src_col]
                        cur_val = df.at[idx, src_col]
                        if pd.notna(prev_val) and pd.notna(cur_val):
                            df.at[idx, dst_col] = cur_val - prev_val  # Armazena variação absoluta
                prev_area = df.at[prev_idx, 'ventricle_area'] if 'ventricle_area' in df.columns else np.nan  # Área anterior
                cur_area = df.at[idx, 'ventricle_area'] if 'ventricle_area' in df.columns else np.nan  # Área atual
                if pd.notna(prev_area) and pd.notna(cur_area) and prev_area:
                    df.at[idx, 'area_change_percent'] = ((cur_area - prev_area) / prev_area) * 100  # Variação percentual da área
                prev_idx = idx  # Atualiza referência para próxima iteração

        return df  # Retorna DataFrame com colunas de mudança preenchidas

    def process_all_images(self):  # Processa todas as imagens aplicando segmentação e salvando resultados
        if not messagebox.askyesno("Processar Tudo", f"Processar {len(self.image_list)} imagens?"): return  # Confirmação do usuário
        if not NIB_AVAILABLE:
            messagebox.showerror("Dependência ausente", "O módulo nibabel é necessário para ler arquivos NIfTI.\nInstale com 'pip install nibabel'.")
            return

        processed = 0  # Contador de imagens processadas
        for img_path in self.image_list:  # Itera sobre todas as imagens disponíveis
            try:
                self.image_path = img_path  # Define caminho atual
                data = np.squeeze(nib.load(img_path).get_fdata())  # Carrega volume NIfTI e remove dimensões unitárias
                if data.ndim == 3:
                    data = data[:, :, 0]  # Seleciona primeira fatia se for 3D
                img = ImageUtils.normalize_array(data)  # Normaliza imagem
                mask, _, _, _ = ImageUtils.grow_region(img)  # Segmenta ventrículos
                self._save_segmentation_results(mask)  # Salva máscara e descritores
                processed += 1  # Incrementa contador
                self.info_label.config(text=f"Processando: {processed}/{len(self.image_list)}")  # Atualiza status na UI
                self.root.update()  # Força refresh da interface
            except Exception as e:
                print(f"Falha em {img_path}: {e}")  # Log simples de falha

        messagebox.showinfo("Fim", f"Processadas {processed} imagens.")  # Notifica conclusão
        self.load_existing_segmentations()  # Recarrega segmentações salvas

    def load_existing_segmentations(self):  # Carrega máscaras pré-existentes da pasta de saída
        """Carrega máscaras já salvas na pasta output."""  # Docstring mantém descrição
        if Image is None:
            # Evita impedir o carregamento da UI; apenas informa a ausência da dependência
            try:
                messagebox.showwarning("Dependência ausente", "O módulo 'Pillow' é necessário para carregar segmentações salvas.\nInstale com 'pip install pillow'.")
            except Exception:
                pass
            return
        cnt = 0  # Contador de máscaras carregadas
        for f in self.output_dir.glob("*_segmented.png"):  # Itera sobre arquivos PNG segmentados
            orig_name = f.name.replace("_segmented.png", "")  # Extrai nome base para buscar imagem original
            for path in self.image_list:
                if orig_name in path:
                    arr = np.array(Image.open(f))  # Lê PNG para array
                    self.segmented_images[path] = (arr > 0).astype(np.uint8)  # Salva máscara em cache
                    cnt += 1  # Incrementa contador de máscaras carregadas
                    break
        print(f"Segmentações carregadas: {cnt}")  # Loga total carregado

    def mark_not_viable(self):  # Marca imagem como inviável e move para pasta dedicada
        if not self.image_path: return  # Só continua se houver imagem atual
        if not messagebox.askyesno("Confirmar", "Marcar como inviável e mover arquivo?"): return  # Confirma ação

        dst = self.not_viable_dir / Path(self.image_path).name  # Caminho de destino na pasta de inviáveis
        mri_clean = Path(self.image_path).stem.replace('.nii', '').replace('_axl', '')  # Nome base sem extensões
        if self.image_path.endswith('.nii.gz'):
            mri_clean = Path(self.image_path).name.replace('.nii.gz', '').replace('_axl', '')  # Ajusta para .nii.gz
        subject_id = mri_clean.split('_MR')[0] if '_MR' in mri_clean else mri_clean  # Extrai ID do paciente
        try:
            shutil.move(self.image_path, dst)  # Move arquivo para pasta de inviáveis
            self._set_viable_flag(mri_clean, subject_id, False)  # Atualiza flag de viabilidade
            self.image_list.remove(self.image_path)  # Remove caminho da lista de imagens
            if self.image_path in self.segmented_images: del self.segmented_images[self.image_path]  # Remove máscara cacheada
            self.load_image_at_index(self.current_image_index)  # Recarrega imagem atual (ou próxima disponível)
        except Exception as e:
            messagebox.showerror("Erro", str(e))  # Exibe erro em caso de falha

    def view_not_viable_images(self):  # Abre janela para restaurar imagens marcadas como inviáveis
        win = tk.Toplevel(self.root)  # Cria nova janela
        win.title("Restaurar Imagens")  # Define título
        lb = tk.Listbox(win, width=50); lb.pack(padx=10, pady=10)  # Lista arquivos inviáveis
        files = list(self.not_viable_dir.glob("*"))  # Lista arquivos na pasta inviáveis
        for f in files: lb.insert(tk.END, f.name)  # Popula listbox com nomes

        def restore():  # Função interna para restaurar item selecionado
            sel = lb.curselection()  # Captura seleção
            if not sel: return  # Sai se nada selecionado
            f = files[sel[0]]  # Arquivo escolhido
            shutil.move(str(f), str(self.dataset_dir / f.name))  # Move arquivo de volta ao dataset
            mri_clean = f.name.replace('.nii.gz', '').replace('.nii', '').replace('_axl', '')  # Limpa nome
            subject_id = mri_clean.split('_MR')[0] if '_MR' in mri_clean else mri_clean  # Extrai ID do sujeito
            self._set_viable_flag(mri_clean, subject_id, True)  # Marca como viável novamente
            win.destroy()  # Fecha janela
            self.auto_load_first_image()  # Recarrega lista de imagens

        tk.Button(win, text="Restaurar Selecionada", command=restore).pack(pady=5)  # Botão para restaurar
