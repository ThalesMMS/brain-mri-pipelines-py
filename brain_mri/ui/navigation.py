import tkinter as tk  # Interface gráfica com Tkinter base
from tkinter import messagebox  # Caixa de diálogo do Tkinter
from pathlib import Path  # Manipulação de caminhos

import numpy as np  # Operações numéricas
try:
    import cv2  # OpenCV para processamento de imagens
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False
try:
    import nibabel as nib  # Leitura de arquivos NIfTI
    NIB_AVAILABLE = True
except ImportError:
    nib = None
    NIB_AVAILABLE = False
try:
    from PIL import Image  # Manipulação de imagens
except ImportError:
    Image = None
from matplotlib.patches import Rectangle  # Desenho de retângulos em plots

from ..utils.image_utils import ImageUtils  # Utilitário de imagens


class NavigationMixin:  # Métodos de navegação e exibição de imagens
    def auto_load_first_image(self):  # Popula lista de imagens e abre a primeira
        files = sorted(self.dataset_dir.glob("*.nii*"))  # Lista arquivos NIfTI ordenados
        if files:  # Se houver arquivos
            self.image_list = [str(f) for f in files]  # Guarda caminhos em lista
            self.current_image_index = 0  # Define índice inicial
            self.load_image_at_index(0)  # Carrega primeira imagem

    def load_image_at_index(self, idx):  # Carrega imagem e máscara associada
        if 0 <= idx < len(self.image_list):  # Garante índice válido
            try:
                path = self.image_list[idx]  # Obtém caminho da imagem pelo índice
                self.image_path = path  # Armazena caminho atual

                if path.endswith(('.nii', '.gz')):  # Se arquivo NIfTI
                    if not NIB_AVAILABLE:
                        # Evita popup bloqueante: apenas informa no painel inferior
                        self.info_label.config(text="Instale 'nibabel' para carregar arquivos NIfTI (pip install nibabel).")
                        return
                    nii = nib.load(path)  # Carrega volume NIfTI
                    data = np.squeeze(nii.get_fdata())  # Extrai dados e remove dimensões unitárias
                    if data.ndim == 3:  # Se volume 3D
                        data = data[:, :, 0]  # Seleciona primeira fatia
                    img_norm = ImageUtils.normalize_array(data)  # Normaliza intensidades para 0-1
                else:  # Caso seja imagem convencional
                    if Image is None:
                        self.info_label.config(text="Instale 'Pillow' para carregar imagens comuns (pip install pillow).")
                        return
                    pil_img = Image.open(path).convert('L')  # Abre em escala de cinza
                    img_norm = np.array(pil_img) / 255.0  # Normaliza pixels para 0-1

                self.original_image = img_norm  # Guarda imagem original normalizada
                self.current_image = img_norm.copy()  # Copia para imagem atual exibida
                self.zoom_level = 1.0  # Reseta nível de zoom

                loaded_mask = self.segmented_images.get(path)  # Recupera máscara já segmentada se existir
                self.original_segmented = loaded_mask  # Salva máscara original
                self.current_segmented = loaded_mask.copy() if loaded_mask is not None else None  # Copia máscara para edição/exibição

                self.display_image()  # Atualiza visualização
                self.info_label.config(text=f"Img {idx+1}/{len(self.image_list)}: {Path(path).name}")  # Atualiza label de status
            except Exception as e:
                messagebox.showerror("Erro", f"Falha ao carregar: {e}")  # Exibe erro de carregamento

    def prev_image(self):  # Move índice de imagem para trás
        if self.current_image_index > 0:  # Garante que não ultrapasse início
            self.current_image_index -= 1  # Decrementa índice
            self.load_image_at_index(self.current_image_index)  # Carrega nova imagem

    def next_image(self):  # Move índice de imagem para frente
        if self.current_image_index < len(self.image_list) - 1:  # Verifica limite superior
            self.current_image_index += 1  # Incrementa índice
            self.load_image_at_index(self.current_image_index)  # Carrega nova imagem

    def _zoom(self, factor):  # Multiplica nível de zoom pelo fator fornecido
        if self.original_image is None: return  # Não faz nada se nenhuma imagem carregada
        self.zoom_level *= factor  # Atualiza fator de zoom
        self._apply_zoom()  # Aplica transformações de zoom

    def reset_zoom(self):  # Restaura imagem e máscara originais
        self.zoom_level = 1.0  # Reseta fator de zoom
        self.current_image = self.original_image.copy()  # Restaura imagem original
        if self.original_segmented is not None:  # Se há máscara
            self.current_segmented = self.original_segmented.copy()  # Restaura máscara original
        self.display_image()  # Atualiza exibição

    def _apply_zoom(self):  # Ajusta conteúdos conforme nível de zoom
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV (cv2) é necessário para aplicar zoom. Instale com 'pip install opencv-python'.")
        h, w = self.original_image.shape  # Obtém dimensões originais
        new_h, new_w = int(h * self.zoom_level), int(w * self.zoom_level)  # Calcula novas dimensões

        resized = cv2.resize(self.original_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)  # Redimensiona imagem com interpolação linear

        if self.zoom_level > 1.0:  # Crop center
            y, x = (new_h - h) // 2, (new_w - w) // 2  # Calcula recorte central
            self.current_image = resized[y:y+h, x:x+w]  # Recorta região central para manter tamanho
        else:  # Pad
            y, x = (h - new_h) // 2, (w - new_w) // 2  # Calcula padding necessário
            self.current_image = np.pad(resized, ((y, h-new_h-y), (x, w-new_w-x)), mode='constant')  # Preenche com zeros

        if self.original_segmented is not None:  # Ajusta máscara se estiver disponível
            mask = self.original_segmented.astype(np.float32)  # Converte máscara para float
            resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)  # Redimensiona máscara mantendo rótulos
            if self.zoom_level > 1.0:
                y, x = (new_h - h) // 2, (new_w - w) // 2  # Calcula recorte central da máscara
                mask_zoom = resized_mask[y:y+h, x:x+w]  # Recorta área central
            else:
                y, x = (h - new_h) // 2, (w - new_w) // 2  # Calcula padding para máscara
                mask_zoom = np.pad(resized_mask, ((y, h-new_h-y), (x, w-new_w-x)), mode='constant')  # Aplica padding
            self.current_segmented = (mask_zoom > 0.5).astype(np.uint8)  # Binariza máscara redimensionada

        self.display_image()  # Atualiza exibição após aplicar zoom

    def display_image(self):  # Desenha figuras na interface
        if self.current_image is None: return  # Sai se não há imagem carregada
        self.ax_orig.clear(); self.ax_seg.clear()  # Limpa eixos antes de redesenhar

        disp_img = np.rot90(self.current_image, k=1)  # Rotaciona imagem para orientação desejada
        self.ax_orig.imshow(disp_img, cmap='gray')  # Exibe imagem original em escala de cinza
        self.ax_orig.set_title(f"Original ({self.zoom_level:.1f}x)")  # Título com zoom atual
        self.ax_orig.axis('off')  # Oculta eixos

        if self.current_segmented is not None:  # Se existe máscara de segmentação
            mask_disp = self.current_segmented.astype(float)  # Converte máscara para float para visualização
            if self.original_image.shape != self.current_image.shape:  # Ajusta máscara se dimensões diferem
                h, w = self.original_image.shape  # Dimensões originais
                nh, nw = int(h * self.zoom_level), int(w * self.zoom_level)  # Dimensões redimensionadas
                resized_mask = cv2.resize(mask_disp, (nw, nh), interpolation=cv2.INTER_NEAREST)  # Redimensiona máscara
                if self.zoom_level > 1:
                    y, x = (nh - h) // 2, (nw - w) // 2  # Calcula recorte central
                    mask_disp = resized_mask[y:y+h, x:x+w]  # Recorta máscara redimensionada
                else:
                    y, x = (h - nh) // 2, (w - nw) // 2  # Calcula padding necessário
                    mask_disp = np.pad(resized_mask, ((y, h-nh-y), (x, w-nw-x)), mode='constant')  # Aplica padding

            overlay = np.stack([disp_img] * 3, axis=-1)  # Cria overlay RGB a partir da imagem base
            if overlay.max() <= 1.0:  # Se imagem está em escala 0-1
                overlay = (overlay * 255).astype(np.uint8)  # Converte para 0-255 uint8

            mask_rot = np.rot90(mask_disp, k=1) > 0  # Rotaciona máscara para alinhar com imagem
            overlay[mask_rot] = [255, 0, 0]  # Colore regiões segmentadas de vermelho
            self.ax_seg.imshow(overlay)  # Exibe overlay com máscara
            self.ax_seg.set_title("Segmentação")  # Título da subplot segmentada
        else:
            self.ax_seg.text(0.5, 0.5, "Sem Segmentação", ha='center')  # Mensagem quando não há máscara

        if self.show_roi and self.roi_mask is not None:  # Se flag ROI ativa e máscara disponível
            self._draw_roi_rect(self.ax_orig)  # Desenha ROI na imagem original
            self._draw_roi_rect(self.ax_seg)  # Desenha ROI na imagem segmentada

        self.canvas.draw()  # Atualiza canvas Tkinter com novos desenhos

    def _draw_roi_rect(self, ax):  # Adiciona retângulo da ROI no eixo fornecido
        rot_roi = np.rot90(self.roi_mask, k=1)  # Rotaciona máscara de ROI para alinhar com exibição
        coords = np.where(rot_roi)  # Obtém coordenadas dos pixels verdadeiros
        if len(coords[0]) > 0:  # Se há ROI a desenhar
            ymin, xmin = coords[0].min(), coords[1].min()  # Coordenadas mínimas do retângulo
            h, w = coords[0].max() - ymin, coords[1].max() - xmin  # Altura e largura do retângulo
            rect = Rectangle((xmin, ymin), w, h, linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')  # Configura retângulo
            ax.add_patch(rect)  # Adiciona retângulo ao plot

    def toggle_roi(self):  # Liga ou desliga exibição do retângulo de ROI
        self.show_roi = not self.show_roi  # Inverte flag de exibição de ROI
        self.display_image()  # Re-renderiza imagens para refletir mudança
