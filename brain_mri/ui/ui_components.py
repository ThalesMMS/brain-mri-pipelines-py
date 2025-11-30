import tkinter as tk  # Interface gráfica com Tkinter base
from tkinter import ttk  # Widgets estilizados do Tkinter

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # Canvas e barra de ferramentas do Matplotlib integrados ao Tkinter
from matplotlib.figure import Figure  # Objeto de figura do Matplotlib


class UIMixin:  # Métodos relacionados à interface gráfica
    def _setup_ui(self):  # Constrói todos os componentes visuais do app
        menubar = tk.Menu(self.root)  # Cria barra de menu principal
        self.root.config(menu=menubar)  # Associa barra de menu à janela
        view_menu = tk.Menu(menubar, tearoff=0)  # Cria submenu de visualização sem destacável
        menubar.add_cascade(label="Exibir", menu=view_menu)  # Adiciona submenu "Exibir" ao menubar
        view_menu.add_command(label="Aumentar Fonte", command=lambda: self._change_font(2))  # Ação para aumentar fontes
        view_menu.add_command(label="Diminuir Fonte", command=lambda: self._change_font(-2))  # Ação para reduzir fontes
        view_menu.add_checkbutton(label="Mostrar ROI", command=self.toggle_roi)  # Alterna exibição do ROI

        container = tk.Frame(self.root)  # Frame principal para sidebar e conteúdo
        container.pack(fill=tk.BOTH, expand=True)  # Expande para ocupar toda a janela

        self.sidebar = tk.Frame(container, width=250, bg="#f0f0f0")  # Barra lateral com largura fixa
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)  # Posiciona sidebar à esquerda
        self.sidebar.pack_propagate(False)  # Evita que widgets internos alterem largura

        tk.Label(self.sidebar, text="Painel de Controle", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(pady=10)  # Título da sidebar

        self._create_nav_panel()  # Cria painel de navegação entre imagens
        self._create_zoom_panel()  # Cria painel de zoom das imagens
        self._create_analysis_panel()  # Cria painel de ações de análise
        self._create_ml_panel()  # Cria painel de funções de ML

        self.info_label = tk.Label(self.sidebar, text="", bg="#f0f0f0", wraplength=230, justify=tk.LEFT)  # Label para informações dinâmicas
        self.info_label.pack(side=tk.BOTTOM, padx=10, pady=10)  # Posiciona label no rodapé da sidebar

        self.content_area = tk.Frame(container, bg="white")  # Área para exibição das figuras
        self.content_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)  # Expande área de conteúdo

        self.fig = Figure(figsize=(16, 8), dpi=100)  # Figura Matplotlib com duas subplots
        self.ax_orig = self.fig.add_subplot(121); self.ax_orig.axis('off')  # Eixo para imagem original sem eixos
        self.ax_seg = self.fig.add_subplot(122); self.ax_seg.axis('off')  # Eixo para imagem segmentada sem eixos

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.content_area)  # Canvas que integra Matplotlib ao Tkinter
        self.canvas.draw()  # Desenha a figura inicial
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Adiciona widget do canvas na área de conteúdo

        toolbar = NavigationToolbar2Tk(self.canvas, tk.Frame(self.content_area))  # Barra de ferramentas de navegação Matplotlib
        toolbar.pack(fill=tk.X)  # Posiciona barra no topo do conteúdo
        toolbar.update()  # Atualiza estado da barra

    def _create_button(self, parent, text, cmd, color=None):  # Cria botão padrão na interface
        btn = tk.Button(parent, text=text, command=cmd, font=("Arial", self.font_size))  # Instancia botão com comando
        if color: btn.config(bg=color)  # Ajusta cor se fornecida
        btn.pack(fill=tk.X, padx=5, pady=2)  # Posiciona botão ocupando largura do frame

    def _create_nav_panel(self):  # Monta controles de navegação
        frame = tk.LabelFrame(self.sidebar, text="Navegação", bg="#f0f0f0")  # Frame agrupando botões de navegação
        frame.pack(fill=tk.X, padx=10, pady=5)  # Posiciona frame na sidebar
        self._create_button(frame, "◀ Anterior", self.prev_image)  # Botão para voltar imagem
        self._create_button(frame, "Seguinte ▶", self.next_image)  # Botão para avançar imagem
        self._create_button(frame, "Marcar Inviável", self.mark_not_viable)  # Botão para marcar imagem inviável
        self._create_button(frame, "Gerenciar Inviáveis", self.view_not_viable_images)  # Botão para abrir gestão de inviáveis

    def _create_zoom_panel(self):  # Cria botões de ajuste de zoom
        frame = tk.LabelFrame(self.sidebar, text="Zoom", bg="#f0f0f0")  # Frame para controles de zoom
        frame.pack(fill=tk.X, padx=10, pady=5)  # Adiciona frame na sidebar
        self._create_button(frame, "Ampliar (+)", lambda: self._zoom(1.2))  # Botão de ampliar
        self._create_button(frame, "Reduzir (-)", lambda: self._zoom(1/1.2))  # Botão de reduzir
        self._create_button(frame, "Resetar", self.reset_zoom)  # Botão para resetar zoom

    def _create_analysis_panel(self):  # Cria botões relacionados à análise
        frame = tk.LabelFrame(self.sidebar, text="Análise", bg="#f0f0f0")  # Frame de análise
        frame.pack(fill=tk.X, padx=10, pady=5)  # Posiciona frame
        self._create_button(frame, "Crescimento Região", self.apply_grow_region)  # Botão para segmentação
        self._create_button(frame, "Processar Todas", self.process_all_images)  # Botão para processar em lote
        self._create_button(frame, "Scatterplots", self.generate_scatterplots)  # Botão para gerar scatterplots
        self._create_button(frame, "Heatmap", self.show_correlation_heatmap)  # Botão para exibir heatmap
        self._create_button(frame, "t-SNE/UMAP Embeds", self.run_tsne_umap)  # Botão para visualização dos embeddings

    def _create_ml_panel(self):  # Constrói botões de ML
        frame = tk.LabelFrame(self.sidebar, text="Machine Learning", bg="#f0f0f0")  # Frame de ML
        frame.pack(fill=tk.X, padx=10, pady=5)  # Posiciona frame na sidebar
        self._create_button(frame, "Criar Dataset", self.create_exam_level_dataset)  # Botão para gerar dataset
        self._create_button(frame, "Treinar SVM", self.open_feature_selection_dialog)  # Botão para treinar SVM
        self._create_button(frame, "Treinar XGBoost (Reg)", self.open_feature_selection_dialog_xgboost)  # Botão para treino XGBoost
        self._create_button(frame, "DenseNet Classif.", self.train_densenet_classifier)  # Botão para classificador DenseNet
        self._create_button(frame, "DenseNet Regressor", self.train_densenet_regressor)  # Botão para regressão DenseNet
        self._create_button(frame, "Ver Histórico", self.view_experiment_history)  # Botão para histórico de experimentos

    def _change_font(self, delta):  # Ajusta fontes recursivamente nos widgets
        self.font_size = max(6, self.font_size + delta)  # Atualiza tamanho limitado a mínimo 6

        def recursive_update(w):  # Função interna para percorrer árvore de widgets
            try: w.configure(font=("Arial", self.font_size))  # Tenta aplicar fonte ao widget atual
            except: pass  # Ignora widgets que não suportam configuração de fonte
            for child in w.winfo_children(): recursive_update(child)  # Atualiza recursivamente os filhos

        recursive_update(self.sidebar)  # Inicia atualização a partir da sidebar
