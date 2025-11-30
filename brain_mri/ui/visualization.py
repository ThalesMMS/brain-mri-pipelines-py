from pathlib import Path  # Manipulação de caminhos
import tkinter as tk  # Interface gráfica
from tkinter import messagebox  # Diálogos do Tkinter

import numpy as np  # Operações numéricas
try:
    import pandas as pd  # Manipulação de dados tabulares
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
try:
    import seaborn as sns  # Visualização estatística
except ImportError:
    sns = None
try:
    from sklearn.manifold import TSNE  # Redução de dimensionalidade
    from sklearn.preprocessing import StandardScaler  # Normalização
    SKLEARN_AVAILABLE = True
except ImportError:
    TSNE = StandardScaler = None
    SKLEARN_AVAILABLE = False
try:
    import umap  # type: ignore  # UMAP opcional
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # Canvas e toolbar do Matplotlib no Tkinter
from matplotlib.figure import Figure  # Objeto de figura do Matplotlib


class VisualizationMixin:  # Métodos de visualização de resultados
    def _show_plot_window(self, title, figure):  # Abre janela genérica para exibir um Figure do Matplotlib
        win = tk.Toplevel(self.root)  # Cria nova janela de nível superior
        win.title(title)  # Define título da janela
        win.geometry("800x600")  # Ajusta tamanho padrão da janela
        canvas = FigureCanvasTkAgg(figure, master=win)  # Associa figura ao canvas Tkinter
        canvas.draw()  # Desenha figura no canvas
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Expande canvas na janela

    def _resolve_original_path(self, mri_id: str) -> str:  # Resolve caminho relativo do exame original
        for ext in (".nii.gz", ".nii"):  # Tenta extensões comprimida e não comprimida
            p = self.dataset_dir / f"{mri_id}_axl{ext}"  # Constrói caminho esperado
            if p.exists():  # Se arquivo existe
                try:
                    return str(p.relative_to(self.output_dir.parent)).replace("\\", "/")  # Retorna caminho relativo se possível
                except ValueError:
                    return str(p)  # Se não der relativo, retorna absoluto
        return ""  # Se não encontrou, retorna string vazia

    def generate_scatterplots(self):  # Gera scatterplots entre pares de descritores e exibe em navegação
        try:
            if sns is None:
                messagebox.showerror("Dependência ausente", "O módulo 'seaborn' é necessário para gerar scatterplots.\nInstale com 'pip install seaborn'.")
                return
            if not PANDAS_AVAILABLE:
                messagebox.showerror("Dependência ausente", "O módulo 'pandas' é necessário para gerar scatterplots.\nInstale com 'pip install pandas'.")
                return
            if not self.descriptors_csv.exists():  # Verifica se CSV de descritores existe
                messagebox.showwarning("Arquivo ausente", f"Arquivo de descritores não encontrado:\n{self.descriptors_csv}")  # Alerta sobre ausência
                return

            df_desc = pd.read_csv(self.descriptors_csv)  # Lê descritores segmentados
            df_demo = pd.read_csv(self.csv_path, sep=';', decimal=',')  # Lê dados demográficos

            df_demo = df_demo.rename(columns=lambda x: x.strip())  # Remove espaços em nomes de colunas
            if 'MRI ID' in df_demo.columns:
                df_demo = df_demo.rename(columns={'MRI ID': 'MRI_ID'})  # Alinha nome para merge

            merged = pd.merge(df_desc, df_demo[['MRI_ID', 'Group']], on='MRI_ID', how='left')  # Combina descritores com grupos

            color_map = {  # Mapeamento de grupos para rótulos e cores
                'Converted': ('Convertido', 'black'),
                'Nondemented': ('Não Demente', 'blue'),
                'Demented': ('Demente', 'red')
            }

            descriptor_translation = {  # Tradução de nomes de descritores para rótulos legíveis
                'ventricle_area': 'Área',
                'ventricle_perimeter': 'Perímetro',
                'ventricle_circularity': 'Circularidade',
                'ventricle_eccentricity': 'Excentricidade',
                'ventricle_solidity': 'Solidez',
                'ventricle_major_axis_length': 'Eixo Maior',
                'ventricle_minor_axis_length': 'Eixo Menor'
            }

            descriptor_cols = [c for c in df_desc.columns if c not in ['viable', 'MRI_ID', 'Subject_ID', 'segmented_path']]  # Lista apenas descritores numéricos
            plot_figures = []  # Lista para armazenar figuras geradas

            for i in range(len(descriptor_cols)):
                for j in range(i + 1, len(descriptor_cols)):
                    x_col = descriptor_cols[i]  # Coluna do eixo X
                    y_col = descriptor_cols[j]  # Coluna do eixo Y

                    merged[x_col] = pd.to_numeric(merged[x_col], errors='coerce')  # Converte para numérico
                    merged[y_col] = pd.to_numeric(merged[y_col], errors='coerce')  # Converte para numérico

                    plot_df = merged.dropna(subset=[x_col, y_col])  # Remove linhas com NaN nos eixos
                    if plot_df.empty:
                        continue  # Pula se não houver dados válidos

                    fig = Figure(figsize=(8, 6))  # Cria figura para o par
                    ax = fig.add_subplot(111)  # Adiciona eixo único

                    for group_name, (group_label, color) in color_map.items():
                        grp = plot_df[plot_df['Group'] == group_name]  # Filtra grupo
                        if not grp.empty:
                            ax.scatter(
                                grp[x_col], grp[y_col],
                                c=color, label=f"{group_label} ({len(grp)})",
                                alpha=0.8, edgecolors='w', linewidths=0.3
                            )  # Plota pontos do grupo

                    unknown = plot_df[~plot_df['Group'].isin(color_map.keys())]  # Registros sem grupo mapeado
                    if not unknown.empty:
                        ax.scatter(
                            unknown[x_col], unknown[y_col],
                            c='gray', label=f"Outros ({len(unknown)})", alpha=0.6
                        )  # Plota grupo indefinido

                    x_label = descriptor_translation.get(x_col, x_col)  # Label amigável eixo X
                    y_label = descriptor_translation.get(y_col, y_col)  # Label amigável eixo Y

                    ax.set_xlabel(x_label)  # Define label X
                    ax.set_ylabel(y_label)  # Define label Y
                    ax.set_title(f"{y_label} vs {x_label}")  # Define título do gráfico
                    ax.legend(title="Grupo")  # Mostra legenda com grupos
                    ax.grid(True, alpha=0.3)  # Habilita grade leve
                    fig.tight_layout()  # Ajusta layout

                    out_path = self.output_dir / f"scatter_{x_col}_vs_{y_col}.png"  # Caminho de saída do PNG
                    fig.savefig(out_path)  # Salva figura
                    plot_figures.append((fig, f"{y_label} vs {x_label}"))  # Armazena figura e título

            if not plot_figures:
                messagebox.showinfo("Concluído", "Nenhum scatterplot gerado.")  # Informa ausência de gráficos
                return

            viewer = tk.Toplevel(self.root)  # Janela para navegação dos plots
            viewer.title("Scatterplots de Descritores")  # Título da janela
            viewer.geometry("900x750")  # Dimensões da janela

            current_plot = {"idx": 0}  # Índice atual do plot exibido

            plot_frame = tk.Frame(viewer)  # Frame para o canvas do gráfico
            plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  # Posiciona frame

            lbl_title = tk.Label(viewer, text="", font=("Arial", 13, "bold"))  # Label para título do plot
            lbl_title.pack(side=tk.TOP, pady=4)  # Posiciona label

            nav_frame = tk.Frame(viewer)  # Frame para botões de navegação
            nav_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)  # Posiciona frame inferior

            def show_plot(idx):  # Renderiza plot pelo índice
                for widget in plot_frame.winfo_children():
                    widget.destroy()  # Limpa widgets anteriores

                fig, title = plot_figures[idx]  # Seleciona figura e título
                canvas = FigureCanvasTkAgg(fig, master=plot_frame)  # Cria novo canvas
                canvas.draw()  # Desenha figura
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Exibe canvas

                toolbar = NavigationToolbar2Tk(canvas, plot_frame)  # Cria toolbar
                toolbar.update()  # Atualiza estado

                lbl_title.config(text=f"{title} ({idx + 1}/{len(plot_figures)})")  # Atualiza título com índice

                btn_prev.config(state=tk.NORMAL if idx > 0 else tk.DISABLED)  # Habilita/desabilita botão anterior
                btn_next.config(state=tk.NORMAL if idx < len(plot_figures) - 1 else tk.DISABLED)  # Habilita/desabilita botão próximo

            def go_prev():  # Vai para gráfico anterior
                if current_plot["idx"] > 0:
                    current_plot["idx"] -= 1  # Decrementa índice
                    show_plot(current_plot["idx"])  # Atualiza exibição

            def go_next():  # Vai para próximo gráfico
                if current_plot["idx"] < len(plot_figures) - 1:
                    current_plot["idx"] += 1  # Incrementa índice
                    show_plot(current_plot["idx"])  # Atualiza exibição

            btn_prev = tk.Button(nav_frame, text="< Anterior", width=15, command=go_prev)  # Botão anterior
            btn_next = tk.Button(nav_frame, text="Próximo >", width=15, command=go_next)  # Botão próximo
            btn_prev.pack(side=tk.LEFT, padx=20, pady=10)  # Posiciona botão anterior
            btn_next.pack(side=tk.RIGHT, padx=20, pady=10)  # Posiciona botão próximo

            show_plot(0)  # Exibe primeiro gráfico

        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao gerar scatterplots:\n{str(e)}")  # Exibe erro ao gerar plots

    def show_correlation_heatmap(self):  # Gera heatmap de correlação do dataset com classes one-hot
        """Visualiza heatmap de correlação incluindo as Classes (Demented/Nondemented)."""  # Docstring da função
        try:
            if sns is None:
                messagebox.showerror("Dependência ausente", "O módulo 'seaborn' é necessário para gerar o heatmap.\nInstale com 'pip install seaborn'.")
                return
            if not PANDAS_AVAILABLE:
                messagebox.showerror("Dependência ausente", "O módulo 'pandas' é necessário para gerar o heatmap.\nInstale com 'pip install pandas'.")
                return
            split_path = self.output_dir / "exam_level_dataset_split.csv"  # Caminho do CSV combinado
            if not split_path.exists():  # Se não existir dataset
                messagebox.showwarning("Arquivo não encontrado", "Execute 'Criar Dataset' primeiro.")  # Solicita criação
                return

            df = pd.read_csv(split_path)  # Lê dataset combinado

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Identifica colunas numéricas
            exclude_cols = ['converted']  # Remover flags técnicas se necessário
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]  # Remove colunas indesejadas

            df_numeric = df[numeric_cols].dropna(axis=1, how='all')  # Remove colunas totalmente vazias

            if 'Final_Group' in df.columns:
                class_dummies = pd.get_dummies(df['Final_Group'], prefix='Class')  # One-hot das classes
                df_numeric = pd.concat([df_numeric, class_dummies], axis=1)  # Anexa colunas de classe

            if len(df_numeric.columns) < 2:
                messagebox.showwarning("Dados insuficientes", "Colunas insuficientes para correlação.")  # Avisa se poucas colunas
                return

            correlation_matrix = df_numeric.corr()  # Calcula matriz de correlação

            heatmap_window = tk.Toplevel(self.root)  # Cria janela para o heatmap
            heatmap_window.title("Correlação de Features - Dataset Completo")  # Título da janela
            try: heatmap_window.state('zoomed')
            except: pass

            main_frame = tk.Frame(heatmap_window)  # Frame para o canvas
            main_frame.pack(fill=tk.BOTH, expand=True)  # Expande frame

            n_features = len(correlation_matrix.columns)  # Número de features
            fig_size = max(10, min(25, n_features * 0.8))  # Ajusta tamanho base
            fig_size *= 0.2  # Escala tamanho final
            annot_size = 8 if n_features < 15 else 6  # Tamanho da fonte de anotação
            show_annot = n_features <= 20  # Desliga anotações se houver muitas features

            fig_general = Figure(figsize=(fig_size, fig_size))  # Cria figura do heatmap
            ax_general = fig_general.add_subplot(111)  # Adiciona eixo único

            sns.heatmap(correlation_matrix, annot=show_annot, fmt='.2f', cmap='coolwarm',
                        center=0, square=True, linewidths=0.5,
                        cbar_kws={"shrink": 0.8},
                        ax=ax_general, vmin=-1, vmax=1,
                        annot_kws={"size": annot_size})  # Renderiza heatmap com anotações opcionais

            ax_general.set_title(f"Matriz de Correlação ({n_features} features)", fontsize=14, fontweight='bold')  # Título do heatmap

            ax_general.set_xticks(range(len(correlation_matrix.columns)))  # Define ticks no eixo X
            ax_general.set_yticks(range(len(correlation_matrix.columns)))  # Define ticks no eixo Y
            ax_general.set_xticklabels(correlation_matrix.columns, rotation=90, ha='right', fontsize=9)  # Rotula eixo X
            ax_general.set_yticklabels(correlation_matrix.columns, rotation=0, fontsize=9)  # Rotula eixo Y

            fig_general.tight_layout()  # Ajusta layout para evitar sobreposição

            canvas_general = FigureCanvasTkAgg(fig_general, master=main_frame)  # Canvas do heatmap
            canvas_general.draw()  # Desenha figura
            canvas_general.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Exibe canvas

            toolbar = NavigationToolbar2Tk(canvas_general, main_frame)  # Toolbar de navegação
            toolbar.update()  # Atualiza toolbar

            # Salva automaticamente na pasta de saída
            heatmap_path = self.output_dir / "correlation_heatmap.png"
            fig_general.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap salvo em: {heatmap_path}")

        except ImportError:
            messagebox.showerror("Erro", "Seaborn não instalado.")  # Alerta para dependência ausente
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar heatmap:\n{str(e)}")  # Exibe erro genérico

    def run_tsne_umap(self):  # Gera t-SNE/UMAP para embeddings salvos
        # Checagem dinâmica para refletir instalações após o carregamento
        global SKLEARN_AVAILABLE, TSNE, StandardScaler
        if not SKLEARN_AVAILABLE:
            try:
                from sklearn.manifold import TSNE as _TSNE  # type: ignore
                from sklearn.preprocessing import StandardScaler as _StandardScaler  # type: ignore
                TSNE, StandardScaler = _TSNE, _StandardScaler
                SKLEARN_AVAILABLE = True
            except ImportError:
                messagebox.showerror("Dependência ausente", "Instale 'scikit-learn' para gerar t-SNE/UMAP (pip install scikit-learn).")
                return
        if not PANDAS_AVAILABLE:
            messagebox.showerror("Dependência ausente", "Instale 'pandas' para gerar t-SNE/UMAP (pip install pandas).")
            return
        # Importa UMAP dinamicamente na hora (verifica de múltiplas formas)
        umap_available = UMAP_AVAILABLE
        umap_mod = None
        umap_error_msg = None
        
        # Se já estava disponível no topo, tenta usar
        if umap_available:
            try:
                import umap  # type: ignore
                umap_mod = umap
                # Verifica se a classe UMAP realmente existe
                if not hasattr(umap_mod, 'UMAP'):
                    umap_available = False
                    umap_error_msg = "Módulo umap importado, mas classe UMAP não encontrada"
            except (ImportError, ModuleNotFoundError) as e:
                umap_available = False
                umap_error_msg = str(e)
        
        # Se ainda não está disponível, tenta importar agora
        if not umap_available:
            try:
                import importlib
                umap_mod = importlib.import_module("umap")  # type: ignore
                if hasattr(umap_mod, 'UMAP'):
                    umap_available = True
                else:
                    umap_error_msg = "Módulo umap importado, mas classe UMAP não encontrada"
            except (ImportError, ModuleNotFoundError) as e:
                try:
                    # Tenta importar diretamente
                    import umap  # type: ignore
                    umap_mod = umap
                    if hasattr(umap_mod, 'UMAP'):
                        umap_available = True
                    else:
                        umap_error_msg = "Módulo umap importado, mas classe UMAP não encontrada"
                except (ImportError, ModuleNotFoundError) as e2:
                    umap_mod = None
                    umap_error_msg = f"ImportError: {str(e2)}"
                    
        # Log do diagnóstico (para debug)
        if not umap_available and umap_error_msg:
            print(f"[DEBUG] UMAP não disponível: {umap_error_msg}")
            print(f"[DEBUG] UMAP_AVAILABLE (global): {UMAP_AVAILABLE}")
            try:
                import sys
                print(f"[DEBUG] Python path: {sys.executable}")
                import subprocess
                result = subprocess.run([sys.executable, "-m", "pip", "list"], 
                                      capture_output=True, text=True, timeout=5)
                if "umap-learn" in result.stdout or "umap" in result.stdout:
                    print("[DEBUG] umap-learn encontrado na lista de pacotes instalados")
                else:
                    print("[DEBUG] umap-learn NÃO encontrado na lista de pacotes instalados")
            except Exception:
                pass
        emb_files = sorted(self.output_dir.glob("densenet_embeddings_*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)
        if not emb_files:
            messagebox.showwarning("Embeddings não encontrados", "Nenhum arquivo de embeddings encontrado em output/. Treine a DenseNet primeiro.")
            return
        emb_path = emb_files[0]
        try:
            df = pd.read_csv(emb_path)
            if 'target' not in df.columns:
                messagebox.showerror("Erro", f"Arquivo {emb_path.name} não possui coluna 'target'.")
                return
            feature_cols = [c for c in df.columns if c not in {'MRI_ID', 'target'}]
            if not feature_cols:
                messagebox.showerror("Erro", f"Nenhuma coluna de embedding encontrada em {emb_path.name}.")
                return

            X = df[feature_cols].values
            y = df['target'].values
            if len(X) < 3:
                messagebox.showwarning("Dados insuficientes", "Embeddings insuficientes para redução de dimensionalidade.")
                return

            X_scaled = StandardScaler().fit_transform(X)
            perplexity = max(5, min(30, len(X_scaled) - 1))

            # t-SNE
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
            tsne_out = tsne.fit_transform(X_scaled)
            fig_tsne = Figure(figsize=(7, 6))
            ax_tsne = fig_tsne.add_subplot(111)
            if pd.api.types.is_numeric_dtype(df['target']) and df['target'].nunique() > 10:
                sc = ax_tsne.scatter(tsne_out[:, 0], tsne_out[:, 1], c=y, cmap='viridis', alpha=0.8, s=30)
                fig_tsne.colorbar(sc, ax=ax_tsne, label='target')
            else:
                uniq = df['target'].unique()
                palette = sns.color_palette("tab10", n_colors=len(uniq)) if sns else None
                for idx, val in enumerate(uniq):
                    msk = df['target'] == val
                    ax_tsne.scatter(tsne_out[msk, 0], tsne_out[msk, 1],
                                    label=str(val), alpha=0.85, s=30,
                                    c=[palette[idx]] if palette else None)
                ax_tsne.legend(title='target')
            ax_tsne.set_title(f"t-SNE ({emb_path.name})")
            ax_tsne.grid(True, alpha=0.3)
            tsne_path = self.output_dir / f"tsne_{emb_path.stem}.png"
            fig_tsne.tight_layout()
            fig_tsne.savefig(tsne_path, dpi=300, bbox_inches='tight')

            # UMAP opcional
            umap_path = None
            if umap_available and umap_mod is not None:
                try:
                    # Verifica se a classe UMAP existe no módulo
                    if not hasattr(umap_mod, 'UMAP'):
                        raise AttributeError("Módulo umap não possui classe UMAP")
                    reducer = umap_mod.UMAP(n_components=2, random_state=42)
                    umap_out = reducer.fit_transform(X_scaled)
                    fig_umap = Figure(figsize=(7, 6))
                    ax_umap = fig_umap.add_subplot(111)
                    if pd.api.types.is_numeric_dtype(df['target']) and df['target'].nunique() > 10:
                        sc = ax_umap.scatter(umap_out[:, 0], umap_out[:, 1], c=y, cmap='viridis', alpha=0.8, s=30)
                        fig_umap.colorbar(sc, ax=ax_umap, label='target')
                    else:
                        uniq = df['target'].unique()
                        palette = sns.color_palette("tab10", n_colors=len(uniq)) if sns else None
                        for idx, val in enumerate(uniq):
                            msk = df['target'] == val
                            ax_umap.scatter(umap_out[msk, 0], umap_out[msk, 1],
                                            label=str(val), alpha=0.85, s=30,
                                            c=[palette[idx]] if palette else None)
                        ax_umap.legend(title='target')
                    ax_umap.set_title(f"UMAP ({emb_path.name})")
                    ax_umap.grid(True, alpha=0.3)
                    umap_path = self.output_dir / f"umap_{emb_path.stem}.png"
                    fig_umap.tight_layout()
                    fig_umap.savefig(umap_path, dpi=300, bbox_inches='tight')
                except Exception as umap_error:
                    # Se houver erro ao gerar UMAP, continua sem ele mas registra o erro
                    import traceback
                    print(f"Erro ao gerar UMAP (continuando sem ele): {umap_error}")
                    print(traceback.format_exc())
                    umap_path = None

            msg = f"t-SNE salvo em: {tsne_path}"
            if umap_path:
                msg += f"\nUMAP salvo em: {umap_path}"
            else:
                # Mensagem mais informativa sobre o problema do UMAP
                if not umap_available:
                    msg += "\n(UMAP não gerado; o módulo 'umap' não foi encontrado."
                    msg += "\nCertifique-se de que 'umap-learn' está instalado no ambiente virtual correto:"
                    msg += "\npip install umap-learn)"
                else:
                    msg += "\n(UMAP não gerado; ocorreu um erro durante a geração."
                    msg += "\nVerifique o console para mais detalhes.)"
            messagebox.showinfo("t-SNE/UMAP", msg)
        except Exception as e:
            messagebox.showerror("Erro", f"Falha ao gerar t-SNE/UMAP: {e}")
