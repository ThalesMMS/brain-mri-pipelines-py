import json  # Leitura e escrita de dados em formato JSON
import traceback  # Captura e formatação de rastros de exceções
from datetime import datetime  # Manipulação de datas e horas
import tkinter as tk  # Interface gráfica
from tkinter import messagebox, ttk  # Caixas de diálogo e widgets estilizados

import numpy as np  # Operações numéricas
try:
    import seaborn as sns  # Visualização estatística
except ImportError:  # Permite inicializar app mesmo sem seaborn
    sns = None
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # Canvas e toolbar Matplotlib
from matplotlib.figure import Figure  # Objeto de figura do Matplotlib


class ExperimentHistoryMixin:  # Métodos de histórico e visualização de experimentos
    def _save_experiment(self, data):  # Persiste experimento no histórico em disco
        data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Adiciona carimbo de tempo atual
        hist = []  # Lista acumulada de experimentos
        if self.experiment_history_path.exists():  # Se arquivo já existe, carrega histórico
            with open(self.experiment_history_path, 'r') as f: hist = json.load(f)  # Lê JSON existente
        hist.append(data)  # Adiciona experimento atual
        with open(self.experiment_history_path, 'w') as f: json.dump(hist, f, indent=2)  # Escreve histórico atualizado

    def plot_confusion_matrix(self, ax, cm, classes, title):  # Desenha matriz de confusão em eixo fornecido
        """Plota matriz de confusão visualmente (Restaurado)."""  # Docstring mantém contexto
        if sns is None:
            raise ImportError("O módulo seaborn é necessário para plotar a matriz de confusão.")
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,  # Usa heatmap Seaborn com valores inteiros
                    xticklabels=classes, yticklabels=classes, cbar=False)  # Define rótulos e remove barra de cores

        ax.set_title(f'{title}', fontsize=10, fontweight='bold')  # Título do plot
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)  # Rotula eixo X com rotação
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)  # Rotula eixo Y

    def view_experiment_history(self):  # Exibe histórico de experimentos em UI separada
        try:
            if not self.experiment_history_path.exists():  # Verifica se arquivo de histórico existe
                messagebox.showinfo("Histórico", "Nenhum arquivo de histórico encontrado.")  # Informa ausência
                return  # Sai se não houver histórico

            with open(self.experiment_history_path, 'r') as f:  # Abre arquivo de histórico
                history = json.load(f)  # Carrega lista de experimentos

            if not history:  # Se histórico está vazio
                messagebox.showinfo("Histórico", "Histórico vazio.")  # Informa ausência de dados
                return  # Sai
            if sns is None:
                messagebox.showerror("Dependência ausente", "O módulo 'seaborn' é necessário para visualizar o histórico.\nInstale com 'pip install seaborn'.")
                return

            history_window = tk.Toplevel(self.root)  # Cria nova janela para histórico
            history_window.title("Histórico Visual de Experimentos")  # Define título da janela
            try: history_window.state('zoomed')  # Tenta maximizar
            except: pass  # Ignora se não suportado

            main_pane = tk.PanedWindow(history_window, orient=tk.HORIZONTAL)  # Pane horizontal
            main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)  # Ajusta pane

            left_frame = tk.Frame(main_pane)  # Frame para listagem
            listbox = tk.Listbox(left_frame, width=40, font=("Arial", 10))  # Listbox de experimentos
            scrollbar = tk.Scrollbar(left_frame, orient="vertical", command=listbox.yview)  # Scroll vertical
            listbox.config(yscrollcommand=scrollbar.set)  # Conecta scroll ao listbox

            listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Posiciona listbox
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  # Posiciona scrollbar

            for i, exp in enumerate(history):  # Itera sobre experimentos
                metric = ""  # Texto de métrica resumida
                if 'val_r2' in exp: metric = f"R²: {exp['val_r2']:.3f}"  # Métrica para regressão
                elif 'val_accuracy' in exp: metric = f"Acc: {exp['val_accuracy']:.1%}"  # Métrica para classificação
                elif 'val_mae' in exp: metric = f"MAE: {exp['val_mae']:.2f}"  # Métrica MAE

                listbox.insert(tk.END, f"{i+1}. {exp.get('model', 'Model')} - {metric}")  # Insere item na lista

            main_pane.add(left_frame)  # Adiciona frame esquerdo ao pane

            right_frame = tk.Frame(main_pane)  # Frame para abas de detalhes
            notebook = ttk.Notebook(right_frame)  # Notebook com abas
            notebook.pack(fill=tk.BOTH, expand=True)  # Expande notebook

            tab_metrics = tk.Frame(notebook)  # Aba de métricas textuais
            txt_metrics = tk.Text(tab_metrics, font=("Courier", 10), wrap=tk.WORD)  # Text widget para detalhes
            txt_metrics.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)  # Ajusta text widget

            tab_charts = tk.Frame(notebook)  # Para Matrizes de Confusão ou Scatterplots
            tab_curves = tk.Frame(notebook)  # Para Curvas de Aprendizado

            main_pane.add(right_frame)  # Adiciona frame direito ao pane

            def render_charts(exp):  # Atualiza abas gráficas conforme experimento selecionado
                for w in tab_charts.winfo_children(): w.destroy()  # Limpa widgets da aba de gráficos
                for w in tab_curves.winfo_children(): w.destroy()  # Limpa widgets da aba de curvas

                try:
                    notebook.forget(tab_charts)  # Remove aba de gráficos se existir
                except:
                    pass
                try:
                    notebook.forget(tab_curves)  # Remove aba de curvas se existir
                except:
                    pass

                curves = exp.get('learning_curves')  # Recupera curvas de aprendizado

                if not curves and 'train_loss' in exp and isinstance(exp['train_loss'], list):  # Compatibilidade com payload antigo
                    curves = {'train_loss': exp['train_loss'], 'val_loss': exp.get('val_loss', [])}  # Reconstrói dicionário básico

                if curves and isinstance(curves, dict) and (len(curves.get('train_loss', [])) > 1):  # Se há curvas suficientes
                    notebook.add(tab_curves, text="Curvas de Aprendizado")  # Adiciona aba de curvas
                    fig = Figure(figsize=(10, 4), dpi=100)  # Figura para curvas

                    ax1 = fig.add_subplot(121)  # Subplot de loss
                    epochs = range(1, len(curves['train_loss']) + 1)  # Eixo de épocas
                    ax1.plot(epochs, curves['train_loss'], 'b-', label='Treino')  # Loss treino
                    if 'val_loss' in curves and len(curves['val_loss']) == len(epochs):
                        ax1.plot(epochs, curves['val_loss'], 'r-', label='Validação')  # Loss val
                    ax1.set_title("Loss")  # Título
                    ax1.legend()  # Legenda
                    ax1.grid(True, alpha=0.3)  # Grade leve

                    metric_key = None  # Define chave da métrica secundária
                    if 'train_acc' in curves: metric_key = 'acc'  # Usa acurácia
                    elif 'train_mae' in curves: metric_key = 'mae'  # Usa MAE
                    elif 'val_accuracy' in exp and isinstance(exp['val_accuracy'], list): metric_key = 'accuracy'  # Compatibilidade

                    if metric_key:
                        ax2 = fig.add_subplot(122)  # Subplot secundário
                        tr_data = curves.get(f'train_{metric_key}', [])  # Série de treino
                        val_data = curves.get(f'val_{metric_key}', [])  # Série de validação

                        if tr_data: ax2.plot(epochs, tr_data, 'b-', label='Treino')  # Plota treino
                        if val_data: ax2.plot(epochs, val_data, 'r-', label='Validação')  # Plota validação
                        ax2.set_title(metric_key.upper())  # Título com nome da métrica
                        ax2.legend()  # Legenda
                        ax2.grid(True, alpha=0.3)  # Grade leve

                    canvas = FigureCanvasTkAgg(fig, master=tab_curves)  # Canvas para curvas
                    canvas.draw()  # Desenha figura
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Posiciona canvas

                has_cm = 'val_confusion_matrix' in exp  # Indicador de matriz de confusão
                has_reg_scatter = 'y_val' in exp and 'y_val_pred' in exp  # Indicador de scatter de regressão

                if has_cm or has_reg_scatter:
                    tab_name = "Matrizes de Confusão" if has_cm else "Predição (Real vs Predito)"  # Título da aba
                    notebook.add(tab_charts, text=tab_name)  # Adiciona aba de gráficos

                    fig2 = Figure(figsize=(10, 5), dpi=100)  # Figura para gráficos de resultado

                    if has_cm:
                        axes_count = 0  # Quantidade de subplots
                        if 'train_confusion_matrix' in exp: axes_count += 1  # Treino presente
                        if 'val_confusion_matrix' in exp: axes_count += 1  # Validação presente
                        if 'test_confusion_matrix' in exp: axes_count += 1  # Teste presente

                        if axes_count > 0:
                            curr_ax = 1  # Índice do subplot atual
                            if 'val_confusion_matrix' in exp:
                                ax = fig2.add_subplot(1, axes_count, curr_ax)  # Subplot para validação
                                cm = np.array(exp['val_confusion_matrix'])  # Converte matriz
                                classes = exp.get('val_classes', ['0', '1'])  # Classes usadas
                                self.plot_confusion_matrix(ax, cm, classes, "Validação")  # Desenha matriz
                                curr_ax += 1  # Avança índice

                            if 'train_confusion_matrix' in exp:
                                ax = fig2.add_subplot(1, axes_count, curr_ax)  # Subplot para treino
                                cm = np.array(exp['train_confusion_matrix'])  # Matriz de treino
                                classes = exp.get('train_classes', ['0', '1'])  # Classes de treino
                                self.plot_confusion_matrix(ax, cm, classes, "Treino")  # Desenha matriz

                        elif has_reg_scatter:
                            y_train = np.array(exp['y_train'])  # Valores reais de treino
                            y_train_pred = np.array(exp['y_train_pred'])  # Predições treino
                            y_val = np.array(exp['y_val'])  # Valores reais de val
                            y_val_pred = np.array(exp['y_val_pred'])  # Predições val

                            ax1 = fig2.add_subplot(121)  # Subplot para treino
                            ax1.scatter(y_train, y_train_pred, alpha=0.6, s=60, c='blue',
                                       edgecolors='darkblue', linewidths=0.5)  # Dispersão real vs predito
                            min_val = min(y_train.min(), y_train_pred.min())  # Valor mínimo para linha identidade
                            max_val = max(y_train.max(), y_train_pred.max())  # Valor máximo para linha identidade
                            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3,
                                    label='Predição Perfeita', zorder=10)  # Linha de referência

                            train_r2 = exp.get('train_r2', 0)  # R² treino
                            train_mae = exp.get('train_mae', 0)  # MAE treino
                            train_rmse = exp.get('train_rmse', 0)  # RMSE treino
                            ax1.text(0.05, 0.95,
                                    f'R² = {train_r2:.4f}\n'
                                    f'MAE = {train_mae:.2f} anos\n'
                                    f'RMSE = {train_rmse:.2f} anos\n'
                                    f'N = {len(y_train)} amostras',
                                    transform=ax1.transAxes, fontsize=11, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9,
                                             edgecolor='darkblue', linewidth=2))  # Caixa de métricas
                            ax1.set_xlabel('Idade Real (anos)', fontsize=12, fontweight='bold')  # Label X
                            ax1.set_ylabel('Idade Predita (anos)', fontsize=12, fontweight='bold')  # Label Y
                            ax1.set_title('Treino: Predito vs Real', fontsize=13, fontweight='bold', pad=15)  # Título
                            ax1.legend(loc='lower right', fontsize=10, framealpha=0.9)  # Legenda
                            ax1.grid(True, alpha=0.3, linestyle='--')  # Grade

                            ax2 = fig2.add_subplot(122)  # Subplot para validação
                            ax2.scatter(y_val, y_val_pred, alpha=0.6, s=60, c='orange',
                                       edgecolors='darkorange', linewidths=0.5)  # Dispersão val
                            min_val = min(y_val.min(), y_val_pred.min())  # Min para linha identidade
                            max_val = max(y_val.max(), y_val_pred.max())  # Max para linha identidade
                            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3,
                                    label='Predição Perfeita', zorder=10)  # Linha identidade

                            val_r2 = exp.get('val_r2', 0)  # R² val
                            val_mae = exp.get('val_mae', 0)  # MAE val
                            val_rmse = exp.get('val_rmse', 0)  # RMSE val
                            ax2.text(0.05, 0.95,
                                    f'R² = {val_r2:.4f}\n'
                                    f'MAE = {val_mae:.2f} anos\n'
                                    f'RMSE = {val_rmse:.2f} anos\n'
                                    f'N = {len(y_val)} amostras',
                                    transform=ax2.transAxes, fontsize=11, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9,
                                             edgecolor='darkorange', linewidth=2))  # Caixa de métricas val
                            ax2.set_xlabel('Idade Real (anos)', fontsize=12, fontweight='bold')  # Label X
                            ax2.set_ylabel('Idade Predita (anos)', fontsize=12, fontweight='bold')  # Label Y
                            ax2.set_title('Validação: Predito vs Real', fontsize=13, fontweight='bold', pad=15)  # Título
                            ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)  # Legenda
                            ax2.grid(True, alpha=0.3, linestyle='--')  # Grade

                    fig2.tight_layout()  # Ajusta layout dos gráficos de resultado
                    canvas2 = FigureCanvasTkAgg(fig2, master=tab_charts)  # Canvas para gráficos
                    canvas2.draw()  # Desenha figura
                    canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)  # Exibe canvas

            def on_select(event):  # Handler para seleção de experimento na lista
                sel = listbox.curselection()  # Obtém seleção
                if not sel: return  # Sai se nada selecionado
                idx = sel[0]  # Índice selecionado
                exp = history[idx]  # Experimento correspondente

                txt_metrics.config(state=tk.NORMAL)  # Habilita edição temporária
                txt_metrics.delete('1.0', tk.END)  # Limpa texto

                report = f"Experimento: {exp.get('name', 'N/A')}\n"  # Cabeçalho
                report += f"Modelo: {exp.get('model', 'N/A')}\n"  # Modelo usado
                report += f"Data: {exp.get('timestamp', 'N/A')}\n"  # Timestamp
                report += "-"*40 + "\n"  # Separador

                for k, v in exp.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):  # Apenas numéricos
                        report += f"{k}: {v:.4f}\n"  # Adiciona linha formatada
                    elif k == 'best_params':  # Hiperparâmetros salvos
                        report += f"\nMelhores Parâmetros:\n{json.dumps(v, indent=2)}\n"  # Formata como JSON
                    elif k == 'features':  # Lista de features
                        report += f"\nFeatures ({len(v)}):\n{', '.join(v)}\n"  # Lista features

                txt_metrics.insert(tk.END, report)  # Insere texto no widget
                txt_metrics.config(state=tk.DISABLED)  # Desabilita edição

                render_charts(exp)  # Atualiza abas gráficas com experimento selecionado

            listbox.bind('<<ListboxSelect>>', on_select)  # Associa handler de seleção do listbox

            notebook.add(tab_metrics, text="Detalhes / Métricas")  # Adiciona aba de detalhes textuais

        except Exception as e:  # Captura erros na abertura do histórico
            messagebox.showerror("Erro", f"Erro ao abrir histórico: {str(e)}")  # Mostra erro na UI
            traceback.print_exc()  # Imprime stack trace no console
