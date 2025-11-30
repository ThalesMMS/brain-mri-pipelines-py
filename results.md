# DenseNet Classification Experiments

## Experimentos recentes (pós-penalização / sampler)

| Run | Strategy | Key hparams | Epochs (early stop) | Val best acc | Test acc | Prec | Rec | F1 | Balanced acc | Confusion (TN, FP, FN, TP) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4 | Baseline mixup/ls, no class_balance | mixup=0.2, ls=0.05, dropout=0.3, lr=1e-4, wd=1e-4, balance_penalty=0, class_balance=off | ~13 (best val ~4-5) | ~0.61 | **0.571** | 0.561 | 0.902 | **0.692** | ~0.65 | (7, 29, 4, 37) | Melhor equilíbrio observado (antes das mudanças) |
| 5 | Baseline w/ higher dropout | mixup=0.2, ls=0.05, dropout=0.5, lr=1e-4, wd=1e-4, balance_penalty=0, class_balance=off | ~13 | ~0.61 | **0.571** | 0.563 | 0.878 | **0.686** | ~0.63 | (8, 28, 5, 36) | Similar ao Run4; ligeiro ganho de TN |
| 8 | Weighted sampler (dropout 0.5) | sampler, mixup=0.2, ls=0.05, dropout=0.5, lr=1e-4, wd=1e-4, class_balance=off, balance_penalty=0 | ≤20 | 0.714 | 0.558 | 0.684 | 0.317 | 0.433 | **0.575** | (30, 6, 28, 13) | Balanced acc bom, F1 baixo |
| 9 | Weighted sampler (dropout 0.3) | sampler, mixup=0.2, ls=0.05, dropout=0.3, lr=1e-4, wd=1e-4, class_balance=off, balance_penalty=0.05 | ≤20 | 0.518 | 0.571 | 0.682 | 0.366 | 0.476 | 0.586 | (29, 7, 26, 15) | Balanced acc um pouco melhor que Run8; F1 baixo |
| 7 | balance_penalty=0.1 + thresholds_eval | mixup=0.2, ls=0.05, class_balance=off, dropout=0.5, lr=1e-4, wd=1e-4, balance_penalty=0.1 | 18 | 0.589 | 0.455 | 0.490 | 0.610 | 0.543 | 0.50 | (10, 26, 16, 25) | Não supera baseline |
| 6 | balance_penalty=0.5 | mixup=0.2, ls=0.05, class_balance=off, dropout=0.5, lr=1e-4, wd=1e-4 | 20 | 0.66 (val) | 0.506 | 0.588 | 0.244 | 0.345 | 0.50 | (29, 7, 31, 10) | Viés para Nondemented |

**Legenda:** Val best acc = melhor acurácia de validação no early stopping. Balanced acc = média de sensibilidade/especificidade. F1, precision, recall calculados no threshold default (0.5), exceto se indicado.

## Avaliação de limiares (checkpoint bestval atual)

Após reconstrução de `original_path`, avaliação do bestval salvo (último run) no teste:

| Threshold | Acc | Prec | Rec | F1 | Balanced acc | CM (TN, FP, FN, TP) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.4 | 0.447 | 0.447 | 1.000 | 0.618 | 0.50 | (0, 21, 0, 17) |
| 0.5 | 0.500 | 0.417 | 0.294 | 0.345 | 0.48 | (14, 7, 12, 5) |
| 0.6 | 0.553 | 0.000 | 0.000 | 0.000 | 0.50 | (21, 0, 17, 0) |
| 0.7 | 0.553 | 0.000 | 0.000 | 0.000 | 0.50 | (21, 0, 17, 0) |

(O bestval atual não é o melhor histórico; é do treino mais recente.)

## Observações
- Os melhores resultados continuam sendo dos Runs 4/5 (sem sampler, class_balance off, mixup 0.2, label_smoothing 0.05, dropout 0.3–0.5, lr=1e-4, wd=1e-4, balance_penalty=0): Teste Acc ~0.571, F1 ~0.69, CM ~[[7–8, 28–29], [4–5, 36–37]].
- Penalizações/threshold tuning/sampler não superaram o baseline; alguns melhoram balanced acc mas reduzem F1/Recall.
- O checkpoint bestval salvo (`output/densenet_classification_bestval.pth`) é do último run, não o melhor histórico.

## Próximos passos sugeridos
1) Re-rodar exatamente o preset do Run5 (mixup 0.2, ls 0.05, dropout 0.5, class_balance off, balance_penalty=0) e salvar bestval como `densenet_classification_best_run5.pth`, avaliando thresholds 0.45–0.55.
2) Para balanced acc, usar WeightedRandomSampler (Run9) com threshold 0.5 (balanced acc ~0.586), sabendo que o F1 fica menor.
3) Manter `original_path` reconstruído no split para evitar erros de leitura.

## Novos experimentos (rodada final)

| Run | Strategy | Key hparams | Epochs (early stop) | Val best acc | Test acc | Prec | Rec | F1 | Balanced acc | Confusion (TN, FP, FN, TP) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5-replay | baseline replay (mixup/ls, dropout 0.5) | mixup=0.2, ls=0.05, dropout=0.5, lr=1e-4, wd=1e-4, balance_penalty=0, class_balance=off | 11 (best @4) | 0.737 | 0.447 | 0.167 | 0.059 | 0.087 | 0.26 | (16, 5, 16, 1) | Re-run degradou, sem ganho |
| 9 (prev) | sampler, dropout=0.3, bal_pen=0.05 | mixup=0.2, ls=0.05, sampler, dropout=0.3, lr=1e-4, wd=1e-4, class_balance=off | <=20 | 0.518 | 0.571 | 0.682 | 0.366 | 0.476 | 0.586 | (29, 7, 26, 15) | Balanced acc > baseline, F1 baixo |
| 10 | sampler, dropout=0.3, thresholds 0.45–0.5 | mixup=0.2, ls=0.05, sampler, dropout=0.3, lr=1e-4, wd=1e-4, class_balance=off, bal_pen=0 | <=20 | 0.816 | 0.526 | 0.481 | 0.765 | 0.591 | 0.549 | (7, 14, 4, 13) | Melhor sampler desta rodada, mas F1<baseline |

## Checkpoint bestval atual (após rebuild de `original_path`) – thresholds 0.4–0.7

| Threshold | Acc | Prec | Rec | F1 | Balanced acc | CM (TN, FP, FN, TP) |
| --- | --- | --- | --- | --- | --- | --- |
| 0.4 | 0.447 | 0.447 | 1.000 | 0.618 | 0.50 | (0, 21, 0, 17) |
| 0.5 | 0.500 | 0.417 | 0.294 | 0.345 | 0.48 | (14, 7, 12, 5) |
| 0.6 | 0.553 | 0.000 | 0.000 | 0.000 | 0.50 | (21, 0, 17, 0) |
| 0.7 | 0.553 | 0.000 | 0.000 | 0.000 | 0.50 | (21, 0, 17, 0) |

## Novo teste (axl+cor+sag empilhados, grayscale) - run stack5ep
| Run | Strategy | Key hparams | Epochs | Val best acc | Test acc | Prec | Rec | F1 | Balanced acc | Confusion (TN, FP, FN, TP) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stack5ep | 3 vistas empilhadas (axl/cor/sag), grayscale, headless | mixup=0.2, ls=0.05, dropout=0.3, class_balance=off, balance_penalty=0, epochs=5 | 5 | 0.711 | 0.447 | 0.417 | 0.588 | 0.488 | ~0.49 | (7,14,7,10) |

## Novo teste stack_long (axl/cor/sag empilhadas)

| Run | Strategy | Acc@0.5 | Prec | Rec | F1 | Balanced acc | CM (TN, FP, FN, TP) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stack_long | 3 vistas empilhadas (axl/cor/sag), mixup=0.2, ls=0.05, dropout=0.35, class_balance=off, epochs=18 | 0.447 | 0.447 | 1.000 | 0.618 | 0.500 | [[0, 21], [0, 17]] |
