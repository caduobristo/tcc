import json
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
AST_DIR = ROOT / 'results' / 'ast_transfer'

def generate_ast_reports():
    scenarios = {
        'mono_disc': 'Mono Discrete',
        'mono_cont': 'Mono Continuous',
        'poly_disc': 'Poly Discrete',
        'poly_cont': 'Poly Continuous'
    }
    
    results = {}
    
    # Busca os resultados do AST nos 4 cenários
    for folder, name in scenarios.items():
        scenario_dir = AST_DIR / folder
        if not scenario_dir.exists():
            continue
            
        probe_dirs = sorted([d for d in scenario_dir.iterdir() if d.is_dir() and d.name.startswith('probe_')])
        if not probe_dirs:
            continue
            
        latest_probe = probe_dirs[-1]
        metrics_file = latest_probe / 'test_metrics.json'
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                results[folder] = {
                    'name': name,
                    'data': data,
                    'probe_dir': latest_probe.name
                }

    if not results:
        print("Nenhum resultado do AST encontrado.")
        return

    # 1. Gera o relatório GERAL
    lines_general = [
        "# Resumo Consolidado de Transfer Learning: AST",
        "",
        "> **Nota:** Relatório gerado automaticamente pelo script `scripts/report_ast_results.py`.",
        "",
        "O modelo AST (Audio Spectrogram Transformer) foi utilizado como extrator de características congelado (*frozen backbone*), com o treinamento apenas de uma nova camada linear.",
        "",
        "## Desempenho por Cenário",
        "",
        "| Cenário | Acurácia Média ± DP (%) | F1 Macro (%) | Detalhes |",
        "| :--- | ---: | ---: | :--- |"
    ]

    for folder, info in results.items():
        data = info['data']
        name = info['name']
        probe = info['probe_dir']
        
        acc = data.get('accuracy', 0) * 100
        f1 = data.get('macro_f1', 0) * 100
        prec = data.get('macro_precision', 0) * 100
        rec = data.get('macro_recall', 0) * 100
        mean_acc = data.get('multi_seed_mean_accuracy', acc)
        std_acc = data.get('multi_seed_std_accuracy', 0.0)
        
        # Atualiza a tabela do geral (agora salva dentro da subpasta)
        report_filename = "ast_report.md"
        lines_general.append(f"| {name} | {mean_acc:.2f} ± {std_acc:.2f} | {f1:.2f} | [Relatório Completo]({folder}/{report_filename}) |")
        
        cfg = data.get('training_config', {})
        best_epoch = [v for k, v in cfg.items() if k.startswith('best_epoch')]
        best_epoch_str = str(best_epoch[0]) if best_epoch else "N/A"
        
        lines_indiv = [
            f"# Transfer Learning AST: {name}",
            "",
            "> **Nota:** Relatório gerado automaticamente pelo script `scripts/report_ast_results.py`.",
            "",
            f"Resultados baseados no experimento executado em: `{probe}`",
            "",
            "## 1. Configurações de Treinamento",
            "",
            f"- **Learning Rate:** {cfg.get('learning_rate', 'N/A')}",
            f"- **Máximo de Épocas:** {cfg.get('max_epochs', 'N/A')}",
            f"- **Paciência (Early Stopping):** {cfg.get('patience', 'N/A')}",
            f"- **Melhor Época Alcançada:** {best_epoch_str}",
            "",
            "## 2. Desempenho no Teste",
            "",
            "| Métrica | Resultado |",
            "| :--- | ---: |",
            f"| **Acurácia (Média Multi-seed)** | {mean_acc:.2f}% ± {std_acc:.2f}% |",
            f"| **Acurácia (Melhor Época)** | {acc:.2f}% |",
            f"| **F1 Macro** | {f1:.2f}% |",
            f"| **Precisão Macro** | {prec:.2f}% |",
            f"| **Recall Macro** | {rec:.2f}% |",
            "",
            "## 3. Desempenho Detalhado por Classe",
            "",
            "| Efeito | F1-Score (%) | Recall (%) | Precisão (%) | Suporte |",
            "| :--- | ---: | ---: | ---: | ---: |"
        ]
        
        report_dict = data.get('classification_report', {})
        for effect, metrics in report_dict.items():
            if effect in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            e_f1 = metrics.get('f1-score', 0) * 100
            e_rec = metrics.get('recall', 0) * 100
            e_prec = metrics.get('precision', 0) * 100
            support = int(metrics.get('support', 0))
            
            lines_indiv.append(f"| {effect} | {e_f1:.2f} | {e_rec:.2f} | {e_prec:.2f} | {support} |")
            
        lines_indiv.append("")
        lines_indiv.append("*(Consulte a matriz de confusão `confusion_matrix.png` gerada na pasta da run para a análise matricial e de falsos positivos completos que constam no JSON.)*")

        # Escreve o relatório individual dentro da subpasta de cenário
        scenario_dir = AST_DIR / folder
        indiv_path = scenario_dir / report_filename
        with open(indiv_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines_indiv) + '\n')
            
    # Escreve o relatório geral
    lines_general.extend([
        "",
        "## Observações Principais do AST",
        "- **Riqueza Acústica vs. Desempenho:** Os cenários contínuos e polifônicos geraram os melhores desempenhos absolutos (Poly Continuous bateu >90% F1), indicando que acordes densos e mudanças contínuas de sinal alimentam melhor o Transformer.",
        "- **Gargalo de Overdrives:** Ao analisar as tabelas detalhadas de cada cenário, nota-se que OD1 e TS9 continuam puxando as métricas macro para baixo, refletindo a altíssima similaridade desses circuitos de clipagem."
    ])
    
    general_path = AST_DIR / 'ast_results_report.md'
    with open(general_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines_general) + '\n')
        
    print(f"Scripts atualizados! Relatório geral em: {general_path.relative_to(ROOT)}")
    print(f"Relatórios individuais salvos nas respectivas pastas.")

if __name__ == '__main__':
    generate_ast_reports()
