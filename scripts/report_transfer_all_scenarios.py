"""Combine verified reports without running extraction or training."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.paths import SCENARIOS, data_root


def main():
    combined = {'schema':1,'encoder_frozen':True,'fine_tuning':False,
                'comparison_scope':'separate per-scenario heads and tests; fixed source split, five training seeds',
                'scenarios':{}}
    lines = ['# Transfer learning congelado nos quatro cenários GUITAR-FX-DIST','',
             'PaSST e HTS-AT usam seus encoders pré-treinados congelados e uma cabeça linear de 9.997 parâmetros. WAVs oficiais de dois segundos, reamostrados para 32 kHz, com o frontend nativo de cada modelo. Os mel16/mel32 Kaldi não alimentam estes experimentos.',
             '', '## Resultados no teste', '',
             'Média ± desvio padrão amostral entre cinco sementes (101, 202, 303, 404 e 505), mantendo a mesma partição por fonte em cada cenário.', '',
             '| Cenário | Modelo | Acurácia (%) | F1 macro (%) | Precisão macro (%) | Recall macro (%) |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for scenario in ('mono_disc','mono_cont','poly_disc','poly_cont'):
        stem = 'transfer_learning_consolidated' if scenario=='mono_disc' else f'transfer_learning_{scenario}_consolidated'
        report = json.loads((ROOT/'results/passt_htsat_transfer'/(stem+'_summary.json')).read_text(encoding='utf-8'))
        if not report['all_predictions_reproduced'] or report['scenario']!=scenario or report['fine_tuning']:
            raise ValueError('A scenario report lacks completed verification')
        expected = {(model,seed) for model in ('passt','htsat') for seed in (101,202,303,404,505)}
        verified = report['verification']
        if (len(verified)!=10 or {(r['model'],r['seed']) for r in verified}!=expected
                or any(not r['predictions_match'] or r['verified_predictions']!=report['split_counts']['test'] for r in verified)):
            raise ValueError('A scenario lacks complete verification of all ten final heads')
        waveform = report['waveform_inference_check']
        if isinstance(waveform,list):  # original Mono Discrete report schema
            waveform = {r['model']:{'samples':r['waveforms'],
                'exact_match':r['max_abs_logit_difference']==0 and r['predicted_classes_match'] and r['finite']} for r in waveform}
        if (set(waveform)!={'passt','htsat'}
                or any(r['samples']!=16 or not r['exact_match'] for r in waveform.values())):
            raise ValueError('A scenario lacks exact waveform/cache inference checks')
        combined['scenarios'][scenario] = report
        for model in ('passt','htsat'):
            metrics = report['models'][model]['aggregate']
            values = [f'{100*metrics[key]["mean"]:.2f} ± {100*metrics[key]["std"]:.2f}'.replace('.',',')
                      for key in ('accuracy','macro_f1','macro_precision','macro_recall')]
            lines.append('| '+SCENARIOS[scenario].replace('_',' ')+' | '+('PaSST' if model=='passt' else 'HTS-AT')+' | '+' | '.join(values)+' |')
    lines += ['', '## Dados, tempos e seleção', '',
              '| Cenário | Áudios selecionados | Fontes | Treino / validação / teste | Extração PaSST / HTS-AT (min) | Consolidação (min) |',
              '| --- | ---: | ---: | --- | ---: | ---: |']
    for scenario,report in combined['scenarios'].items():
        extraction = []
        for model in ('passt','htsat'):
            seconds = report['models'][model].get('embedding_extraction_seconds')
            if seconds is None:
                cache = json.loads((ROOT/f'results/transfer_learning/{scenario}/{model}/embedding_cache.json').read_text())
                seconds = cache['elapsed_seconds']
            extraction.append(seconds/60)
        split = report['split_counts']
        lines.append(f'| {SCENARIOS[scenario].replace("_"," ")} | {report["samples"]:,} | {report["source_groups"]} | {split["train"]:,} / {split["validation"]:,} / {split["test"]:,} | {extraction[0]:.2f} / {extraction[1]:.2f} | {report["elapsed_seconds"]/60:.2f} |')
    lines += ['', 'Cada cenário tem 34 treinamentos de cabeça: oito candidatos por modelo, duas finalistas com mais duas seeds e cinco novas seeds finais por modelo. Busca batch 128/1.024 × scaler ausente/padronização no treino × pesos ausentes/balanceados no treino. AdamW LR 0,001, weight decay 0,01, teto de 200 épocas; scheduler e parada antecipada por F1 macro de validação. As dez cabeças finais de cada cenário são concluídas antes do seu teste.',
              '', 'Extração é medida separadamente; consolidação inclui seleção, confirmação, validação e teste sobre vetores. Os tempos não incluem downloads, configuração do ambiente ou verificações posteriores. Hardware: RTX 5080 de 16 GB. Houve preparação de dados e compressão local em paralelo; os tempos representam esta execução e não um benchmark de desempenho sob carga controlada. Consulte os relatórios individuais para épocas e segundos de cada seed.',
              '', 'A extração HTS-AT de Poly Continuous foi interrompida na execução anterior. O arquivo parcial foi preservado e essa extração foi refeita, antes de iniciar o treinamento das cabeças do cenário; o cache PaSST completo foi reutilizado. A tabela registra a extração concluída e não inclui a tentativa interrompida nem o intervalo entre as execuções.',
              '', '## Relatórios individuais', '']
    for scenario in combined['scenarios']:
        stem = 'transfer_learning_consolidated' if scenario=='mono_disc' else f'transfer_learning_{scenario}_consolidated'
        lines.append(f'- [{SCENARIOS[scenario].replace("_"," ")}]({stem}_report.md): busca, sementes, métricas por classe e artefatos verificados.')
    lines += ['', '## Integridade, fontes e limites', '',
              'Todos os volumes oficiais usados foram verificados pelos MD5 publicados; os WAVs extraídos passaram por CRC, leitura completa, formato/valores finitos e SHA-256. Nenhuma fonte sonora nem conteúdo WAV idêntico atravessa treino/validação/teste dentro de cada cenário. Recarregamos as 40 cabeças finais/scalers e reproduzimos todas as predições salvas. O caminho WAV → encoder → scaler → cabeça também foi comparado em 16 áudios por modelo/cenário, sem diferença nos logits.',
              '', 'Poly Continuous contém 12 pares de WAVs idênticos na classe MGS: dez pares no treino e dois na validação, nenhum no teste. Foram preservados conforme os metadados oficiais, com cada par na mesma fonte e partição; não há vazamento por esses duplicados. Os outros três cenários não contêm duplicados WAV selecionados. O protocolo de Poly Continuous usa os 130.000 arquivos, sem deduplicação de conteúdo.',
              '', 'Os três downloads novos usaram uma área temporária; após MD5 e extração/CRC completos, os volumes temporários foram descartados para caber no disco. WAVs selecionados, metadados, checksums publicados, índice de membros/CRC, manifests e recibos da preparação foram preservados localmente. Os ZIPs anteriormente baixados pelo usuário e os volumes de Mono Discrete foram preservados.',
              '', 'Fontes oficiais: [Mono Discrete](https://zenodo.org/records/4298000), [Mono Continuous](https://zenodo.org/records/4296040), [Poly Discrete](https://zenodo.org/records/4298025), [Poly Continuous](https://zenodo.org/records/4298017).',
              '', 'Mono Discrete teve teste consultado na rodada preliminar. Nos outros três cenários, este protocolo não fez rodada preliminar nem usou métricas de teste na seleção. Há resultados históricos de outros modelos nestes conjuntos, portanto não são fontes externas inéditas ao projeto. O DP entre seeds mede variação do treinamento, não variação entre partições. Contínuo/discreto podem compartilhar fontes dentro da família mono/poly; os resultados são de treinamentos separados, sem afirmar transferência entre cenários.',
              '', 'O baseline FxNet histórico usa divisão por arquivo e outro frontend. Comparações com ele são descritivas e não equivalem a uma comparação controlada. Não houve fine-tuning nem treino dos encoders.',
              '', 'Código, protocolos, notebooks e resumos estão no Git. Dados, embeddings, checkpoints, logs completos e documentação de contexto permanecem locais. Consulta: `notebooks/transfer_learning_all_scenarios.ipynb`, com execução desabilitada por padrão. Inferência: `src.models.consolidated_probe.load_consolidated_probe`, carregando obrigatoriamente o scaler salvo junto à cabeça.']
    (ROOT/'results/passt_htsat_transfer/transfer_learning_all_scenarios_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'results/passt_htsat_transfer/transfer_learning_all_scenarios_summary.json').write_text(json.dumps(combined,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('ALL_SCENARIOS_REPORT_COMPLETE')


if __name__=='__main__':
    main()
