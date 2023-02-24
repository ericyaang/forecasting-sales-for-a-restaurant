# Sobre esses dados

Esses dados foram extraídos manualmente do sistema point-of-sale (POS) do estabelecimento. O dados são confidenciais por isso não são disponbilizados.

Usando o script `flat_to_pandas.py` foram extraídos 736 arquivos no formato `.fiscal`. Segue o exemplo do formato bruto:
```
['****Batch****-Auswertung',
 'Fiskal & Z-Bericht Nr. 1 |Mittwoch 13.11.2019',
 'Gedruckt am Donnerstag 14.11.2019 um 05:00:06',
 'Abrechnungen aller Bediener',
 'Abgerechnete Tische......:         67',
 'Stornos..................:         10',
 'Stornobetrag.............:      31.90 EUR',
 'Im Umsatz enthalten',
 '===================',
 'BARZAHLUNG               :    2443.20 EUR',
 'EC-KARTE                 :    1760.90 EUR',
 '                           --------------',
 'Zahlg.im Umsatz..........:    4204.10 EUR',
 'Summe aller Zahlungen....:    4204.10 EUR',
 'Tip......................:     -77.80 EUR',
 '                           --------------',
 'Gesamtumsatz.............:    4126.30 EUR',
 'Bestellumsätze',
 '==============',
 'Summe Im Haus............:    4126.30 EUR',
 'Summe Außer Haus.........:       0.00 EUR',
 'Nicht im Umsatz..........:       0.00 EUR',
 '                           --------------',
 'Gesamt Bestellung Brutto.:    4126.30 EUR',
 'MWST  19.00 %............:     658.82 EUR',
 'Netto 19.00 %............:    3467.48 EUR',
 'Summe Brutto 19.00 %.....:    4126.30 EUR',
 'Summe Netto..............:    3467.48 EUR',
 'Gesamt Bestellung Netto..:    3467.48 EUR',
 'Kassenbetrag',
 '============',
 'BARZAHLUNG               :    2443.20 EUR',
 '- Euro               :    2443,20 EUR',
 'Tip......................:     -77.80 EUR',
 '                           --------------',
 'Kassenbetrag.............:    2365.40 EUR',
 'Gaststatistik',
 '=============',
 'Anzahl Gäste.............:         67',
 'Umsatz pro Gast..........:      61.59 EUR',
 'All amount per guest.....:      61.59 EUR',
 'Umsatz pro Gast Netto....:      51.75 EUR',
 '################################################################################'
```
Variáveis selecionadas:
- `Abgerechnete Tische`: Total number of tables
- `Gesamt Bestellung Brutto.`: Gross sales
- `Gesamt Bestellung Netto`: Net sales

Após a extração e limpeza, o conjuntos de dados fica assim:

| date                |   id |   n_tables |   gross_sales |   net_sales |
|---------------------|------|------------|---------------|-------------|
| 2019-11-13 00:00:00 |    1 |         67 |       4126.3  |     3467.48 |
| 2019-11-14 00:00:00 |    2 |         71 |       5584.35 |     4692.73 |
| 2019-11-15 00:00:00 |    3 |         79 |       5131.4  |     4312.1  |
| 2019-11-16 00:00:00 |    4 |         84 |       6789.9  |     5705.8  |
| 2019-11-17 00:00:00 |    5 |         80 |       5671.7  |     4768.43 |

Esse conjunto de dados então é salvo como `data_raw.parquet`.

- Não será usado a `n_tables` pois ela não representa o real número de mesas do estabelecimento.
- `id` é redundante já que cada observação representa um único dia.
- No final, vamos acabar usando apenas as colunas `net_sales` (venda líquida em EURO) e `date`.

A limpeza e construção desse conjunto de dados não é necessária pois já é possível obte-los em .csv ou por acesso da base de dados SQL.
