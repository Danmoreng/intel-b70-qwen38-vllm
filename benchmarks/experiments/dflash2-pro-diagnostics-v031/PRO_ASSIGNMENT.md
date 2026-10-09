# Codex-Arbeitsauftrag: B70 / Qwen3.8-27B EXL3 / DFlash2

Stand: Review des Archivs vom 9. Oktober 2026. Dieser Auftrag ist eine Arbeitsentscheidung, keine Performance- oder Korrektheitsqualifikation. Die Review-Umgebung hat keine B70-Tests ausgeführt. 25 Messzeilen wurden aus 75 gespeicherten Rohzusammenfassungen nachgerechnet; zusätzlich wurden die DFlash7-Positionszähler gegen Prometheus-Differenzen geprüft.

## Ziel und Grenzen

Erhöhe realen nutzbaren Decode-Durchsatz bei langem C1-Kontext, nicht nur die Akzeptanzquote. Produktion bleibt zunächst unverändert: vLLM 0.31.0, Qwen3.8-27B EXL3, FP16, MTP3, FP8-KV, 262144 Gesamt-Tokens, 180 W. Target und Draft rechnen FP16; notwendige interne FP32-Akkumulation und Offline-Referenzen sind erlaubt. Kein BF16/FP16-Bridging im Serving. Keine neue Power-Matrix, keine erneute große MTP-Depth-Matrix und keine Wiederholung der qualifizierten MTP3-Baseline als erster Schritt. Vorhandenen Experiment-Checkout und Rollback nutzen; nicht blind Produktion oder Referenz-Checkouts patchen.

Ein späterer DFlash-Serving-Kandidat benötigt mindestens 192000 Gesamt-Tokens. Unquantisierte 64000–96000-Token-Läufe sind nur Diagnose. „Server startet mit max_model_len=192000“ ist kein Nachweis für tatsächlich benutzbare 192000 Tokens.

## Gesicherter Ausgangspunkt

32K C1, jeweils drei gespeicherte Wiederholungen, native Decode-Mediane:

| Variante | tok/s | nützliche Tokens/Runde, näherungsweise | Rundenzeit-Äquivalent, ms |
|---|---:|---:|---:|
| MTP3, gespeicherte Produktion | 55,19 | 2,657 | 49,63 |
| MTP4 | 59,15 | 3,138 | 53,46 |
| MTP5 | 55,95 | 3,230 | 57,12 |
| MTP6 | 59,56 | 3,488 | 60,85 |
| MTP7 | 59,54 | 3,542 | 62,66 |
| DFlash7, breiter M04 | 52,75 | 3,319 | 63,25 |

Rundenzeit-Äquivalente enthalten Host, Draft und Target; sie sind keine GPU-Komponentenprofile. Mediane und aggregierte Zähler nicht zu einer vermeintlich exakt passenden Einzelrunde kombinieren. Bei 64K liegen MTP4 / DFlash7 bei 52,51 / 45,13 tok/s. MTP6/7 sind mit rund 0,7 % gegenüber MTP4 und erheblicher Streuung nicht qualifiziert besser.

DFlash7 32K: 929 Runden, akzeptierte Präfixe je Position [677, 484, 343, 241, 181, 130, 98]. Bedingte Fortsetzungsraten etwa 70–75 %, kein einzelner auffälliger Absturz. Für denselben Achterblock ergeben sich aus diesen Zählern U7≈3,319 und U4≈2,878. Ein statisch auf vier verifizierte Tokens gekürzter Achterblock müsste unter unveränderten Verteilungen rund 13,3 % Rundenzeit einsparen, um den vollständigen Achterblock zu schlagen. Das ist eine Screening-Rechnung, kein gemessener Effekt. Der bisherige DFlash4-Lauf benutzte einen Fünferblock und ist nicht dieser Versuch.

## Code-Orte und Arbeitsannahmen

Im Archiv bedeutet V `runtime/experiment/installed-vllm`, E `runtime/experiment/exl3xpu`. Übertrage die Stellen kontrolliert auf den tatsächlich attestierten Experiment-Checkout.

1. **FP16-Residual-Patch zunächst erhalten.** `V/model_executor/models/qwen3_dflash2.py:273–275,301–314,405–406`: Residual-Skalierung a=1/16, beide Blocknormen und finale Norm mit epsilon*a²; Branch-Ausgänge einmal skaliert. In exakter Arithmetik konsistent. Context-hidden-norm und Q/K-Normen nicht pauschal mitskalieren. Vorliegende Replay-Tests decken Selector, q-aware Rejection und paged FP8-KV nicht vollständig ab. Epsilon muss intern mit ausreichender Präzision verarbeitet werden: 1e-6/256≈3,90625e-9.
2. **Blockgröße ist momentan an Verifikationslänge gekoppelt.** `qwen3_dflash2.py:284`, `V/v1/worker/gpu/spec_decode/dflash/speculator.py:62,309–311,365–366,416–445,488–495`. Nicht nur die Convolution ändern: Query-Geometrie, Attention-Metadaten, Positionen und Graphs müssen denselben Draftblock repräsentieren.
3. **Adaptive-Schalter ist kein verwendbarer Shortcut.** `E/attention_dispatch.py:168–176,184–188`: enable_adaptive_verification schaltet fixed_query_partition ab und nimmt damit den geprüften schnellen Attention-Pfad heraus. `V/config/speculative.py:552–554` dokumentiert die Option für DSpark. Keine neue DFlash-Adaption durch bloßes Setzen dieses Flags.
4. **W4A16 ist kein Drop-in.** Draft verwendet globale Runtime-Layer-IDs 64–68 (`qwen3_dflash.py:773–777`), Checkpoints lokale 0–4. `E/vllm_plugin.py:104–126` sucht kanonische Runtime-Namen im Checkpoint-Inventar und wählt bei vollständig fehlendem Treffer UnquantizedLinearMethod. `E/loader_audit.py:10–35` verwendet das Target-Inventar auch für den Draft. `E/checkpoint.py:15–19` verwirft bei Nicht-Layer-Modulen übergeordnete Namensbestandteile. Rollenbezogene Zuordnung und getrennte Inventare erforderlich.
5. **Quantisierter Context-KV-Pfad hat vermeidbare Arbeit.** `qwen3_dflash.py:602–611` projiziert vollständiges QKV und verwirft Q. Geometrie: Q=4096, K+V=2048; also 6144 statt 2048 Projektionsausgaben. Erst korrekt quantisieren, dann diese Mehrarbeit gesondert optimieren. Nicht ungeprüft gepackte Gewichte slicen oder dauerhaft nach FP16 dequantisieren.

## Phase A — kleiner Korrektheits- und Profiling-Schritt

### A0: Cache-Ledger ohne neue GPU-Matrix

Die realen DFlash-Startlogs melden zwei Padding-Layer bei den 48 Mamba-Layern und vier bei den 16 Full-Attention-Layern. `V/v1/core/kv_cache_utils.py:1558–1592` wählt hier Gruppengröße 5. Die 25%-Warnung ist eine Obergrenze für den betreffenden Layer-Bucket, kein pauschaler Verlust von 25 % des gesamten Pools.

Erzeuge aus den tatsächlichen KVCacheGroupSpecs eine Byte-/Seitenbilanz für 96000 und 192000, ohne lange Generierungen: echte Layer, Padding, page_size_bytes, resident blocks, SWA-Retention, Mamba-Speculationsblöcke, Graph-Scratch. Die 107345 Cachetokens bei B8/V7 sind gemäß `get_kv_cache_capacity` ein aus Maximal-Concurrency abgeleiteter Wert bei konfigurierten 96000, nicht die gemessene maximal mögliche Einzellänge.

Eine bedingte Rekonstruktion reproduziert diesen Wert: 369 Poolblöcke, 330 pro 96K-Anfrage, 1664 Tokens/Seite, Gruppengröße 5. Unter den unveränderten angenommenen Reservierungen ergibt sie für 192K etwa 8,918 GiB Poolbedarf; diese Annahmen müssen gegen den Spec-Dump bestätigt werden. Mit Gruppengröße 4 statt 5 ergäben sich 4 Full-, 12 Mamba- und 2 Draft-SWA-Gruppen und grob 7,465 GiB. Das ist ein CPU-Rechenmodell, kein bereits getesteter Patch oder Kapazitätsnachweis. Gruppengröße 4 ist ein gezielter Kandidat: kein Padding bei 16 Full-/48 Mamba-Layern, zusätzliches Padding nur beim kurzlebigen SWA-Draft. Prüfe zusätzliche Gruppierungs-/Schedulerkosten, statt blind GCD=1 oder andere Defaults einzuführen. Quantisierung und Gruppierung in getrennten Schritten messen.

Außerdem melden die Logs, dass keine Draft-KV-Gruppe identifiziert wurde (`evidence/runs/wide-dflash7/startup/worker.log:89–92`). `_annotate_eagle_groups` erkennt einen speziellen Non-Causal-Marker oder einen MTP-Fallback, aber keinen allgemeinen fünf-lagigen DFlash-Draft (`kv_cache_utils.py:2138–2154,2170–2225`). Ermittle die tatsächlichen Consumers von `is_eagle_group` und teste Prefix-Reuse/Rejection an Seitenrändern. Markiere nicht blind nur die letzte Gruppe, insbesondere wenn der Draft künftig auf zwei Gruppen verteilt ist. Dieser Befund beweist noch keine Ursache der niedrigen Akzeptanz.

### A1: Trace/Replay statt neuem großen Serving-Benchmark

Lasse B=8, V=7, Draftgewichte, FP8-KV, Samplingparameter und 180 W unverändert. Nutze vorhandene kurze numerische Captures, wo ausreichend; ergänze einen kleinen deterministischen 32K-C1-Capture mit etwa 64–128 Decode-Runden. Kein Tensor-Dump in dem Lauf, dessen Performance später verglichen wird.

Erfasse für ausgewählte Runden:

- Request-State-ID, absolute Tokenpositionen, Anchor-Token-ID, Mask-ID 248070, tatsächliche gültige Kontextlänge, Rejection-Länge und Cache-Slots.
- Target-Features [5,19,33,47,61] nach vollständigem Block einschließlich Residualaddition; Reihenfolge vor fc.
- Normierte Branch-Eingaben, Convolution-Koeffizienten und finale Draft-Hidden-States für wenige Zeilen.
- Top16-IDs und Unary-Logits, Selector-Kantenwerte, den tatsächlich benutzten Vorgänger pro Schritt sowie tatsächlich gesampelte Tokens.
- Tatsächlich gespeicherte FP32-q-Scores vor Temperatur und normalisierte q-Verteilung, Target-p nach dessen Sampling-Transformationen, RNG-Seed/Position/Draft-Salt, Acceptance-/Residual-Entscheidungen.
- Draft-K/V vor Schreiben, gelesene dequantisierte Werte, Skalen und Kontext-/Query-Slots für ausgewählte Layer und Seitenränder.

Ergänze die tatsächlichen transitiven Runtime-Quellen/Hashes für Qwen3Next-Forward, Aux-State-Helfer und RMSNorm. Die im Archiv sichtbare +1-Übersetzung der Feature-Taps ist mit der upstream Post-Layer-Konvention vereinbar; nicht blind „korrigieren“.

Replay-Gates:

1. Rekonstruiere q aus den realisierten Selector-Scores und genau einer Temperaturdivision. q außerhalb der 16 Kandidaten ist null. Rejection muss dasselbe q verwenden. Target-top-p/top-k nicht nachträglich auf q anwenden, sofern die Proposalziehung das nicht ebenfalls getan hat.
2. Prüfe im gewöhnlichen Rejection-Modus min(1,p(y)/q(y)) und normalisierte positive Restmasse max(p-q,0). Greedy separat; synthetische oder Block-Rejection nicht versehentlich aktivieren.
3. Kleine exakt enumerierbare Vokabulare als unabhängige Referenz; zusätzlich statistischer GPU-Test mit vorab festgelegter Fehlerwahrscheinlichkeit. Fälle: p=q, disjunkte Supports, fast gleiche p/q, T=0/T>0, erste Ablehnung, komplette Annahme, EOS/Tokenlimit, wechselnde Request-Slots, Graph-Replay.
4. Vergleiche paged Draft-Attention mit unabhängiger dichter Referenz für dieselben FP8-quantisierten K/V und danach mit FP16-K/V. GQA=4, RoPE theta=1e7, nicht-kausaler Draftblock und SWA-Grenzen 2047/2048/2049, Block-/Seitengrenzen, veraltete/rejected Slots sowie Batches C1/C4 prüfen. Der M04-Test ist hierfür keine Referenz.
5. Berichte TV-Abstand von q, Kandidatenmassen-Coverage c=sum(p auf Top16), Überlappung eta=sum(min(p,q)), Rangwechsel und Error-Maxima. Nicht nur Cosinus/Endlichkeit ausgeben. Toleranzen vor Interpretation fixieren; offensichtliche Index-/Verteilungsfehler zuerst beheben und keine Performance-Promotion durchführen.

### A2: Komponentenprofil am unveränderten Graph-Pfad

Ein 32K-C1- und nur bei Bedarf ein 64K-C1-Trace der eingefrorenen Fixtures. Erzeuge eine Timeline und GPU-Zeiten für:

Target-EXL3-GEMMs; Target-GDN inklusive spekulativer Zustandsverwaltung; Target-Attention/M04; Context-Feature-FC; Context-KV-Projektion/RoPE/Cachewrite; Draft-Backbone inklusive eigener Attention; vollständiger geteilter Head und Top16; Selector/Convolution; q-Rejection; Host-/Synchronisationslücken.

Graph-Replay und Eager-Diagnose getrennt ausweisen. Asynchrone CPU-Aufrufzeiten nicht als Kernelzeiten summieren. Tensor-Capture nicht in die Leistungszahlen einrechnen. Falls Profiling selbst messbar stört, Kennzahlen nur diagnostisch verwenden und einen unveränderten, nicht instrumentierten Anker behalten.

Liefere pro Optimierung eine obere Schranke der möglichen Gesamtwirkung. Ein kleiner Selector-Anteil begründet keine große Selector-Neuentwicklung. Bei unveränderter Akzeptanz benötigt DFlash7 gegenüber MTP4 rund 10,8 % weniger Rundenzeit bei 32K bzw. 14 % bei 64K. Keine weitere große Depth-Suche.

## Phase B — Draftblock und statische Verifikation sauber trennen

Erst nach A, sofern das Profil die erwartbare Einsparung rechtfertigt. Neue, explizite Größen:

```text
draft_block_size = 8                  # Anchor + sieben Maskpositionen
max_draft_proposals = 7
verification_length = 4 oder 7       # zunächst pro Worker konstant
target_query_rows = 1 + verification_length
```

Zunächst B8/V7 als No-op-Refaktor nachweisen. Danach nur V auf 4 ändern. B8/V4 ist nicht B5/V4.

Der Draft-Backbone rechnet weiterhin alle acht Zeilen, mit identischen nicht-kausalen Attention- und Conv-Grenzen. Target-Verifikation muss tatsächlich nur V+1 Zeilen rechnen; sieben Proposals zu berechnen und lediglich drei als verworfen zu markieren spart keine Target-Rechenarbeit. Drafter-Query-Puffer, Target-Kontext-Feature-Puffer, tatsächliche Proposalzahl und Rejection-q-Puffer getrennt dimensionieren.

Top16/Selector dürfen nach dem vollständigen Achter-Backbone auf das benötigte Präfix beschränkt werden, weil die Selector-Kette vorwärts läuft. Zuerst diagnostisch nachweisen, dass dieses Präfix mit dem vollständigen Siebenerpfad identisch ist; anschließend diese Head-Einsparung als getrennte Optimierung messen. Alle q-Scores des verifizierten Präfixes behalten; verworfene Suffixe dürfen weder Target-KV noch GDN-Zustand committen. Draft benötigt weiterhin Lookahead für acht Querypositionen, auch wenn Target weniger verifiziert.

Graph-/AOT-Keys um Draftblock und Verifikationsgeometrie ergänzen; bestehendes Hashing von num_speculative_tokens allein reicht für die neue Entkopplung nicht. Source-Guards und Rollback aktualisieren. M04-Routing für konstante V explizit attestieren. Heterogene V innerhalb eines Batches später und nur mit passenden Buckets oder geprüftem Fallback.

RNG-Positionen und Draft-Salt nicht verschieben; keine zusätzlichen stateful Ziehungen in das verifizierte Präfix einführen. Dynamische Budgetwahl erst später: Die Entscheidung, Token j zu verifizieren, darf nicht von dessen gezogener Identität oder davon abhängigen Zukunftsscores abhängen. Einfache sichere erste Policy: V vor der Proposalziehung aus vergangenen Runden und Kontextkosten bestimmen. Nachträglich „gute“ Pfade auswählen verzerrt Sampling.

Go: verbesserter tatsächlich ausgegebener Decode-Durchsatz bei gleichem korrekten Target-Sampling; kein stale state, kein Routing-Rückfall. Stop: kein Rundenzeitgewinn, weniger nützliche Tokens ohne Kompensation, inkonsistente q-Logs oder Verlust des schnellen Target-Pfads.

## Phase C — W4A16-Draft und 192K

Falls A zeigt, dass Draftkosten relevant sind, oder um die zwingende Speicherhürde zu prüfen, vor einer adaptiven Neuentwicklung einen realen W4A16-Draft integrieren.

Zuerst fc, Q/K/V/O und MLP-Projektionen quantisieren. Normen, Convolution-Basiskernel und zunächst auch Convolution-Kernelprojektionen, Selector-Codebooks, Selector-Hidden-Projektion sowie Mask-/Embedding-Pfad FP16 halten. Geteilten Target-Head nicht duplizieren oder durch den MTP-Pruned-Head ersetzen. Etwa 1,076 GiB ist die optimistische Rohrechnung einschließlich quantisierter Conv-Projektionen; mit FP16-Conv-Projektionen etwa 1,168 GiB, jeweils vor Metadaten/Arbeitsbereichen. Kein Kapazitätsnachweis.

Implementiere Rollen-Namensräume für Inventare, z.B. (target, canonical_module) und (draft, local_checkpoint_module). Runtime-Layer-IDs 64–68 für Attention nicht global zurücknummerieren; nur Checkpoint-Lookup explizit auf 0–4 abbilden. Separate Inventar-, Shape-, Shard-, Bitbreiten-, Codebook- und Dtype-Audits. Shared-Head-Alias anhand Objekt-/Storage-Identität attestieren. Für veröffentlichte Quants auch bare Selector-Codebook-Namen versus `.weight` kontrolliert mappen. Unbekannte oder fehlende Module müssen hart fehlschlagen; keine stillen Dense-Fallbacks.

Als nächsten getrennten Optimierungsschritt KV-only-EXL3-Projektion bauen: dieselben quantisierten K/V-Mitglieder und Gruppen wie im vollen QKV verwenden, aber Q nicht berechnen. Gegen den korrekten vollen quantisierten QKV-Pfad vergleichen. Die dichte FP16-KV-Kopie des bisherigen Pfades nicht unbeabsichtigt zusätzlich dauerhaft behalten.

Beim Speicher messen: tatsächlich geladene und residente Gewichte, Quant-Metadaten, Graph-Pools/Scratch, dichte q-/Logitpuffer, Context-KV-Buffer, KV-Gruppierung/Padding, SWA-Retention sowie Lookahead-/Mamba-State-Reservierungen bei gleichem max_num_seqs. Keine Speicherfreigabe durch stilles Absenken von Concurrency oder Erhöhen der Leistungsgrenze erkaufen.

192K-Gate: tatsächliche tokenisierte Eingabe plus Ausgabe >=192000, ohne OOM, Preemption oder unerwartete Recomputes; zusätzlich Sampling-/State-Smoke nahe Kontextende. Fehlende Draft-Lookahead-Slots müssen sicher behandelt werden, notfalls durch Target-only am Ende. Ein veränderter Draft darf die Target-Verteilung nicht ändern; Akzeptanzverlust wird als Effizienzproblem gemessen und gegen die Zeitersparnis abgewogen.

## Ergänzende Diagnosen nur durch Befund ausgelöst

Aufgaben-/Thinking-Effekt: zunächst kleines gemischtes Set mit wenigen Mathematik-, Code- und Dialogaufgaben, identische Templates/Seeds und Samplingwerte, nur Thinking an/aus verändern. Separat dasselbe bestehende 32K-Fixture behalten. Nicht gleichzeitig Greedy, Modellquantisierung und Kontext wechseln.

FP8-Draft-KV: nur Draft-KV auf FP16 umstellen, Target-KV unverändert; kurze Diagnose mit fixierten Features/Proposals und dichter Referenz. Nur bei relevantem Fehler oder positiver Gegenprobe weiterverfolgen.

Target-Quantisierung: zuerst Coverage/Overlap am bestehenden EXL3-Target. Eine echte Gegenprobe braucht identische teacher-forced Präfixe und originale Target-Gewichte/Features/Head. Dequantisierte EXL3-Gewichte in FP32 isolieren Rechennumerik, nicht den ursprünglichen Quantisierungsverlust. Fehlen originale Gewichte, diese Frage ausdrücklich offenlassen.

## Begrenzte MTP4-Qualifikation

Separater kleiner Track, keine automatische Promotion. Gespeicherte Referenzen zuerst verwenden. Die relevanten offenen Zellen sind langer C1-Kontext sowie Prosa/C4 mit kontrolliertem Cold-/Warm-Prefix-Zustand; historische warme Cache-Treffer unterschieden sich und dürfen nicht als reiner Depth-Effekt gelten. Kontextkapazität 262144 und State-Reservierungen erhalten.

Nur falls eine aktuelle direkte Referenz für die Promotion wegen Image-/Zeitdrift nötig ist, begründete kleine MTP3/MTP4-Paarung in genau diesen Zellen; keine volle alte Baseline erneut starten. Vorab eine realistische Gewichtung der Nutzerlast festlegen. Vorschlag für Promotion: mindestens 5 % gewichteter Decode-Gewinn, konsistente gepaarte Messungen mit Unsicherheitsintervall über null, keine relevante Prosa-/C4- oder Latenzregression, unveränderte Speicher-/Korrektheitsgrenzen. Bei unklarem Ergebnis MTP3 behalten, nicht automatisch weitere Depths testen.

## Erwartete Lieferung

Erster Commit nur Cache-Ledger, Trace-/Replay-/Profiling-Infrastruktur und Quellenattestierung, ohne Änderung der Produktionsdefaults. Danach schriftliche Entscheidung mit gemessener Kostenaufteilung: entweder eine konkret lokalisierte Korrektur, statisches B8/V4, W4A16-Draft oder Abbruch weiterer DFlash-Optimierung. Pro Experiment eine veränderte Variable, identische Fixtures, Rohdaten, Zählerdefinitionen, Source-/Image-Hashes und Rollback-Nachweis. Kein „PASS“ allein wegen endlicher Ausgaben, gestiegener Akzeptanz oder erfolgreichem Startup.
