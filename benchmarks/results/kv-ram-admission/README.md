# RAM-KV-Cache: Messstand vom 17. Oktober 2026

Status: isolierter Engine-Kandidat, keine produktive Freigabe. Der bestehende
Release, sein Image-Alias und seine Konfiguration wurden nicht ersetzt.

## Nachgewiesene Kapazität

| Größe | Messwert |
|---|---:|
| GPU-KV-Pool im vollen Profil | 292.898 Tokens |
| Maximale Eingabe plus Ausgabe je Modellanfrage | 262.144 Tokens |
| Konfiguriertes RAM-Cache-Budget | 12 GiB |
| Tatsächliche Cache-Allokation und RSS | 12.867.993.600 Bytes / 11,984375 GiB |
| Nach GPU-Reset aus RAM geladener großer Präfix | 259.200 Tokens |
| Nach GPU-Reset aus RAM geladener zweiter Präfix | 33.600 Tokens |
| Zusammen tatsächlich aus RAM geladen | **292.800 unterschiedliche Tokens** |
| Zugehörige CPU-zu-GPU-Transfers | 10.507.911.168 Bytes |
| Kleinster verfügbarer Host-RAM im Kapazitätstest | 7,82 GiB |

Der große Test hatte 262.059 Eingabetokens. Die erste Anfrage dauerte 551,98 s,
zwei Wiederholungen mit GPU-Präfix 12,16 und 12,12 s. Nach dem Leeren ausschließlich
des GPU-Präfix-Caches benötigte dieselbe Anfrage 12,82 s und lud ihren Präfix
nachweislich aus RAM. Die zweite Historie hatte 35.257 Eingabetokens und wurde
mit 33.600 RAM-Treffern in 2,07 s wiederaufgenommen. Es gab keine Preemption.
Alle Antworten enthalten dieselben drei korrekten, nur im Verlauf vorhandenen
Prüfcodes. Die restlichen 4.516 Eingabetokens beider Anfragen wurden neu berechnet.

Die Historien wurden vor dem GPU-Reset jeweils dreimal verwendet. Der Test
belegt nutzbare RAM-Kapazität und korrekte Wiederherstellung wiederverwendeter
Präfixe; er garantiert diese Ausnutzung nicht für beliebige einmalige Eingaben.
12 GiB sind ungefähr 12,9 dezimale GB.

## Behobene Fehler und weitere Tests

Zwei verschiedene Allokationen mussten begrenzt werden. Die ursprünglichen
XPU-Pinned-Tensoren rundeten den CPU-Cache intern nach oben auf; ein logisches
10-GiB-Budget belegte dadurch etwa 17 GiB. Der Kandidat verwendet stattdessen
den nativen zusammenhängenden mmap-Puffer. Gemessen wurden dort bei 10 GiB
Budget 9,96 GiB und bei 12 GiB Budget 11,98 GiB.

Beim ersten großen RAM-Reload kam zusätzlich ein großer nativer Transferpuffer
hinzu. Das führte zum globalen Host-OOM, nicht nur zu einer zu kleinen
Containergrenze. Der korrigierte Ladepfad überträgt höchstens 128 MiB je Batch
und wartet jeweils auf dessen Abschluss, bevor er den nächsten Batch startet.
Ein separater Test mit komplett belegtem 12-GiB-Quellpuffer kopierte 9,18 GB in
0,80 s korrekt zur GPU; anschließend bestand auch der Modell-Kapazitätstest.

Die Zulassung neuer Anfragen reserviert den vollständigen Input, das erlaubte
Output-Budget und den spekulativen Vorlauf. Zwei Anfragen mit je 200.023
Eingabetokens warteten im vollen GPU-Profil aufeinander, ohne Preemption. Dieser
Test stammt aus v3 vor der späteren ARC-Korrektur; sein Cache-Rückkehrtest schlug
damals fehl. Mit der korrigierten ARC-Anbindung bestanden natürliche GPU-
Verdrängung und RAM-Rückkehr in einem verkleinerten GPU-Pool: zwei 64K-Anfragen
warteten, der vorherige 16K-Verlauf wurde danach korrekt aus RAM geladen.
Auch drei gleichzeitig gestellte RAM-Wiederaufnahmen bestanden dort.

Die Host-Unit-Tests bestehen: 30 bestanden, drei bestehende Tests übersprungen,
weil sie ihre gepinnte Image-Umgebung benötigen. Die schnellen Leistungsproben
der Zwischenversion lagen bei drei kalten 16K-Anfragen rund 13 % hinter dem
gleichen Image ohne Offloading. Das sind einzelne synthetische Messungen und
keine Freigabe eines generellen Geschwindigkeitsgewinns.

## Noch offene Freigabe

Der Wechseltest **A → B → A → B ist fehlgeschlagen**: A hatte 262.059,
B 262.057 Eingabetokens. B wurde in 552,87 s kalt aufgebaut; A kam danach mit
259.200 CPU-Treffern und unverändert korrekter Antwort in 12,79 s zurück. Beim
erneuten B wurden jedoch null GPU- und null CPU-Präfix-Tokens gefunden. Der
Test brach diese Anfrage früh ab, statt weitere neun Minuten neu zu berechnen.
Damit ist die strenge Anforderung zweier abwechselnd erhaltener voller Verläufe
**nicht erfüllt**, und dieser inklusive Kandidat darf dafür nicht freigegeben werden.

CPU- und GPU-Cache halten teilweise dieselben Daten. Deshalb darf man
292.898 GPU-Tokens und 292.800 RAM-Tokens nicht einfach zu einem garantiert
exklusiv verwendbaren Cache-Pool addieren. Ein RAM-Kapazitätstest nach GPU-Reset
beweist noch keinen dauerhaften Erhalt zweier voller Agentenverläufe.
Die [Entwurfsnotiz](../../../docs/KV_RAM_EXCLUSIVE_TIERING.md) beschreibt die
erforderliche exklusive Verwaltung; sie ist keine implementierte Lösung.

Nach dem fehlgeschlagenen Wechseltest bestanden im v6-Kandidaten drei kalte
16K-Anfragen parallel (30,63 s bis zur letzten Antwort), eine kalte 8K-Anfrage
(7,83 s), Abbruch mit anschließender Erholung sowie drei gleichzeitig gestellte
RAM-Wiederaufnahmen mit korrekten Prüfcodes (je 3,71 s). Das gibt keine Freigabe
für die weiterhin fehlende Zwei-Vollverläufe-Eigenschaft.

Der Cockpit-Relay benötigt außerdem die vorbereitete Timeout-Korrektur:
120 Sekunden ohne Upstream-Daten reichen für kalte große Präfills nicht aus.
Die isolierten Cockpit-Commits `2c91491` und `ed34749` erlauben konfigurierbare
30 Minuten und erhalten Task-Abbruch sowie Capability-Widerruf. Sie wurden
getestet, aber nicht produktiv installiert.

Vor einer produktiven Übernahme sind ein eigener attestierter Release mit den
neuen Runtime-Hashes sowie die gemeinsame Installation mit dem Relay-Fix nötig.

Details, Laufzeit-Hashes und Ergebniszusammenfassungen: [assessment.json](assessment.json).
Die danebenliegenden `.json.gz`-Dateien enthalten die vollständigen synthetischen
Messdaten einschließlich Zeitreihen. Die mit archivierte fehlgeschlagene
v3-Rückkehrprüfung darf nicht als bestanden dargestellt werden.

Start- und Testanleitung: [KV_RAM_ADMISSION_CANDIDATE.md](../../../docs/KV_RAM_ADMISSION_CANDIDATE.md).
