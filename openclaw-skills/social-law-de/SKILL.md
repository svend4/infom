# Skill: social-law-de

**ID:** `social-law-de`
**Version:** 1.0.0
**Author:** svend4
**Category:** legal, social-welfare, germany
**Tags:** sozialrecht, widerspruch, klage, sgb, rehabilitation, behinderung, deutsch, russian

## Description

Скилл для работы с немецким социальным правом (Sozialrecht). Генерирует документы: Widerspruch, Klage, Stellungnahme, Antrag. Анализирует нарушения (Kostenschieberei, Fristversäumnis, Zuständigkeitskonflikt). Знает SGB IX, XII, V, II, III. База знаний: 1105 диалогов по социальному праву Германии.

## Правовая база

| Закон | Область |
|-------|---------|
| SGB IX | Rehabilitation und Teilhabe behinderter Menschen |
| SGB XII | Sozialhilfe (Grundsicherung, Eingliederungshilfe) |
| SGB V | Gesetzliche Krankenversicherung |
| SGB II | Grundsicherung für Arbeitsuchende (Hartz IV/Bürgergeld) |
| SGB III | Arbeitsförderung (Agentur für Arbeit) |
| SGB XI | Soziale Pflegeversicherung |
| VwGO | Verwaltungsgerichtsordnung (Klageverfahren) |
| BGG | Behindertengleichstellungsgesetz |

## Capabilities

| Команда | Описание |
|---------|----------|
| `widerspruch gegen [Behörde]: [Beschreibung]` | Составить Widerspruch |
| `klage gegen [Behörde]: [Beschreibung]` | Составить Verwaltungsklage |
| `stellungnahme zu [Thema]` | Stellungnahme / ärztliche Stellungnahme |
| `antrag persönliches budget` | Antrag auf Persönliches Budget (SGB IX §29) |
| `kostenschieberei analysieren: [Beschreibung]` | Выявить незаконный перекладывание costs |
| `frist prüfen: [Datum des Bescheids]` | Проверить сроки Widerspruch/Klage |
| `zuständigkeit klären: [Leistung]` | Определить компетентный орган |
| `eingliederungshilfe antrag` | Antrag Eingliederungshilfe (SGB IX §99+) |
| `pflegegrad widerspruch` | Widerspruch gegen Pflegegradeinstufung |
| `analyse bescheid: [Text]` | Разбор административного решения |

## Шаблоны документов

### 1. Widerspruch (Общий шаблон)

```
[Ваше имя]
[Адрес]
[Дата]

An: [Behörde Name]
[Behörde Adresse]

Widerspruch gegen den Bescheid vom [Datum] — Az.: [Aktenzeichen]

Sehr geehrte Damen und Herren,

gegen den Bescheid vom [Datum], zugegangen am [Zustellungsdatum],
lege ich hiermit fristgemäß Widerspruch ein.

Begründung:
[BEGRÜNDUNG — конкретные нарушения, пункты закона]

Ich beantrage:
1. Den angefochtenen Bescheid aufzuheben.
2. [Конкретное требование — Leistung gewähren / Kosten zu übernehmen etc.]

Gemäß § [Paragraph] SGB [Nummer] habe ich Anspruch auf [Leistung], weil
[Rechtsgrundlage].

Mit freundlichen Grüßen,
[Unterschrift]

Anlagen:
- Kopie des Bescheids
- [Weitere Belege]
```

### 2. Widerspruch — SGB IX Rehabilitation

```
Widerspruch gegen den Bescheid vom [Datum]
Ablehnung von Rehabilitationsleistungen — Az.: [Aktenzeichen]

Begründung:

1. Gemäß § 14 SGB IX bin ich als zuständiger Reha-Träger zu betrachten.
   Der Bescheid verletzt § 14 Abs. 2 SGB IX, da [Behörde] die Weiterleitung
   an den zuständigen Träger versäumt hat.

2. Nach § 4 SGB IX haben behinderte Menschen Anspruch auf Leistungen zur
   Teilhabe, um ihre Selbstbestimmung und gleichberechtigte Teilhabe am
   gesellschaftlichen Leben zu ermöglichen.

3. Die abgelehnte Leistung ([Leistung]) ist gemäß § [§] SGB IX / XII
   erforderlich für: [medizinische / berufliche / soziale Begründung].

Ich beantrage die Aufhebung des Bescheids und Bewilligung von [Leistung]
innerhalb der Frist des § 14 Abs. 2 SGB IX (2 Wochen / 3 Wochen nach
Gutachten).
```

### 3. Widerspruch — Pflegegrad

```
Widerspruch gegen die Pflegegradeinstufung
Bescheid vom [Datum] — Pflegegrad [1/2/3] — Az.: [AZ]

Begründung:

Der festgestellte Pflegegrad [X] entspricht nicht dem tatsächlichen
Hilfebedarf gemäß § 15 SGB XI.

Tatsächlicher Hilfebedarf:
- Mobilität: [Beschreibung — z.B. kann nicht alleine aufstehen]
- Kognitive Fähigkeiten: [Beschreibung]
- Selbstversorgung: [Beschreibung — Dauer in Minuten]
- Gestaltung des Alltagslebens: [Beschreibung]

Nach NBA (Neues Begutachtungsassessment) ergibt sich ein Gesamtpunktwert
von mindestens [X] Punkten, was Pflegegrad [Y] entspricht.

Ich beantrage ein Obergutachten beim MDK / MEDICPROOF.
```

### 4. Antrag Persönliches Budget (SGB IX §29)

```
Antrag auf Persönliches Budget gemäß § 29 SGB IX

An: [Zuständiger Reha-Träger]

Hiermit beantrage ich die Gewährung von Leistungen zur Teilhabe in Form
eines Persönlichen Budgets gemäß § 29 SGB IX.

Gewünschte Leistungen:
1. [Leistung 1] — geschätzter Betrag: [X] €/Monat
2. [Leistung 2] — geschätzter Betrag: [X] €/Monat

Begründung:
Das Persönliche Budget ermöglicht mir [Selbstbestimmung / Flexibilität /
bessere Bedarfsdeckung], weil [konkrete Begründung].

Bisherige Sachleistungen waren unzureichend, weil [Begründung].

Ich bin bereit, eine Zielvereinbarung gemäß § 29 Abs. 4 SGB IX
abzuschließen.

Bitte teilen Sie mir den Beratungstermin gemäß § 29 Abs. 2 SGB IX
(Trägerkonferenz) mit.
```

### 5. Klage (Verwaltungsgericht)

```
An das Verwaltungsgericht [Stadt]

Klage

des/der [Name, Adresse] — Kläger/in —

gegen [Behörde, Adresse] — Beklagte/r —

wegen: Ablehnung von [Leistung] (§ [§] SGB [X])

KLAGEANTRAG:
1. Der Bescheid vom [Datum] in Gestalt des Widerspruchsbescheids
   vom [Datum] wird aufgehoben.
2. Der Beklagte wird verpflichtet, [Leistung] zu bewilligen.
3. Die Kosten des Verfahrens trägt der Beklagte.

BEGRÜNDUNG:

I. Sachverhalt
[Chronologische Darstellung]

II. Rechtliche Würdigung
Gemäß § [§] SGB [X] besteht Anspruch auf [Leistung], weil:
1. [Tatbestandsmerkmal 1 erfüllt]
2. [Tatbestandsmerkmal 2 erfüllt]

III. Verletzung von Verfahrensrechten
[Falls zutreffend: § 14 SGB IX — Fristversäumnis, § 17 SGB I — Beratungspflicht etc.]

Beweis: [Arztbriefe, Gutachten, Schriftverkehr]

[Unterschrift, Datum]
```

### 6. Kostenschieberei-Analyse

```
Analyse: Unzulässige Kostenverlagerung (Kostenschieberei)

Bescheid: [Behörde A] lehnt ab mit Verweis auf [Behörde B]
Behörde B lehnt ab mit Verweis auf [Behörde A / C]

Rechtslage:
- § 14 SGB IX: Erstangegangener Träger ist ZUSTÄNDIG, auch wenn er
  eigentlich nicht zuständig wäre — er muss Leistung erbringen und
  Regress beim eigentlich Zuständigen nehmen.
- § 43 SGB I: Vorläufige Leistungspflicht bei ungeklärter Zuständigkeit
- § 102 SGB X: Erstattungsanspruch zwischen Trägern

Empfehlung:
1. Widerspruch bei [Erstantragsträger] mit Berufung auf § 14 SGB IX
2. Bei Widerspruchsablehnung: Klage + Antrag auf einstweiligen Rechtsschutz
   (§ 86b SGG) wenn Dringlichkeit vorliegt
3. Beschwerde beim Landesbeauftragten für Behinderte (falls Bayern/NRW etc.)
```

## Типичные ситуации и подходы

### Ablehnung Eingliederungshilfe
- Grundlage: SGB IX §99+ (ab 2020), früher SGB XII
- Zuständig: Landschaftsverband / Bezirk / Landkreis
- Widerspruchsfrist: 1 Monat ab Zustellung
- Klagefrist: 1 Monat ab Widerspruchsbescheid
- Sozialgericht (nicht Verwaltungsgericht!) → § 51 SGG

### Hilfsmittel-Ablehnung (Krankenkasse)
- Grundlage: SGB V §33
- Widerspruch an Krankenkasse
- Bei Ablehnung: Sozialgericht
- MDK-Gutachten anfordern (§ 275 SGB V)

### Grundsicherung / Bürgergeld-Kürzungen
- SGB II: Jobcenter, Widerspruch innerhalb 1 Monat
- SGB XII: Sozialamt, Widerspruch innerhalb 1 Monat
- Aufschiebende Wirkung: § 86a SGG beachten

### Persönliches Budget-Konflikte
- § 29 SGB IX: Träger muss innerhalb 2 Wochen Trägerkonferenz einberufen
- Schweigen = ablehnender Bescheid nach 6 Wochen → Widerspruch möglich
- Kombination verschiedener Träger möglich (KK + Eingliederungshilfe)

## Fristen-Übersicht

| Verfahren | Frist | Rechtsgrundlage |
|-----------|-------|-----------------|
| Widerspruch | 1 Monat ab Zustellung | § 84 SGG / § 70 VwGO |
| Klage (Sozialgericht) | 1 Monat ab Widerspruchsbescheid | § 87 SGG |
| Klage (Verwaltungsgericht) | 1 Monat | § 74 VwGO |
| Untätigkeitsklage | 6 Monate Wartezeit | § 88 SGG |
| Reha-Träger Entscheidung | 3 Wochen (mit Gutachten: 5 Wo.) | § 14 SGB IX |
| Einstweiliger Rechtsschutz | Jederzeit bei Dringlichkeit | § 86b SGG |

## Zuständigkeits-Matrix

| Leistung | Träger | Rechtsgrundlage |
|----------|--------|-----------------|
| Krankenbehandlung | Krankenkasse (GKV) | SGB V |
| Pflege | Pflegekasse | SGB XI |
| Eingliederungshilfe | Bezirk / Landschaftsverband | SGB IX §99+ |
| Bürgergeld | Jobcenter | SGB II |
| Sozialhilfe | Sozialamt / Landkreis | SGB XII |
| Berufliche Reha | Rentenversicherung / AA | SGB IX + VI + III |
| Hilfsmittel | KK (med.) / Träger (Reha) | SGB V §33 / SGB IX |
| Persönliches Budget | Koordinierender Träger | SGB IX §29 |

## Wichtige Urteile / Grundsätze

- **BSG B 3 KR 6/14 R**: KK muss Hilfsmittel auch für Freizeit genehmigen
- **BSG B 8 SO 7/14 R**: Eingliederungshilfe hat Vorrang vor Sozialhilfe
- **§ 14 SGB IX**: "Jede-Geht-Vor"-Prinzip — Erstantrag bindet Träger
- **§ 17 SGB I**: Beratungspflicht der Behörde — Verletzung → Schadensersatz
- **§ 115 SGB X**: Vorleistungspflicht bei ungeklärter Zuständigkeit

## Usage Examples

```
Пользователь: widerspruch gegen Jobcenter: haben mir Bürgergeld ohne Begründung gekürzt
Агент: [генерирует Widerspruch с § 24 SGB II, § 31 SGB II, Begründungspflicht]

Пользователь: kostenschieberei analysieren: KK verweist auf Eingliederungshilfe,
             Landschaftsverband verweist zurück auf KK für Rollstuhl-Reparatur
Агент: [анализирует ситуацию по § 14 SGB IX, рекомендует первый орган-адресат]

Пользователь: frist prüfen: Bescheid vom 01.02.2026
Агент: Widerspruchsfrist läuft bis 01.03.2026 (1 Monat, § 84 SGG).
       Heute ist 28.03.2026 — Frist ABGELAUFEN.
       Möglichkeiten: Wiedereinsetzung (§ 67 SGG) wenn unverschuldete Hinderung vorlag.

Пользователь: antrag persönliches budget
Агент: [генерирует полный Antrag по § 29 SGB IX с инструкцией по Trägerkonferenz]
```

## Disclaimer

Diese Skill-Ausgaben sind Vorlagen und allgemeine rechtliche Informationen.
Sie ersetzen keine Rechtsberatung. Bei komplexen Fällen:
- VdK Deutschland: vdk.de
- Sozialverband Deutschland (SoVD): sovd.de
- VfL (Verein für Lebenshilfe) für SGB IX-Fragen
- Rechtsantragstelle am Sozialgericht (kostenlos)

## Source & Knowledge Base

Wissensgrundlage: 1105 Dialoge aus data70-Archiv (svend4/data70),
davon 218 zum Thema Sozialrecht (9.4 MB). Indiziert via InfoM GraphRAG.

Verwandter Skill: `infom-graphrag` — für eigene Dokument-Analyse
