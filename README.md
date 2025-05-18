# Etapa I: Definirea problemei.

## Ce se dă și ce se cere?

Se dau mai multe fotografii cu diverse afecțiuni dermatologice al feței (riduri, puncte negre, acnee etc.).  
Pe baza acestora trebuie dezvoltat un sistem inteligent care să permită identificarea problemelor cosmetice ale tenului și generarea de recomandări personalizate pentru îngrijirea pielii,
bazate pe analiza imaginilor.

Sistemul ar trebui să fie antrenat pe un set variat de date pentru a putea "învăța" să facă distincții precise între problemele pielii.
De asemenea, este important ca soluția să fie folosită doar pentru scopuri cosmetice și nu ca înlocuitor pentru un consult dermatologic.

## De ce e nevoie de AI pentru a rezolva problema?

### <ins>2. Capacitatea de detecție și segmentarea de înaltă precizie</ins>

AI-ul este capabil să:

- **detecteze automat** zonele afectate de riduri, acnee, puncte negre, pete, pori dilatați etc.;
- **segmenteze precis** zonele problematice din imagine (ex.: conturul exact al unui rid sau al unei leziuni acneice);
- **cuantifice problemele** (ex.: numărul de leziuni, adâncimea aparentă a ridurilor, distribuția punctelor negre etc.).

  Aceste sarcini nu sunt chiar așa de ușor de realizat nici pentru un dermatolog, dar AI poate oferi aceste informații în mod automat și constant.
  <br> <br>

# Etapa II: Analiza datelor de intrare

## Câte date avem?

Avem un total de 1 000 de imagini distribuite uniform în 4 categorii relevante pentru analiza tenului: **acens**, **blackheads**, **wrinkles** și **darkspots**, fiecare conținând câte 250 de imagini. Această distribuție echilibrată este ideală pentru antrenarea unui model de clasificare sau segmentare, deoarece reduce riscul de bias față de o anumită clasă. Fiecare imagine este presupusă a reprezenta clar o singură categorie de problemă cosmetică, ceea ce contribuie la o etichetare coerentă și la o acuratețe mai bună în procesul de antrenare. În funcție de complexitatea modelului și a sarcinii, acest set poate fi extins ulterior pentru a crește generalizarea.

## Ce distribuție au datele?

Distribuția datelor din setul de imagini este uniformă pentru cele patru categorii: **acens**, **blackheads**, **wrinkles** și **darkspots**. Această distribuție echilibrată este importantă pentru antrenarea unui model de învățare automată, deoarece previne supra-reprezentarea unei clase și permite modelului să învețe mai corect diferențele subtile dintre tipurile de imperfecțiuni cutanate.
