## Structura folderului de date

Datele pentru acest proiect sunt organizate astfel:

```
data/
├── acnes/        # 250 imagini cu acnee
├── blackheads/   # 250 imagini cu puncte negre
├── wrinkles/     # 250 imagini cu riduri
└── darkspots/    # 250 imagini cu pete pigmentare
```

Notă: Folderul de date (`data/`) este inclus în `.gitignore` și nu este versionat. Contribuitorii trebuie să obțină setul de date separat.

# Etapa I: Definirea problemei.

## Ce se dă și ce se cere?

Se dau mai multe fotografii cu diverse afecțiuni dermatologice al feței (riduri, puncte negre, acnee etc.).  
Pe baza acestora trebuie dezvoltat un sistem inteligent care să permită identificarea problemelor cosmetice ale tenului și generarea de recomandări personalizate pentru îngrijirea pielii,
bazate pe analiza imaginilor.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b.	Algoritmi de AI pentru a identifica problemele pielii <br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;c.	Generarea recomandărilor:<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**•**	Recomandări bazate pe tipul afecțiunii identificate<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**•**	Sugerarea unor schimbări în stilul de viață și alimentație


Sistemul ar trebui să fie antrenat pe un set variat de date pentru a putea "învăța" să facă distincții precise între problemele pielii.
De asemenea, este important ca soluția să fie folosită doar pentru scopuri cosmetice și nu ca înlocuitor pentru un consult dermatologic.

## De ce e nevoie de AI pentru a rezolva problema?

### <ins>1. Complexitatea vizuală a pielii umane</ins>

Pielea umană este un organ variabil și complex. Textura, culoarea, luminozitatea, tonul pielii și tipurile de leziuni (acnee, puncte negre, riduri etc.) variază considerabil de la o persoana la alta, în funcție de: **genetica individuală, vârsta, etnie, obiceiuri de îngrijire, expunerea la factori de mediu (soare, poluare, etc.)**.

Aceasta <ins>complexitate</ins> face dificilă aplicarea unor reguli sau a unor meotde clasice de analiză. În special, rețelele neuronale convoluționale (CNN) sunt capabile să învețe aceste variații din mii de imagini și să generalizeze corect cazuri noi.

### <ins>2. Capacitatea de detecție și segmentarea de înaltă precizie</ins>

AI-ul este capabil să:

- **detecteze automat** zonele afectate de riduri, acnee, puncte negre, pete, pori dilatați etc.;
- **segmenteze precis** zonele problematice din imagine (ex.: conturul exact al unui rid sau al unei leziuni acneice);
- **cuantifice problemele** (ex.: numărul de leziuni, adâncimea aparentă a ridurilor, distribuția punctelor negre etc.).

  Aceste sarcini nu sunt chiar așa de ușor de realizat nici pentru un dermatolog, dar AI poate oferi aceste informații în mod automat și constant.
  <br> <br>

### <ins>4.	Reducerea erorilor și a subiectivității </ins>
  În diagnosticarea tradițională, există un grad mare de **subiectivitate** – doi dermatologi pot oferi opinii diferite iar unele probleme pot fi omise vizual în graba consultației. AI aduce **consistență, obiectivitate și acuratețe**, deoarece utilizează aceiași algoritmi  și criterii pentru toate cazurile analizate.

### <ins>5.	Integrare ușoară în produse comerciale </ins>
  Tehnologia AI poate fi integrată în aplicații mobile de beauty & health, oglinzi inteligente, aparatură de diagnostic la domiciliu, produse cosmetice personalizare (cu scanare facială). Aceasta oferă un avantaj competitiv companiilor din domeniul cosmetic și wellness.

# Etapa II: Analiza datelor de intrare

## Ce tip de date avem?

Pentru antrenarea sistemului de detecție a problemelor cosmetice utilizăm un set de date ce conține imagini color impărțite în 4 categorii specifice(acnee, riduri, puncte negre, pete pigmentare). Imaginile faciale redau fie întreaga față, fie doar o porțiune. 

## Câte date avem?

Avem un total de 1 000 de imagini distribuite uniform în 4 categorii relevante pentru analiza tenului: **acens**, **blackheads**, **wrinkles** și **darkspots**, fiecare conținând câte 250 de imagini. Această distribuție echilibrată este ideală pentru antrenarea unui model de clasificare sau segmentare, deoarece reduce riscul de bias față de o anumită clasă. Fiecare imagine este presupusă a reprezenta clar o singură categorie de problemă cosmetică, ceea ce contribuie la o etichetare coerentă și la o acuratețe mai bună în procesul de antrenare. În funcție de complexitatea modelului și a sarcinii, acest set poate fi extins ulterior pentru a crește generalizarea.

## Ce distribuție au datele?

Distribuția datelor din setul de imagini este uniformă pentru cele patru categorii: **acens**, **blackheads**, **wrinkles** și **darkspots**. Această distribuție echilibrată este importantă pentru antrenarea unui model de învățare automată, deoarece previne supra-reprezentarea unei clase și permite modelului să învețe mai corect diferențele subtile dintre tipurile de imperfecțiuni cutanate.

În ceea ce privește dimensiunile imaginilor, există o variație semnificativă în termeni de înălțime și lățime. De asemenea, raportul de aspect al imaginilor variază, indicând un set divers de imagini care pot include atât imagini cu întreaga față, cât și porțiuni ale feței.
