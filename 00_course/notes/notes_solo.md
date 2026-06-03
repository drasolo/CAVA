N:Debug proiect la setup✅

Q: ce e un gradient mai exact(al unei imagini,in general)?
A: ceva o formula din articol

Q: ce e Sobel, ce face?

N: de inv sintaxa si de facu t ex pregatitoare pentru proiect

CURS 5
Q:Edgels???
❕N:Transformata Hough

parametrizarea y=mx+b no bueno bc nu poate repr linii verticale (better: ax+by+c=0)

Q:de ce geometrie in a gasii linii intr-o imagine(detectare de linii isn't enough??)

Q:Discretizare??? tf (gen tabel pentru voturi)

N:Extensii de repetat +gradientul de la acest capitol

Salut! Iată un sumar detaliat al ideilor principale din cursul tău, structurat pentru a te ajuta să înveți.

### 📚 Informații Generale Despre Curs

- [cite_start]**Numele Cursului:** Concepte și aplicații în Vederea Artificială[cite: 1].
- [cite_start]**Profesori:** Bogdan Alexe [cite: 2, 5] [cite_start]și Radu Ionescu[cite: 2, 6].
- [cite_start]**Perioada:** Anii III/IV, semestrul I, 2024-2025[cite: 4].
- [cite_start]**Subiecte Anterioare:** Cursul anterior a acoperit aplicații ale filtrelor (extragerea muchiilor) [cite: 7][cite_start], detectarea liniilor cu transformata Hough [cite: 8, 11] [cite_start]și RANSAC [cite: 13][cite_start], precum și compararea contururilor[cite: 14].

---

### 1. Transformata Hough pentru Detectarea Liniilor

Transformata Hough este o tehnică de extragere a caracteristicilor utilizată pentru a găsi forme geometrice (cum ar fi linii sau cercuri) într-o imagine, folosind un sistem de votare.

#### A. Principiul de Bază: Dualitatea Punct-Linie

- **Parametrizarea $y = mx + b$:**
  - [cite_start]O dreaptă în planul imaginii este definită de ecuația $y = m \cdot x + b$ [cite: 18][cite_start], unde $m$ este panta [cite: 19] [cite_start]și $b$ este deplasarea[cite: 20]. [cite_start]Aceștia sunt parametrii dreptei[cite: 21].
  - [cite_start]**Spațiul Hough:** Este un spațiu al parametrilor (de exemplu, un plan $m-b$)[cite: 36, 37].
  - **Dualitatea:**
    1.  [cite_start]O **dreaptă** din spațiul imaginii corespunde unui **singur punct** $(m, b)$ în spațiul Hough[cite: 44, 53, 64].
    2.  [cite_start]Un **punct** $(x_0, y_0)$ din spațiul imaginii corespunde unei **drepte** în spațiul Hough[cite: 52, 63]. [cite_start]Prin punctul $(x_0, y_0)$ trec o infinitate de linii [cite: 55][cite_start], care în spațiul Hough se află pe dreapta $b = -x_0 \cdot m + y_0$[cite: 67].
- [cite_start]**Găsirea liniei:** Linia care trece prin punctele $(x_0, y_0)$ și $(x_1, y_1)$ [cite: 69] [cite_start]corespunde în spațiul Hough punctului de intersecție al dreptelor $b = -x_0m + y_0$ [cite: 70] [cite_start]și $b = -x_1m + y_1$[cite: 71, 73].

#### B. Problema Parametrizării $(m, b)$

- [cite_start]Această reprezentare ($y = mx + b$) are o problemă majoră: panta $m$ poate lua o valoare infinită pentru liniile verticale[cite: 105].

#### C. Parametrizarea Polară (Soluția)

- [cite_start]Pentru a rezolva problema liniilor verticale, se folosește o altă parametrizare, anume coordonatele polare $(\rho, \theta)$[cite: 106, 112].
- [cite_start]**Ecuația:** $x \cdot \cos(\theta) + y \cdot \sin(\theta) = \rho$ [cite: 119] (ecuație dedusă din algoritm).
  - [cite_start]$\rho$ (rho): distanța perpendiculară de la originea $[0,0]$ la linie[cite: 101].
  - [cite_start]$\theta$ (theta): unghiul dintre axa Ox și acea perpendiculară[cite: 102].
- **Dualitatea (în coordonate polare):**
  - [cite_start]Un **punct** $(x, y)$ din spațiul imaginii corespunde unei **curbe sinusoidale** în spațiul Hough $(\rho, \theta)$[cite: 103].
  - [cite_start]Punctele care se află pe aceeași linie în imagine vor genera curbe sinusoidale care se intersectează într-un singur punct $(\rho, \theta)$ în spațiul Hough[cite: 110].

#### D. Algoritmul Hough (cu Votare)

[cite_start]Algoritmul funcționează prin acumularea de voturi într-un spațiu discret al parametrilor[cite: 90].

1.  [cite_start]**Inițializare:** Se creează un acumulator 2D (o matrice) $H[d, \theta]$ și se inițializează cu 0[cite: 114].
2.  [cite_start]**Votare:** Pentru fiecare punct $(x, y)$ detectat ca muchie (edgel) în imaginea $I$[cite: 89, 115]:
    - [cite_start]Se iterează $\theta$ de la 0 la 360 de grade (în intervale discrete)[cite: 116].
    - [cite_start]Se calculează $d = x \cdot \cos(\theta) + y \cdot \sin(\theta)$[cite: 119].
    - [cite_start]Se incrementează celula corespunzătoare din acumulator: $H[d, \theta] += 1$[cite: 117].
3.  [cite_start]**Găsirea Maximului:** Se caută valorile $(d^*, \theta^*)$ pentru care $H[d^*, \theta^*]$ este maxim[cite: 118].
4.  [cite_start]**Rezultat:** Perechile $(d^*, \theta^*)$ cu cele mai multe voturi (cele mai "luminoase" puncte din spațiul Hough [cite: 134][cite_start]) reprezintă liniile detectate în imagine[cite: 118, 119].

#### E. Extensii ale Algoritmului Hough

- [cite_start]**Extensia 1 (Folosirea Gradientului):** În loc să se itereze prin toate valorile posibile ale lui $\theta$ [cite: 116][cite_start], se folosește direct orientarea gradientului imaginii în punctul $(x, y)$ pentru a determina $\theta$[cite: 162, 165, 174]. Acest lucru reduce numărul de calcule.
- [cite_start]**Extensia 2 (Vot Ponderat):** În loc de a incrementa cu 1 ($H[d, \theta] += 1$), se incrementează cu magnitudinea gradientului în punctul $(x, y)$[cite: 179]. Astfel, muchiile mai puternice contribuie cu mai multe voturi.

---

### 2. Transformata Hough pentru Cercuri

[cite_start]Principiul votării poate fi extins și la alte forme, cum ar fi cercurile[cite: 184].

- **Ecuația Cercului:** $(x - a)^2 + (y - b)^2 = r^2$. [cite_start]Parametrii sunt centrul $(a, b)$ și raza $r$[cite: 186, 189, 191].
- [cite_start]**Caz 1: Raza $r$ este fixă (cunoscută)** [cite: 185, 190]
  - [cite_start]Spațiul Hough este 2D, având parametrii $(a, b)$[cite: 188, 193].
  - Un punct $(x, y)$ din imaginea de muchii "votează" pentru toate centrele posibile $(a, b)$ care se află la distanța $r$ de el. [cite_start]Aceste voturi formează un cerc în spațiul Hough[cite: 193].
  - [cite_start]Intersecția mai multor astfel de cercuri în spațiul Hough indică centrul $(a, b)$ al cercului din imagine[cite: 194].
- [cite_start]**Caz 2: Raza $r$ este necunoscută** [cite: 196, 205]
  - [cite_start]Spațiul Hough devine 3D, având parametrii $(a, b, r)$[cite: 207, 210, 211].
  - [cite_start]Un punct $(x, y)$ din imagine votează pentru un **con** în spațiul Hough 3D[cite: 212, 213].
  - [cite_start]Intersecția mai multor conuri [cite: 217] indică parametrii $(a, b, r)$ ai cercului detectat.
- **Optimizare (cu gradient):** Dacă se cunoaște direcția gradientului în punctul $(x, y)$, se știe că centrul cercului trebuie să se afle pe linia definită de gradient. [cite_start]Acest lucru reduce spațiul de votare 3D (con) la o linie 2D[cite: 218].

---

### 3. Transformata Hough Generalizată

[cite_start]Această metodă permite detectarea **formelor arbitrare**, nu doar a celor analitice (linii, cercuri)[cite: 239].

- [cite_start]**Principiul:** Se bazează pe "cuvinte vizuale" (patch-uri sau șabloane mici) învățate dintr-o imagine de antrenare[cite: 240, 242].
- [cite_start]**Antrenare:** Pentru fiecare "cuvânt vizual" dintr-o formă de antrenare (de ex., o roată de mașină), se stochează un vector de deplasare de la acel "cuvânt" la centrul obiectului[cite: 243].
- [cite_start]**Detecție:** Într-o imagine de test [cite: 248][cite_start], atunci când este detectat un "cuvânt vizual", acesta "votează" pentru o locație a centrului obiectului, folosind vectorul său de deplasare stocat[cite: 246]. Locațiile care primesc cele mai multe voturi indică prezența și centrul obiectului.

---

### 4. RANSAC (RANdom SAmple Consensus)

[cite_start]RANSAC este un algoritm robust folosit pentru a estima parametrii unui model dintr-un set de date care conține _outlieri_ (puncte care nu aparțin modelului, de ex., zgomot)[cite: 251, 304].

#### A. Ideea Principală

- [cite_start]Scopul este de a evita impactul punctelor _outlier_[cite: 305].
- [cite_start]Algoritmul se concentrează pe găsirea unui subset de date care sunt _inlieri_ (puncte care se potrivesc bine modelului)[cite: 305].
- [cite_start]**Intuiția:** Dacă un model (de ex., o dreaptă) este construit folosind un punct _outlier_, foarte puține alte puncte vor susține acest model (va avea un număr mic de _inlieri_)[cite: 306].

#### B. Algoritmul RANSAC (pentru linii)

[cite_start]Algoritmul se repetă de $N$ ori[cite: 308]:

1.  [cite_start]**Selectare Aleatoare:** Se selectează aleator un număr minim de puncte necesare pentru a defini modelul (de ex., $k=2$ puncte pentru o dreaptă)[cite: 258, 309].
2.  [cite_start]**Construire Model:** Se construiește o ipoteză de model (o dreaptă) care trece prin aceste $k$ puncte[cite: 262, 310].
3.  [cite_start]**Numărare Inlieri:** Se numără câte alte puncte din setul de date sunt _inlieri_ pentru această ipoteză[cite: 273, 301]. [cite_start]Un punct este considerat _inlier_ dacă distanța sa față de dreaptă este mai mică decât un prag (threshold) $t$[cite: 311].
4.  [cite_start]**Validare:** Dacă numărul de _inlieri_ depășește un prag $d$, ipoteza este acceptată[cite: 312].
5.  [cite_start]**Păstrarea celui mai bun:** Se păstrează ipoteza (dreapta) care a avut cel mai mare număr de _inlieri_ de-a lungul celor $N$ iterații[cite: 313].
6.  [cite_start]**(Opțional) Rafinare:** Modelul final este re-estimat folosind _toate_ punctele _inlier_ găsite pentru cea mai bună ipoteză[cite: 312].

#### C. Avantaje și Dezavantaje

- **Avantaje:**
  - [cite_start]Este simplu și general[cite: 317].
  - [cite_start]Se aplică în multe probleme (detectare de linii, calcul de homografie pentru panorame etc.)[cite: 318].
  - [cite_start]Funcționează bine în practică[cite: 319].
- **Dezavantaje:**
  - [cite_start]Nu funcționează bine când raportul inlier/outlier este foarte mic[cite: 322].
  - [cite_start]Poate fi problematic dacă modelul are foarte mulți parametri[cite: 321].

#### D. Alegerea Numărului de Itarații ($N$)

- [cite_start]Trebuie să alegem $N$ suficient de mare pentru a avea o probabilitate mare (de ex., $p=0.99$) că am selectat cel puțin o dată un set de puncte format doar din _inlieri_[cite: 325, 332].
- [cite_start]Fie $r$ = procentul de puncte _inlier_ (de ex., 0.5) și $k$ = numărul de puncte necesare (de ex., 2 pentru o dreaptă)[cite: 326, 327].
- [cite_start]Probabilitatea ca o selecție de $k$ puncte să fie corectă (toți _inlieri_) este $r^k$[cite: 328].
- [cite_start]Probabilitatea ca toate cele $N$ selecții să fie greșite este $(1 - r^k)^N$[cite: 330].
- [cite_start]Probabilitatea ca cel puțin o selecție să fie corectă este $p = 1 - (1 - r^k)^N$[cite: 331].
- [cite_start]Se alege $N$ astfel încât $p$ să fie apropiat de 1 (de ex., 0.99)[cite: 332].

---

### 5. Compararea Contururilor (Formelor)

#### A. Distanța Chamfer

- [cite_start]Este o metodă de a măsura similaritatea dintre două forme (contururi)[cite: 433, 435].
- [cite_start]Măsoară distanța medie dintre punctele unui șablon (template) $T$ și cea mai apropiată mulțime de puncte (de ex., muchiile) dintr-o imagine $I$[cite: 438].
- [cite_start]**Formula:** $D_{Chamfer}(T, I) = \frac{1}{|T|} \sum_{t \in T} \min_{i \in I} \text{dist}(t, i)$[cite: 438, 445, 446].
  - [cite_start]$|T|$ este numărul de puncte din șablon[cite: 445].
  - [cite_start]$\min_{i \in I} \text{dist}(t, i)$ este distanța de la un punct $t$ din șablon la cel mai apropiat punct $i$ din imagine[cite: 446].
  - [cite_start]Distanța poate fi Euclidiană sau Manhattan[cite: 447, 449].
- [cite_start]**Implementare Naivă:** O implementare naivă este foarte lentă (complexitate $\approx |T| \cdot |I|^2$)[cite: 456, 463].

#### B. Transformata Distanță (Distance Transform - TD)

- [cite_start]**Definiție:** Este o tehnică de optimizare pentru Distanța Chamfer[cite: 531]. [cite_start]Este o funcție care, pentru fiecare pixel $p$ dintr-o imagine, asignează distanța de la $p$ la cel mai apropiat punct de interes (de ex., cel mai apropiat pixel de muchie)[cite: 530, 538].
- **Utilizare:**
  1.  [cite_start]Se calculează o singură dată Transformata Distanță pentru imaginea de muchii $I$[cite: 542]. Imaginea rezultată $TD_I$ va avea la fiecare pixel $(x, y)$ valoarea distanței până la cea mai apropiată muchie.
  2.  Pentru a calcula Distanța Chamfer, se suprapune șablonul $T$ peste $TD_I$.
  3.  [cite_start]Pentru fiecare punct $t$ din șablon, se citește direct valoarea de sub el din $TD_I$ (aceasta este $\min \text{dist}(t, i)$)[cite: 542].
  4.  Se face media acestor valori. Acest proces este mult mai rapid.

---

### 6. Aplicații Prezentate

- [cite_start]**Detectarea liniilor:** Găsirea liniilor în imagini cu clădiri [cite: 141] [cite_start]sau terenuri de sport[cite: 143].
- [cite_start]**Detectarea cercurilor:** Găsirea monedelor într-o imagine[cite: 227, 234].
- [cite_start]**Rezolvarea Sudoku:** Extragerea informațiilor dintr-un careu de Sudoku[cite: 403], incluzând:
  - [cite_start]Aplicarea de filtre (Gaussian blur, eroziune)[cite: 420].
  - [cite_start]Detecția muchiilor cu Canny[cite: 421].
  - [cite_start]Găsirea celui mai mare contur (grila)[cite: 422].
  - [cite_start]Corectarea perspectivei și detectarea liniilor grilei[cite: 423, 427].
  - [cite_start]Clasificarea celulelor (goale vs. pline) [cite: 424, 429] [cite_start]și recunoașterea cifrelor[cite: 430].

Sper că acest sumar detaliat îți este de folos!

Poți să îmi ceri să detaliez oricare dintre aceste concepte, de exemplu, cum funcționează exact algoritmul RANSAC pentru calculul homografiei?

### intrebari si tips&tricks proiect

Q: cum potrivim figurile care se pot afla mai pe la colturi si margini pe un tile
sfat: de lucrat la template-uri considerabil
sfat: template matching e o tehnica de cautare...
outputul este doar fisierul cu structura stabilita
!!!de scos bine tabla, gen piesele unde

deci to figure out what could go wrong and to build around it and the basic challenges that would come with variations in the images(different light/ tiles not being perfectly in their place (), and )
